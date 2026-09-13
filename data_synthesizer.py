# models/synthesis/data_synthesizer.py

import pandas as pd
import numpy as np
import torch
import time
import os
import json
import argparse
import sys
import logging
import traceback

from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Default random seed for reproducible synthesis (GaussianCopula path is fully deterministic).
DEFAULT_SEED = 42

# --- 1. LOGGING SETUP ---
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

def setup_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def run_synthesis(job_id, job_dir, details_file_path, log_file_path, seed=DEFAULT_SEED):
    logger = setup_logger(log_file_path)
    logger.info(f"--- Starting Synthesis for Job ID: {job_id} (seed={seed}) ---")

    # Reproducibility: fix NumPy and PyTorch RNGs before any stochastic step.
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
    except Exception:
        pass

    try:
        # --- 2. Load Configuration ---
        with open(details_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        real_data_path = config['original_file_path']
        na_values = ['-', 'NA', 'N/A', 'n/a', 'na', 'NaN', 'nan', '', ' ', 'null', 'NULL', 'None', 'none', '#N/A', '#NA']

        # Robust CSV Loading
        encodings_to_try = ['utf-8-sig', 'cp1252', 'latin1']
        df_real = None

        for encoding in encodings_to_try:
            try:
                df_real = pd.read_csv(real_data_path, encoding=encoding, na_values=na_values, sep=None, engine='python')
                if df_real.shape[1] == 1 and df_real.shape[0] > 0:
                     first_val = str(df_real.iloc[0, 0])
                     if ',' in first_val or ',' in str(df_real.columns[0]):
                         df_real = pd.read_csv(real_data_path, encoding=encoding, na_values=na_values, sep=',')
                break
            except Exception:
                continue

        if df_real is None:
            raise ValueError(f"Could not decode the CSV file.")

        logger.info(f"Loaded real data. Shape: {df_real.shape}")

        # --- 3. Prepare Data (Include Input and Output columns for Joint Distribution P(X,Y)) ---
        columns_for_synthesis = [c['name'] for c in config['column_configs'] if c.get('column_role') in ['Input', 'Output', 'Feature', 'Target']]
        if not columns_for_synthesis:
            columns_for_synthesis = list(df_real.columns)
        column_config_map = {c['name']: c for c in config['column_configs']}

        missing_columns = [col for col in columns_for_synthesis if col not in df_real.columns]
        if missing_columns:
            raise KeyError(f"CRITICAL: Missing columns: {missing_columns}")

        df_for_synthesis = df_real[columns_for_synthesis].copy()
        df_for_synthesis = df_for_synthesis.dropna()

        if len(df_for_synthesis) == 0:
            raise ValueError("No valid data remaining after removing rows with N/A values.")

        # --- PRE-PROCESSING & CLEANING ---
        for col_name in df_for_synthesis.columns:
            user_config = column_config_map.get(col_name, {})
            user_type = str(user_config.get('column_data_type', '')).lower()

            # 1. SKIP parsing ONLY for strictly categorical columns.
            if user_type in ['categorical']:
                continue

            # Clean Object -> Numeric (Only for continuous/numeric inputs)
            if df_for_synthesis[col_name].dtype == 'object':
                 df_for_synthesis[col_name] = pd.to_numeric(
                    df_for_synthesis[col_name].astype(str).str.replace(',', '').str.strip(),
                    errors='coerce'
                )

        # 2. NUMERICAL TYPE TRANSFORMATION (PRE-TRAINING)
        # Handle "Discrete -> Continuous" with Smart Adaptive Jitter
        for col_name in df_for_synthesis.columns:
            if col_name not in column_config_map: continue

            user_config = column_config_map[col_name]
            target_type = str(user_config.get('column_data_type', '')).lower()

            # ONLY apply jitter if the user explicitly wants Continuous/Float
            if target_type in ['float', 'numeric', 'continuous', 'decimal']:
                is_integer_like = np.all(np.mod(df_for_synthesis[col_name].dropna(), 1) == 0)
                if is_integer_like:
                    unique_vals = np.sort(df_for_synthesis[col_name].unique())
                    if len(unique_vals) > 1:
                        min_step = np.min(np.diff(unique_vals))
                        noise_scale = min_step * 0.4
                    else:
                        # Constant column: jitter would destroy the constant value -> skip
                        logger.info(f"Column '{col_name}': Skipping jitter (constant column).")
                        continue

                    logger.info(f"Column '{col_name}': Applying jitter for continuous output.")
                    noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(df_for_synthesis))
                    df_for_synthesis[col_name] = df_for_synthesis[col_name] + noise

        # --- 4. Metadata & Model Selection ---
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_for_synthesis)

        # 3. ENFORCE USER CONFIG IN METADATA
        for col_name in df_for_synthesis.columns:
            if col_name in column_config_map:
                cfg = column_config_map[col_name]
                ctype = str(cfg.get('column_data_type', '')).lower()

                if ctype in ['categorical']:
                    metadata.update_column(column_name=col_name, sdtype='categorical')
                elif ctype in ['continuous', 'float', 'numeric', 'discrete', 'int', 'integer']:
                    metadata.update_column(column_name=col_name, sdtype='numerical')

        # Force numeric treatment (only for columns the USER typed as numeric;
        # never override a user-marked categorical even if the dtype is numeric)
        for col_name in df_for_synthesis.columns:
            user_type = str(column_config_map.get(col_name, {}).get('column_data_type', '')).lower()
            if user_type in ['categorical']:
                continue
            if pd.api.types.is_numeric_dtype(df_for_synthesis[col_name]):
                metadata.update_column(column_name=col_name, sdtype='numerical')

        synthesis_parameters = config.get('synthesis_parameters', {})
        user_model_selection = config.get('model_selected', 'CTGAN')
        num_rows = len(df_for_synthesis)
        warnings_list = []

        if num_rows < 500:
            logger.warning(f"Small dataset ({num_rows} rows). Forcing GaussianCopula.")
            warnings_list.append(
                f"Dataset has {num_rows} rows (<500). Model selection was overridden to GaussianCopula for numerical stability."
            )
            synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        else:
            logger.info(f"Dataset > 500 rows. Using {user_model_selection}.")
            use_gpu = torch.cuda.is_available()
            if user_model_selection != 'CTGAN':
                logger.warning(f"Model '{user_model_selection}' not supported; using CopulaGAN instead.")
                warnings_list.append(
                    f"Model selection '{user_model_selection}' is not supported; CopulaGAN was used instead."
                )
                synthesizer_class = CopulaGANSynthesizer
            else:
                synthesizer_class = CTGANSynthesizer
            synthesizer = synthesizer_class(
                metadata=metadata,
                epochs=synthesis_parameters.get('epochs', 500),
                batch_size=synthesis_parameters.get('batch_size', 50),
                verbose=True,
                cuda=use_gpu
            )

        # --- 5. Training ---
        logger.info("Starting training...")
        np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
        except Exception:
            pass
        synthesizer.fit(df_for_synthesis)

        # --- 6. Generation with Deduplication & Range Enforcement ---
        target_samples = int(synthesis_parameters.get('num_samples', 100))
        logger.info(f"Generating unique samples until we hit target: {target_samples}...")

        # Initialize an empty DataFrame to hold unique results
        final_synthetic_data = pd.DataFrame()

        attempts = 0
        max_attempts = 20  # Increased attempts to handle high collision rates

        range_constraints = {}
        for col_name in df_for_synthesis.columns:
            if col_name in column_config_map:
                cfg = column_config_map[col_name]
                try:
                    min_val = float(cfg.get('min')) if cfg.get('min') not in [None, ''] else None
                    max_val = float(cfg.get('max')) if cfg.get('max') not in [None, ''] else None
                    if min_val is not None or max_val is not None:
                        range_constraints[col_name] = (min_val, max_val)
                except:
                    pass

        while len(final_synthetic_data) < target_samples and attempts < max_attempts:
            # How many more do we need? Oversample by 20% to account for duplicates/clipping issues
            needed = target_samples - len(final_synthetic_data)
            batch_size = int(max(needed * 1.2, 200)) # Minimum batch 200 to keep it moving

            logger.info(f"Attempt {attempts+1}: Generating batch of {batch_size} (Needed: {needed})")

            batch = synthesizer.sample(num_rows=batch_size)

            # --- Range Injection (The "Force" Method) ---
            for col, (min_v, max_v) in range_constraints.items():
                if col not in batch.columns or not pd.api.types.is_numeric_dtype(batch[col]):
                    continue

                if min_v is not None and max_v is not None:
                    current_min = batch[col].min()
                    current_max = batch[col].max()

                    requested_span = max_v - min_v
                    generated_span = current_max - current_min

                    col_cfg = column_config_map.get(col, {})
                    c_type = str(col_cfg.get('column_data_type', '')).lower()
                    is_discrete = c_type in ['int', 'integer', 'discrete']

                    if requested_span > 0 and (generated_span / requested_span) < 0.8:
                        mask = np.random.choice([True, False], size=len(batch), p=[0.5, 0.5])

                        if is_discrete:
                            random_vals = np.random.randint(int(min_v), int(max_v) + 1, size=mask.sum())
                        else:
                            random_vals = np.random.uniform(min_v, max_v, size=mask.sum())

                        batch.loc[mask, col] = random_vals

            # --- Clipping ---
            for col, (min_v, max_v) in range_constraints.items():
                if col not in batch.columns or not pd.api.types.is_numeric_dtype(batch[col]):
                    continue
                if min_v is not None:
                    batch[col] = batch[col].clip(lower=min_v)
                if max_v is not None:
                    batch[col] = batch[col].clip(upper=max_v)

            # --- CRITICAL: Rounding & Type Transformation INSIDE LOOP ---
            # We must round BEFORE deduplication, otherwise 10.1 and 10.2 count as unique
            # but eventually become the same integer '10'.
            for col_name in batch.columns:
                if col_name not in column_config_map: continue

                user_config = column_config_map[col_name]
                target_type = str(user_config.get('column_data_type', '')).lower()

                if target_type in ['int', 'integer', 'discrete']:
                    batch[col_name] = batch[col_name].round(0).astype(int)

            # --- Accumulate & Deduplicate ---
            final_synthetic_data = pd.concat([final_synthetic_data, batch], ignore_index=True)

            # Remove duplicates based on ALL columns
            before_dedup = len(final_synthetic_data)
            final_synthetic_data = final_synthetic_data.drop_duplicates()
            after_dedup = len(final_synthetic_data)

            logger.info(f"Dropped {before_dedup - after_dedup} duplicates. Current count: {after_dedup}/{target_samples}")

            attempts += 1

        # Final Cleanup
        final_synthetic_data = final_synthetic_data.iloc[:target_samples]
        logger.info(f"Generation Complete. Final count: {len(final_synthetic_data)}")

        # Warn (instead of silently truncating) when the dedup loop could not reach the target
        shortfall_warning = None
        if len(final_synthetic_data) < target_samples:
            shortfall_warning = (
                f"Only {len(final_synthetic_data)}/{target_samples} unique synthetic samples "
                f"could be generated after {max_attempts} sampling attempts."
            )
            logger.warning(shortfall_warning)

        # Fill missing original columns
        for col_name in df_real.columns:
            if col_name not in final_synthetic_data.columns:
                final_synthetic_data[col_name] = np.nan

        final_synthetic_data = final_synthetic_data[df_real.columns]

        # --- 8. Save Output ---
        output_filename = f"{job_id}_synthetic_dataset.csv"
        output_path = os.path.join(job_dir, output_filename)

        final_synthetic_data.to_csv(output_path, index=False, na_rep='')
        logger.info(f"SUCCESS: Synthetic data saved to: {output_path}")

        return {
            'job_id': job_id,
            'output_path': output_path,
            'n_rows': len(final_synthetic_data),
            'target_samples': target_samples,
            'shortfall_warning': shortfall_warning,
            'warnings': warnings_list,
            'seed': seed
        }

    except Exception as e:
        logger.error(f"CRITICAL ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        # Return the error instead of sys.exit(): when run inside a worker
        # thread (Flask) a SystemExit would silently kill the thread without
        # marking the job as failed.
        return {
            'job_id': job_id,
            'error': str(e),
            'output_path': None,
            'n_rows': 0,
            'shortfall_warning': None,
            'warnings': warnings_list if 'warnings_list' in locals() else [],
            'seed': seed
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--job_dir', required=True)
    parser.add_argument('--details_file', required=True)
    parser.add_argument('--log_file', required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for reproducibility (default 42).')
    args = parser.parse_args()

    run_synthesis(args.job_id, args.job_dir, args.details_file, args.log_file, seed=args.seed)
