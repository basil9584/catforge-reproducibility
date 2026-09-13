# evaluators/main_evaluator.py
# A non-interactive script to perform comprehensive evaluation of synthetic data.
# It generates quality reports, ML efficacy reports, and provides plots.

import pandas as pd
import numpy as np
import os
import argparse
import json
import base64
from io import BytesIO
import sys
import warnings

# Core SDMetrics imports
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import LogisticDetection, MissingValueSimilarity
from sdmetrics.single_column import KSComplement, TVComplement, StatisticSimilarity
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.visualization import get_column_plot

# SDV metadata for plotting and other operations
from sdv.metadata import SingleTableMetadata

# Visualization & ML
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Try importing UMAP; handle graceful failure if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: 'umap-learn' not installed. UMAP plots will be skipped.")

# Suppress warnings for cleaner logs, especially from matplotlib/plotly
warnings.filterwarnings("ignore")
# Set a default plotly template
pio.templates.default = "plotly_white"

class CustomJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy/pandas types, including NaN and Infinity,
    which are not valid in standard JSON.
    """
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            # Use np.isfinite to check for NaN, positive, and negative infinity
            if not np.isfinite(obj):
                return None  # Convert all non-finite float values to null
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj): # Catches pandas NaT and other null types
            return None
        return super().default(obj)

def dataframe_to_json_serializable(df):
    """
    Converts a DataFrame to a JSON serializable format (list of dicts),
    ensuring all NaN/inf values are replaced with None. This provides a robust
    way to clean data before serialization.
    """
    # Create a copy to avoid modifying the original DataFrame in place
    serializable_df = df.copy()

    # Iterate over numeric columns and explicitly replace non-finite values
    for col in serializable_df.select_dtypes(include=[np.number]).columns:
        # Apply a function to replace inf, -inf, and NaN with None
        serializable_df[col] = serializable_df[col].apply(
            lambda x: None if not np.isfinite(x) else x
        )

    # For all columns, replace any remaining pandas/numpy null types with None
    serializable_df = serializable_df.replace({pd.NaT: None, np.nan: None})

    return serializable_df.to_dict(orient='records')

def robust_read_csv(path, **kwargs):
    """
    Attempts to read a CSV with UTF-8, falling back to latin-1 if encoding fails.
    This handles files with special characters (like degree symbols) saved in Excel/Windows.
    CRITICAL FIX: Strips whitespace from column names to handle 'Time ( h)' vs 'Time (h)' mismatches.
    """
    try:
        df = pd.read_csv(path, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        print(f"WARNING: UTF-8 decode failed for {path}. Falling back to 'latin-1' encoding.")
        df = pd.read_csv(path, encoding='latin-1', **kwargs)

    # Clean column names: strip whitespace
    df.columns = df.columns.str.strip()
    return df

def safe_save_plotly_fig(fig, filepath):
    """Safely writes a Plotly figure to disk without crashing if Kaleido is unavailable."""
    try:
        fig.write_image(filepath, format="png", engine="kaleido")
    except Exception as e:
        print(f"WARNING: Kaleido image write failed ({e}). Plot saved as HTML fallback.")
        try:
            html_path = os.path.splitext(filepath)[0] + ".html"
            fig.write_html(html_path)
        except Exception:
            pass

def generate_dimensionality_reduction(df_real, df_synth, plots_dir, max_samples=2000):
    """
    Generates PCA and UMAP plots to visualize global distribution similarity.
    Returns a dictionary of relative paths to the generated images.
    """
    results = {}

    # 1. Preprocessing: Select Numerics
    real_num = df_real.select_dtypes(include=[np.number])
    synth_num = df_synth.select_dtypes(include=[np.number])

    common_cols = [c for c in real_num.columns if c in synth_num.columns]

    valid_cols = []
    for col in common_cols:
        if not real_num[col].isnull().all() and not synth_num[col].isnull().all():
            valid_cols.append(col)
        else:
            print(f"WARNING: Skipping column '{col}' for PCA/UMAP because it is entirely null in one dataset.")

    if len(valid_cols) < 2:
        print("INFO: Skipping Dimension Reduction (PCA/UMAP) - Insufficient valid numeric columns (<2).")
        return {}

    if len(df_real) > max_samples:
        real_sample = df_real[valid_cols].sample(n=max_samples, random_state=42)
    else:
        real_sample = df_real[valid_cols]

    if len(df_synth) > max_samples:
        synth_sample = df_synth[valid_cols].sample(n=max_samples, random_state=42)
    else:
        synth_sample = df_synth[valid_cols]

    real_sample['Data Type'] = 'Real'
    synth_sample['Data Type'] = 'Synthetic'

    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    data_for_fit = combined[valid_cols]
    labels = combined['Data Type']

    try:
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data_for_fit)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
    except Exception as e:
        print(f"ERROR: Preprocessing for DimReduction failed: {e}")
        return {}

    # 4. PCA
    try:
        pca = PCA(n_components=2)
        components_pca = pca.fit_transform(data_scaled)

        fig_pca = px.scatter(
            x=components_pca[:, 0],
            y=components_pca[:, 1],
            color=labels,
            title="PCA: Real vs Synthetic Data Distribution",
            labels={'x': 'PC 1', 'y': 'PC 2'},
            color_discrete_map={'Real': '#636EFA', 'Synthetic': '#EF553B'},
            opacity=0.6,
            width=800,
            height=600
        )

        pca_filename = "pca_distribution.png"
        pca_path = os.path.join(plots_dir, pca_filename)
        safe_save_plotly_fig(fig_pca, pca_path)
        results['pca'] = f"plots/{pca_filename}"
        print("PCA plot generated.")

    except Exception as e:
        print(f"ERROR: PCA generation failed: {e}")

    # 5. UMAP
    if HAS_UMAP:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(data_scaled)

            fig_umap = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color=labels,
                title="UMAP: Real vs Synthetic Data Distribution",
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
                color_discrete_map={'Real': '#636EFA', 'Synthetic': '#EF553B'},
                color_discrete_sequence=['#636EFA', '#EF553B'],
                opacity=0.6,
                width=800,
                height=600
            )

            umap_filename = "umap_distribution.png"
            umap_path = os.path.join(plots_dir, umap_filename)
            safe_save_plotly_fig(fig_umap, umap_path)
            results['umap'] = f"plots/{umap_filename}"
            print("UMAP plot generated.")

        except Exception as e:
            print(f"ERROR: UMAP generation failed: {e}")

    return results

def compute_target_efficacy(real_df, synth_df, target_col):
    """
    Train-on-synthetic / test-on-real efficacy for a single target column.

    Protocol: train a model on SYNTHETIC data, evaluate on REAL data.
    - Continuous targets (numeric, >10 unique values): GradientBoostingRegressor -> R2
    - Categorical/binary targets: LogisticRegression -> accuracy
    Rows with NaN target are dropped; constant targets (<2 unique values) return None.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    try:
        if target_col not in real_df.columns or target_col not in synth_df.columns:
            return None

        real_full = real_df.dropna(subset=[target_col])
        synth_full = synth_df.dropna(subset=[target_col])
        real = real_full[[target_col]]
        synth = synth_full[[target_col]]

        # Guard: need at least 2 rows per frame and a non-constant target
        if len(real) < 2 or len(synth) < 2:
            return None
        if real[target_col].nunique() < 2 or synth[target_col].nunique() < 2:
            return None

        y_train = synth[target_col].astype(float) if pd.api.types.is_numeric_dtype(synth[target_col]) else synth[target_col].astype(str)
        y_test = real[target_col].astype(float) if pd.api.types.is_numeric_dtype(real[target_col]) else real[target_col].astype(str)

        # Use all common non-target columns as numeric features
        feat_cols = [c for c in real_df.columns if c != target_col and c in synth_df.columns]
        if not feat_cols:
            return None

        X_synth = synth_full[feat_cols].apply(pd.to_numeric, errors='coerce')
        X_real = real_full[feat_cols].apply(pd.to_numeric, errors='coerce')

        # Drop columns that are entirely null in either frame, then rows with NaN
        keep_cols = [c for c in feat_cols if not X_synth[c].isnull().all() and not X_real[c].isnull().all()]
        if not keep_cols:
            return None
        X_synth = X_synth[keep_cols]
        X_real = X_real[keep_cols]
        X_synth = X_synth.dropna()
        X_real = X_real.dropna()
        y_train = y_train.loc[X_synth.index]
        y_test = y_test.loc[X_real.index]
        if len(X_synth) < 2 or len(X_real) < 2:
            return None

        # Classification vs regression detection
        is_classification = (not pd.api.types.is_numeric_dtype(real[target_col])) or real[target_col].nunique() <= 10

        if is_classification:
            # Align label sets (train on synthetic labels only)
            model = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                              ('clf', LogisticRegression(max_iter=1000, random_state=42))])
            model.fit(X_synth, y_train)
            pred = model.predict(X_real)
            return {'metric': 'accuracy', 'score': round(float(accuracy_score(y_test, pred)), 4)}
        else:
            model = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                              ('reg', GradientBoostingRegressor(random_state=42))])
            model.fit(X_synth, y_train)
            pred = model.predict(X_real)
            return {'metric': 'r2', 'score': round(float(r2_score(y_test, pred)), 4)}
    except Exception as e:
        print(f"WARNING: Target efficacy for '{target_col}' could not be computed: {e}")
        return None

def run_evaluation(job_id, job_dir, real_data_path, synthetic_data_path, target_column=None):
    """
    Performs comprehensive evaluation of synthetic data against real data.
    Returns the results as a dictionary.
    """
    print(f"--- Starting Evaluation for Job ID: {job_id} ---")

    evaluation_results = {
        'job_id': job_id,
        'quality_score': None,
        'ml_efficacy_score': None,
        'detectability_score': None,
        'target_efficacy': None,
        'sanity_checks': {},
        'plots': {'column_distributions': {}, 'pca': None, 'umap': None},
        'errors': [],
        'available_columns_for_plotting': [],
        'column_plotting_status': {},
        'real_data_sample': {'headers': [], 'data': []},
        'diagnostic_details': [],
        'detailed_quality_scores': {}
    }

    try:
        # FIX: Added `thousands=','` to correctly parse numbers like "1,234".
        # FIX: Added `na_values` to explicitly tell pandas what strings represent missing data.
        na_values = ['-', 'NA', 'N/A', '', ' ']

        # FIX: Use robust reader to handle encoding issues and strip whitespace
        df_real = robust_read_csv(real_data_path, thousands=',', na_values=na_values)
        df_synthetic = robust_read_csv(synthetic_data_path, thousands=',', na_values=na_values)

        print(f"Successfully loaded real data: {real_data_path}. Shape: {df_real.shape}")
        print(f"Successfully loaded synthetic data: {synthetic_data_path}. Shape: {df_synthetic.shape}")

        # --- 1. Sanity Checks & Column Filtering ---
        print("Running Sanity Checks and Filtering Columns...")
        real_cols, synthetic_cols = set(df_real.columns), set(df_synthetic.columns)
        common_columns = list(real_cols.intersection(synthetic_cols))

        columns_to_evaluate = []
        skipped_empty = []

        for col in common_columns:
            # Check for columns that are not entirely null
            if not df_real[col].isnull().all() and not df_synthetic[col].isnull().all():
                columns_to_evaluate.append(col)
            else:
                skipped_empty.append(col)
                print(f"INFO: Skipping column '{col}' because it is empty in the real or synthetic data.")

        if not columns_to_evaluate:
            raise ValueError("No common, non-empty columns found for evaluation.")

        evaluation_results['sanity_checks']['column_presence'] = {
            'status': 'PASSED' if not list(real_cols - synthetic_cols) and not list(synthetic_cols - real_cols) else 'FAILED',
            'missing_in_synthetic': list(real_cols - synthetic_cols),
            'extra_in_synthetic': list(synthetic_cols - real_cols),
            'skipped_empty_columns': skipped_empty
        }

        mismatches = [f"Col '{c}': Real ({df_real[c].dtype}) vs Synth ({df_synthetic[c].dtype})" for c in columns_to_evaluate if str(df_real[c].dtype) != str(df_synthetic[c].dtype)]
        evaluation_results['sanity_checks']['data_type_consistency'] = {
            'status': 'PASSED' if not mismatches else 'FAILED',
            'mismatches': mismatches
        }
        print(f"Columns to be evaluated: {columns_to_evaluate}")
        print("Sanity Checks Completed.")

        evaluation_results['real_data_sample'] = {
            'headers': df_real.columns.tolist(),
            'data': dataframe_to_json_serializable(df_real.head(5))
        }

        # --- 2. Generate Metadata ---
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df_real[columns_to_evaluate])
        print("Metadata generated.")

        # --- 3. Run Quality Report ---
        print("Generating Quality Report...")
        try:
            quality_report = QualityReport()
            # Generate report on copies of data to avoid any modification issues
            quality_report.generate(df_real[columns_to_evaluate].copy(), df_synthetic[columns_to_evaluate].copy(), metadata.to_dict())
            quality_score = quality_report.get_score() * 100
            evaluation_results['quality_score'] = round(quality_score, 2) if np.isfinite(quality_score) else None
            print("Quality Report Generated.")
        except Exception as e:
            print(f"ERROR: Failed to generate Quality Report: {e}")
            evaluation_results['errors'].append(f"Failed to generate Quality Report: {e}")

        # --- 3.5. Run Individual Quality Metrics ---
        print("Generating individual quality metric scores...")
        evaluation_results['detailed_quality_scores'] = {
            'KSComplement': {}, 'TVComplement': {},
            'StatisticSimilarity_mean': {}, 'StatisticSimilarity_std': {},
            'CorrelationSimilarity': None, 'MissingValueSimilarity': None
        }

        for col in columns_to_evaluate:
            try:
                real_col_non_null = df_real[col].dropna()
                synth_col_non_null = df_synthetic[col].dropna()

                if real_col_non_null.empty or synth_col_non_null.empty:
                    continue

                if pd.api.types.is_numeric_dtype(real_col_non_null):
                    ks_score = KSComplement.compute(real_col_non_null, synth_col_non_null)
                    mean_score = StatisticSimilarity.compute(real_col_non_null, synth_col_non_null, statistic='mean')
                    std_score = StatisticSimilarity.compute(real_col_non_null, synth_col_non_null, statistic='std')

                    evaluation_results['detailed_quality_scores']['KSComplement'][col] = round(ks_score, 4) if np.isfinite(ks_score) else None
                    evaluation_results['detailed_quality_scores']['StatisticSimilarity_mean'][col] = round(mean_score, 4) if np.isfinite(mean_score) else None
                    evaluation_results['detailed_quality_scores']['StatisticSimilarity_std'][col] = round(std_score, 4) if np.isfinite(std_score) else None

                elif pd.api.types.is_categorical_dtype(real_col_non_null) or pd.api.types.is_object_dtype(real_col_non_null) or pd.api.types.is_bool_dtype(real_col_non_null):
                    tv_score = TVComplement.compute(real_col_non_null.astype(str), synth_col_non_null.astype(str))
                    evaluation_results['detailed_quality_scores']['TVComplement'][col] = round(tv_score, 4) if np.isfinite(tv_score) else None
            except Exception as e:
                print(f"ERROR: Could not compute single-column metric for column '{col}': {e}")

        try:
            corr_sim_score = CorrelationSimilarity.compute(df_real[columns_to_evaluate], df_synthetic[columns_to_evaluate])
            evaluation_results['detailed_quality_scores']['CorrelationSimilarity'] = round(corr_sim_score, 4) if np.isfinite(corr_sim_score) else None
        except Exception as e:
            print(f"ERROR: Could not compute CorrelationSimilarity: {e}")

        try:
            mvs_score = MissingValueSimilarity.compute(df_real[columns_to_evaluate], df_synthetic[columns_to_evaluate], metadata=metadata.to_dict())
            evaluation_results['detailed_quality_scores']['MissingValueSimilarity'] = round(mvs_score, 4) if np.isfinite(mvs_score) else None
        except Exception as e:
            print(f"ERROR: Could not compute MissingValueSimilarity: {e}")

        print("Individual quality metric scores generated.")

        # --- 4. Run Diagnostic Report ---
        print("Generating Diagnostic Report...")
        try:
            diagnostic_report = DiagnosticReport()
            diagnostic_report.generate(df_real[columns_to_evaluate], df_synthetic[columns_to_evaluate], metadata.to_dict())
            dv_details = diagnostic_report.get_details('Data Validity')
            ds_details = diagnostic_report.get_details('Data Structure')
            details_df = pd.concat([dv_details, ds_details])
            if not details_df.empty:
                evaluation_results['diagnostic_details'] = dataframe_to_json_serializable(details_df)
            print("Diagnostic Report Generated.")
        except Exception as e:
            print(f"ERROR: Failed to generate Diagnostic Report: {e}")
            evaluation_results['errors'].append(f"Failed to generate Diagnostic Report: {e}")

        # --- 5. ML Efficacy (Logistic Detection) ---
        # NOTE: sdmetrics LogisticDetection.compute returns 1 - mean(per-fold power),
        # i.e. HIGH = detector cannot distinguish real from synthetic = GOOD.
        print("Running ML Efficacy (Logistic Detection)...")
        try:
            # Drop rows with NaN in any of the evaluation columns before ML efficacy
            real_no_na = df_real[columns_to_evaluate].dropna()
            synth_no_na = df_synthetic[columns_to_evaluate].dropna()

            if not real_no_na.empty and not synth_no_na.empty:
                ld = LogisticDetection()
                detection_score = ld.compute(real_no_na, synth_no_na)
                # detection_score is already "1 - power": high = indistinguishable = good
                ml_score = detection_score * 100
                evaluation_results['ml_efficacy_score'] = round(ml_score, 2) if np.isfinite(ml_score) else None
                # Diagnostic counterpart: lower = harder to detect = better
                evaluation_results['detectability_score'] = round((1 - detection_score) * 100, 2)
                print(f"ML Efficacy (Logistic Detection) Score: {evaluation_results['ml_efficacy_score']}% (higher = more indistinguishable = better)")
                print(f"Detectability Score: {evaluation_results['detectability_score']}% (lower = harder to detect = better)")
            else:
                print("INFO: Skipping ML Efficacy due to insufficient data after dropping NaNs.")
        except Exception as e:
            print(f"ERROR: Failed to run ML Efficacy: {e}")

        # --- 5.5. Target-Conditional Efficacy (train-on-synthetic / test-on-real) ---
        evaluation_results['target_efficacy'] = None
        if target_column:
            print(f"Running Target-Conditional Efficacy for target: {target_column}...")
            target_efficacy = compute_target_efficacy(df_real, df_synthetic, target_column)
            if target_efficacy is None:
                print(f"WARNING: Target efficacy for '{target_column}' could not be computed (target missing, constant, or data too small).")
            else:
                evaluation_results['target_efficacy'] = target_efficacy
                print(f"Target Efficacy ({target_efficacy['metric']}): {target_efficacy['score']}")

        # --- 6. Generate Column Distribution Plots ---
        print("Generating Column Distribution Plots...")
        evaluation_results['available_columns_for_plotting'] = columns_to_evaluate

        # Create a 'plots' subdirectory to keep things clean
        plots_dir = os.path.join(job_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # A. Dimensionality Reduction (PCA/UMAP)
        print("Generating PCA/UMAP Plots...")
        dim_plots = generate_dimensionality_reduction(df_real, df_synthetic, plots_dir)
        evaluation_results['plots'].update(dim_plots)

        # B. Generate Column Distribution Plots
        for col in columns_to_evaluate:
            try:
                if df_real[col].dropna().empty or df_synthetic[col].dropna().empty:
                    print(f"WARNING: Skipping plot for '{col}' due to insufficient data.")
                    evaluation_results['column_plotting_status'][col] = "insufficient_data"
                    continue

                fig = get_column_plot(real_data=df_real, synthetic_data=df_synthetic, column_name=col)
                if fig:
                    # Sanitize column name for filename
                    safe_col_name = "".join([c if c.isalnum() else "_" for c in col])
                    filename = f"dist_plot_{safe_col_name}.png"
                    file_path = os.path.join(plots_dir, filename)

                    # Save directly to disk
                    fig.write_image(file_path, format="png", engine="kaleido")

                    # Store RELATIVE path or FILENAME for the frontend
                    evaluation_results['plots']['column_distributions'][col] = f"plots/{filename}"
                    evaluation_results['column_plotting_status'][col] = "plottable"
                else:
                    evaluation_results['column_plotting_status'][col] = "plot_generation_error"
            except Exception as e:
                print(f"ERROR: Failed to generate plot for column '{col}': {e}")
                # Log the specific error so you can see if it's still Kaleido or something else
                evaluation_results['errors'].append(f"Plot error ({col}): {str(e)}")
                evaluation_results['column_plotting_status'][col] = "plot_generation_error"
        print("Column Distribution Plots Generated.")

    except Exception as e:
        print(f"CRITICAL ERROR during evaluation: {e}")
        evaluation_results['errors'].append(f"Critical evaluation error: {e}")
        import traceback
        traceback.print_exc()

    print(f"--- Evaluation Job {job_id} Finished ---")
    return evaluation_results

# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a non-interactive synthetic data evaluation job.")
    parser.add_argument('--job_id', required=True, help='Unique ID for the job.')
    parser.add_argument('--job_dir', required=True, help='Directory to store job artifacts.')
    parser.add_argument('--real_data_path', required=True, help='Path to the real dataset CSV.')
    parser.add_argument('--synthetic_data_path', required=True, help='Path to the synthetic dataset CSV.')
    parser.add_argument('--target_column', default=None, help='Optional: Target column for train-on-synthetic/test-on-real efficacy evaluation.')
    parser.add_argument('--output_file', required=True, help='Path to a file to save the JSON output.')

    args = parser.parse_args()

    os.makedirs(args.job_dir, exist_ok=True)

    evaluation_output = run_evaluation(
        job_id=args.job_id,
        job_dir=args.job_dir,
        real_data_path=args.real_data_path,
        synthetic_data_path=args.synthetic_data_path,
        target_column=args.target_column
    )

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                # Use the custom encoder as a reliable way to create valid JSON
                json.dump(evaluation_output, f, indent=4, cls=CustomJsonEncoder)
            print(f"INFO: Evaluation results saved to {args.output_file}")
        except Exception as e:
            print(f"ERROR: Failed to save evaluation results to {args.output_file}. Details: {e}")
            # As a fallback, try to write with errors escaped
            try:
                with open(args.output_file + '.err', 'w') as f_err:
                    f_err.write(str(evaluation_output))
            except:
                pass
            sys.exit(1)
