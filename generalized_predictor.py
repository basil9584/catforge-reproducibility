# ./models/prediction/generalized_predictor.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
import warnings
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import re

# Try importing LIME, handle gracefully if missing
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not found. Explanations will be skipped.")

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
import math

warnings.filterwarnings("ignore")

# ==========================================
# 0. PHYSICAL CONSTRAINTS (LOGIT TRANSFORM)
# ==========================================
LOGIT_EPSILON = 1e-4

def logit_transform(y):
    """
    Maps 0-100% range to -inf to +inf using Logit.
    Clips input to (0+eps, 100-eps) first to handle outliers like 311%.
    """
    y = np.array(y)
    # 1. Normalize 0-100 -> 0-1
    y_norm = y / 100.0
    # 2. Clip strictly to avoid log(0) or log(negative)
    y_clipped = np.clip(y_norm, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    # 3. Logit Transform
    return np.log(y_clipped / (1 - y_clipped))

def sigmoid_transform(y_pred):
    """
    Maps -inf to +inf back to 0-100% range.
    """
    y_pred = np.array(y_pred)
    # 1. Sigmoid -> 0-1
    prob = 1 / (1 + np.exp(-y_pred))
    # 2. Scale back to 0-100
    return prob * 100.0

# --- 1. PHYSICS TRANSFORMER ---
class BlindPhysicsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.feature_names_out = None
    def fit(self, X, y=None): return self
    def transform(self, X):
        if hasattr(X, "columns"): X_df = X.copy()
        else: X_df = pd.DataFrame(X)
        X_df.columns = X_df.columns.astype(str)
        X_new = X_df.copy()
        for col in X_df.columns:
            try:
                # Physics shapes
                X_new[f"inv_{col}"] = 1.0 / (X_df[col] + self.epsilon)
                X_new[f"log_{col}"] = np.log(np.abs(X_df[col]) + self.epsilon)
                X_new[f"sqrt_{col}"] = np.sqrt(np.abs(X_df[col]))
                X_new[f"sq_{col}"] = X_df[col] ** 2
            except: continue
        X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.feature_names_out = X_new.columns
        return X_new
    def get_feature_names_out(self, input_features=None): return self.feature_names_out

# --- 2. PLS FEATURE AUGMENTER ---
class PLSFeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pls = None
    def fit(self, X, y=None):
        if y is not None:
            n_samples = X.shape[0]
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            n_comps = max(1, min(self.n_components, n_samples - 1, n_features))
            self.pls = PLSRegression(n_components=n_comps)
            self.pls.fit(X, y)
        return self
    def transform(self, X):
        if self.pls is None: return X
        X_pls = self.pls.transform(X)
        X_pls_df = pd.DataFrame(X_pls, columns=[f"PLS_Comp_{i}" for i in range(X_pls.shape[1])], index=X.index if hasattr(X, "index") else None)
        if hasattr(X, "columns"):
            X_combined = pd.concat([X.reset_index(drop=True), X_pls_df.reset_index(drop=True)], axis=1)
        else:
            X_combined = np.hstack([X, X_pls])
        return X_combined

# --- 3. TUNED XGBOOST WRAPPER ---
class AutoTunedXGB(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.best_estimator_ = None
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'regressor'
        return tags
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Tiny datasets: GridSearchCV folds would be too small to be reliable and
        # cross_val_predict would raise. Fit the base model directly.
        if n_samples < 30:
            self.best_estimator_ = xgb.XGBRegressor(
                objective='reg:squarederror', n_jobs=1, random_state=42,
                n_estimators=100, max_depth=3, learning_rate=0.05
            )
            self.best_estimator_.fit(X, y)
            return self
        cv_folds = max(2, min(3, n_samples))
        if n_samples < 20:
            param_grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
        else:
            param_grid = {'max_depth': [3, 4, 5, 6], 'learning_rate': [0.01, 0.03, 0.05, 0.1], 'n_estimators': [100, 300, 500]}
        xgb_base = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42)
        grid = GridSearchCV(xgb_base, param_grid, cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X, y)
        self.best_estimator_ = grid.best_estimator_
        return self
    def predict(self, X):
        return self.best_estimator_.predict(X)

# ----------------------------------------

def calculate_performance_metrics(model_pipeline, X, y, output_columns, logit_cols=None):
    print("--- Calculating PLS-Boosting Metrics ---")
    metrics = {}
    if logit_cols is None: logit_cols = []
    try:
        n_samples = len(X)
        # Too few samples for any meaningful (even LOO) cross-validation
        if n_samples < 5:
            caveat = (
                f"Too few samples ({n_samples}) for reliable cross-validated metrics; "
                "metric computation was skipped."
            )
            print(f"⚠️ {caveat}")
            metrics['caveat'] = caveat
            metrics['note'] = 'metrics_skipped_due_to_small_sample'
            metrics["overall_model_accuracy_percentage"] = 0.0
            return metrics

        cv_fallback = False
        if n_samples < 10:
            # Leave-One-Out cross-validation for tiny datasets (n_samples folds)
            try:
                y_pred = cross_val_predict(model_pipeline, X, y, cv=n_samples, n_jobs=-1)
            except Exception as e:
                cv_fallback = True
                print(f"Warning: Leave-One-Out CV failed ({e}); using in-sample predictions.")
                y_pred = model_pipeline.predict(X)
        elif n_samples < 30:
            # Small datasets: fixed-seed 5-fold CV (replaces cv=10 GridSearchCV folds)
            try:
                y_pred = cross_val_predict(model_pipeline, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
            except Exception as e:
                cv_fallback = True
                print(f"Warning: 5-fold CV failed ({e}); using in-sample predictions.")
                y_pred = model_pipeline.predict(X)
        else:
            y_pred = cross_val_predict(model_pipeline, X, y, cv=10, n_jobs=-1)

        y_values = y.values
        overall_acc_accum = 0
        valid_targets = 0

        for i, col in enumerate(output_columns):
            y_true_col = y_values[:, i]
            y_pred_col = y_pred[:, i]

            # Inverse Transform for Metrics
            if col in logit_cols:
                y_true_col = sigmoid_transform(y_true_col)
                y_pred_col = sigmoid_transform(y_pred_col)

            r2 = r2_score(y_true_col, y_pred_col)
            rmse = math.sqrt(mean_squared_error(y_true_col, y_pred_col))
            data_range = np.max(y_true_col) - np.min(y_true_col)
            # Heuristic accuracy proxy: maps RMSE relative to the observed target range
            # into a bounded 0-100 percentage. This is NOT a true accuracy metric (which
            # would be meaningless for regression) — it is clamped to [0, 100] so the
            # frontend displays a bounded, interpretable value.
            acc = 100.0 if rmse == 0 else max(0.0, min(100.0, 100.0 * (1 - (rmse / data_range)))) if data_range > 0 else 0.0
            try:
                pearson_r = np.corrcoef(y_true_col, y_pred_col)[0, 1]
                if np.isnan(pearson_r): pearson_r = 0.0
            except: pearson_r = 0.0

            metrics[col] = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "Accuracy_Percentage": round(acc, 2), "Pearson_R": round(pearson_r, 4)}
            overall_acc_accum += acc
            valid_targets += 1

        overall_score = round(overall_acc_accum / valid_targets, 2) if valid_targets > 0 else 0
        metrics["overall_model_accuracy_percentage"] = overall_score
        if cv_fallback:
            metrics['caveat'] = ('Cross-validated predictions were unavailable; metrics were computed on '
                                 'in-sample predictions and may be optimistically biased.')
        print(f"✅ Overall Accuracy: {overall_score}%")
        return metrics
    except Exception as e:
        print(f"Error metrics: {e}")
        return {}

NA_VALUES = ['-', 'NA', 'N/A', 'n/a', 'na', 'NaN', 'nan', '', ' ', 'null', 'NULL', 'None', 'none', '#N/A', '#NA']

def train_and_predict_on_dataframe(training_file_path, prediction_file_path, column_configs, job_dir):
    print("--- Starting PLS-Boosting Training & Prediction ---")
    try:
        # =========================================================
        # 1. LOAD DATA (STRICT SEPARATION OF TRAIN VS PREDICT)
        # =========================================================

        # Load Training Data (Must exist)
        df_train = pd.read_csv(training_file_path, na_values=NA_VALUES)
        print(f"Loaded Training Data: {len(df_train)} rows.")

        # Check Prediction File
        print(f"DEBUG: prediction_file_path input: '{prediction_file_path}'")

        IS_PREDICTION_MODE = False
        if prediction_file_path and isinstance(prediction_file_path, str) and len(prediction_file_path.strip()) > 0:
             # User uploaded a file -> We MUST use this for final output
             print(f"--- 📂 PREDICTION DATASET DETECTED: {os.path.basename(prediction_file_path)} ---")
             df_raw_inference = pd.read_csv(prediction_file_path, na_values=NA_VALUES)
             IS_PREDICTION_MODE = True
             print(f"Loaded Prediction Data: {len(df_raw_inference)} rows.")
        else:
             # No file -> Fallback to using Training Data for inference (Self-Test)
             print("--- ⚠️ NO PREDICTION DATASET: Defaulting to Training Data (Self-Test) ---")
             df_raw_inference = df_train.copy()
             IS_PREDICTION_MODE = False

        # Configs
        input_cols = [c['name'] for c in column_configs if c['role'] == 'Input']
        output_cols = [c['name'] for c in column_configs if c['role'] == 'Output']

        # --- FRONTEND VALIDATION ---
        if not input_cols: raise ValueError("No input columns configured.")
        if not output_cols: raise ValueError("No output columns configured.")

        num_feats = [c['name'] for c in column_configs if c['role'] == 'Input' and c['dataType'] in ['continuous', 'discrete']]
        cat_feats = [c['name'] for c in column_configs if c['role'] == 'Input' and c['dataType'] == 'categorical']

        # Validate columns exist in TRAIN
        available_cols = set(df_train.columns.tolist())
        missing_input = [c for c in input_cols if c not in available_cols]
        missing_output = [c for c in output_cols if c not in available_cols]

        if missing_input:
            raise ValueError(f"Input columns not found in training data: {missing_input}.")
        if missing_output:
            raise ValueError(f"Output columns not found in training data: {missing_output}.")

        # =========================================================
        # 2. PREPARE TRAINING DATA (X_train, y_train)
        # =========================================================
        X_train = df_train[input_cols].copy()
        y_train = df_train[output_cols].copy()

        # Type Conversion
        for c in num_feats: X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
        for c in cat_feats: X_train[c] = X_train[c].astype(str)
        for c in y_train.columns: y_train[c] = pd.to_numeric(y_train[c], errors='coerce')

        # Drop N/A from Training
        combined_train = pd.concat([X_train, y_train], axis=1)
        rows_before = len(combined_train)
        combined_train = combined_train.dropna()
        if len(combined_train) < rows_before:
            print(f"⚠️ Dropped {rows_before - len(combined_train)} rows from TRAINING data due to N/A values.")

        if len(combined_train) == 0:
            raise ValueError("No valid training data remaining after removing rows with N/A values.")

        X_train = combined_train[input_cols].copy()
        y_train = combined_train[output_cols].copy()

        for c in cat_feats:
            X_train[c] = X_train[c].fillna('missing').replace('nan', 'missing')

        # Logit Transform Targets
        # Heuristic: columns whose NAME suggests a bounded 0-100% quantity (e.g. "... (%)",
        # "..._conversion") are mapped through the logit to (-inf, inf) so the regressors
        # never need to predict outside the physical range. Only applied when the observed
        # values actually lie within [0, 100]; a %-like name alone is not enough (the data
        # could be on an entirely different scale).
        logit_cols = []
        for c in output_cols:
            if '%' in c or 'conversion' in c.lower():
                col_vals = pd.to_numeric(y_train[c], errors='coerce').dropna()
                if not col_vals.empty and col_vals.min() >= 0 and col_vals.max() <= 100:
                    logit_cols.append(c)
        if logit_cols:
            print(f"🔒 Applying Logit Transform to: {logit_cols}")
            for c in logit_cols: y_train[c] = logit_transform(y_train[c])
        with open(os.path.join(job_dir, 'target_transform.json'), 'w') as f: json.dump(logit_cols, f)

        # =========================================================
        # 3. TRAIN MODEL (On X_train only)
        # =========================================================
        numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('physics', BlindPhysicsTransformer()), ('scaler', RobustScaler())])
        categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer([('num', numeric_transformer, num_feats), ('cat', categorical_transformer, cat_feats)], remainder='drop')

        pls_comp = max(1, min(3, len(X_train)-1, len(input_cols)))

        # Probe the exact feature count after the preprocessor + PLS augmentation so RFE
        # can never request more features than actually exist.
        try:
            X_prep_probe = preprocessor.fit_transform(X_train)
            pls_probe = PLSFeatureAugmenter(n_components=pls_comp)
            X_aug_probe = pls_probe.fit(X_prep_probe, y_train).transform(X_prep_probe)
            n_features_total = X_aug_probe.shape[1]
        except Exception:
            n_features_total = len(num_feats) * 5 + len(cat_feats) * 3 + pls_comp

        n_features_select = max(1, min(15, max(3, (len(num_feats)*5 + len(cat_feats)*3)//2), n_features_total - 1))
        if n_features_total >= 2:
            selector = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=n_features_select, step=0.1)
        else:
            # Only one feature: RFE cannot select fewer than the total -> pass through
            from sklearn.preprocessing import FunctionTransformer
            selector = FunctionTransformer()

        voting_ensemble = VotingRegressor(estimators=[
            ('xgb', AutoTunedXGB()),
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
            ('pls', PLSRegression(n_components=pls_comp))
        ], weights=[3, 2, 1])

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('pls_augment', PLSFeatureAugmenter(n_components=pls_comp)),
            ('selector', selector),
            ('regressor', MultiOutputRegressor(voting_ensemble))
        ])

        print("Training PLS-Augmented Ensemble...")
        model_pipeline.fit(X_train, y_train)
        joblib.dump(model_pipeline, os.path.join(job_dir, 'dynamic_model_pipeline.pkl'))

        # Calculate Metrics (Validating on Training Data)
        metrics = calculate_performance_metrics(model_pipeline, X_train, y_train, output_cols, logit_cols)
        with open(os.path.join(job_dir, 'model_metrics.json'), 'w') as f: json.dump(metrics, f)

        # Stats
        feature_stats = {}
        for c in num_feats: feature_stats[c] = {'min': float(X_train[c].min()), 'max': float(X_train[c].max()), 'mean': float(X_train[c].mean())}
        for c in cat_feats: feature_stats[c] = {'categories': df_train[c].dropna().unique().tolist()}
        with open(os.path.join(job_dir, 'feature_stats.json'), 'w') as f: json.dump(feature_stats, f)

        # =========================================================
        # 4. PREPARE FINAL INFERENCE DATA (X_final_inference)
        # =========================================================
        print("--- Preparing Final Inference Data ---")
        # We construct X_final_inference strictly from df_raw_inference
        # ensuring it matches the columns expected by the model

        X_final_inference = pd.DataFrame(index=df_raw_inference.index)

        for col in input_cols:
            if col in df_raw_inference.columns:
                X_final_inference[col] = df_raw_inference[col]
            else:
                X_final_inference[col] = np.nan # Fill missing columns with NaN
                print(f"⚠️ Warning: Column '{col}' missing in prediction data. Filled with N/A.")

        # Enforce types
        for c in num_feats:
            X_final_inference[c] = pd.to_numeric(X_final_inference[c], errors='coerce')
        for c in cat_feats:
            X_final_inference[c] = X_final_inference[c].astype(str).replace('nan', 'missing')

        # Clean rows if they have N/A in critical input columns
        # (Optional: we might want to predict anyway, but for safety lets drop rows that are completely broken)
        # Note: We do NOT drop rows here if using Imputers in pipeline, but we should drop if ALL inputs are NaN
        # For now, let's keep all rows to preserve input file length unless strict requirement.

        print(f"Final Inference Set: {len(X_final_inference)} rows.")

        # =========================================================
        # 5. GENERATE ARTIFACTS (Plots & Explanations)
        # =========================================================
        shap_files = []
        lime_files = []
        scatter_files = []

        # A. SHAP (On Training Data Only - for global feature importance)
        try:
            print("Running SHAP (Training Data)...")
            proxy_prep = ColumnTransformer([('num', StandardScaler(), num_feats)], remainder='drop')
            X_proxy = pd.DataFrame(proxy_prep.fit_transform(X_train), columns=num_feats)
            for i, target in enumerate(output_cols):
                pm = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_proxy, y_train.iloc[:, i])
                vals = shap.TreeExplainer(pm).shap_values(X_proxy)
                clean = re.sub('[^0-9a-zA-Z]+', '_', target)
                plt.figure(figsize=(10,6))
                shap.summary_plot(vals, X_proxy, plot_type="bar", show=False)
                plt.title(f"Feature Importance: {target}")
                plt.tight_layout()
                fn = f"shap_{clean}.png"
                plt.savefig(os.path.join(job_dir, fn))
                plt.close()
                shap_files.append(fn)
        except Exception as e: print(f"SHAP skipped: {e}")

        # B. Real vs Predicted (Validation on Training Data)
        print("Generating Real vs Predicted plots (Training Validation)...")
        try:
            n_tr = len(X_train)
            if n_tr < 10:
                # Leave-One-Out CV for tiny datasets (with in-sample fallback)
                try:
                    y_pred_plot = cross_val_predict(model_pipeline, X_train, y_train, cv=n_tr, n_jobs=-1)
                except Exception as e:
                    print(f"Warning: LOO CV for plots failed ({e}); using in-sample predictions.")
                    y_pred_plot = model_pipeline.predict(X_train)
            elif n_tr < 30:
                # Fixed-seed 5-fold CV for small datasets
                y_pred_plot = cross_val_predict(model_pipeline, X_train, y_train, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
            else:
                y_pred_plot = cross_val_predict(model_pipeline, X_train, y_train, cv=5, n_jobs=-1)

            for i, target_col in enumerate(output_cols):
                y_true_vec = y_train[target_col].values
                y_pred_vec = y_pred_plot[:, i]

                if target_col in logit_cols:
                    y_true_vec = sigmoid_transform(y_true_vec)
                    y_pred_vec = sigmoid_transform(y_pred_vec)

                plt.figure(figsize=(8, 8))
                plt.scatter(y_true_vec, y_pred_vec, alpha=0.6, color='#2c3e50', edgecolors='w', s=70)

                min_val = min(y_true_vec.min(), y_pred_vec.min())
                max_val = max(y_true_vec.max(), y_pred_vec.max())
                margin = (max_val - min_val) * 0.05 if (max_val - min_val) > 0 else 0.1
                plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', label='Perfect Prediction')

                plt.xlabel(f'Actual {target_col}')
                plt.ylabel(f'Predicted {target_col}')
                plt.title(f'Actual vs Predicted: {target_col}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                clean_name = re.sub('[^0-9a-zA-Z]+', '_', target_col)
                fn_plot = f"real_vs_pred_{clean_name}.png"
                plt.savefig(os.path.join(job_dir, fn_plot))
                plt.close()
                scatter_files.append(fn_plot)
        except Exception as e:
            print(f"Error creating Real vs Pred plots: {e}")

        # C. LIME (On Inference Data - Explain the actual predictions)
        if LIME_AVAILABLE and len(X_final_inference) > 0:
            print("Running LIME (Inference Data)...")
            try:
                X_train_lime = X_train.copy()
                cat_encoders = {}
                cat_indices = []

                for i, col in enumerate(X_train.columns):
                    if col in cat_feats:
                        le = LabelEncoder()
                        X_train_lime[col] = le.fit_transform(X_train[col].astype(str))
                        cat_encoders[col] = le
                        cat_indices.append(i)

                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train_lime.values,
                    feature_names=X_train.columns.tolist(),
                    categorical_features=cat_indices,
                    class_names=['Value'],
                    mode='regression',
                    verbose=False
                )

                # Explain first row of inference data
                # Fill na just for LIME instance creation
                instance_row = X_final_inference.iloc[0].copy().fillna(0)

                for col, le in cat_encoders.items():
                    try:
                        instance_row[col] = le.transform([str(instance_row[col])])[0]
                    except:
                        instance_row[col] = 0

                instance_np = instance_row.values

                for i, target_col in enumerate(output_cols):

                    def wrapped_predict_fn(z):
                        df_z = pd.DataFrame(z, columns=X_train.columns)
                        for col, le in cat_encoders.items():
                            indices = df_z[col].round().astype(int).clip(0, len(le.classes_)-1)
                            df_z[col] = le.inverse_transform(indices)
                        for col in num_feats: df_z[col] = df_z[col].astype(float)

                        preds_raw = model_pipeline.predict(df_z)
                        target_preds = preds_raw[:, i]

                        if target_col in logit_cols:
                            target_preds = sigmoid_transform(target_preds)
                        return target_preds

                    exp = explainer.explain_instance(instance_np, wrapped_predict_fn, num_features=10)
                    clean_name = re.sub('[^0-9a-zA-Z]+', '_', target_col)

                    exp.save_to_file(os.path.join(job_dir, f'lime_{clean_name}.html'))

                    try:
                        fig = exp.as_pyplot_figure()
                        plt.title(f"LIME Explanation: {target_col}")
                        plt.tight_layout()
                        fn_lime = f"lime_{clean_name}.png"
                        plt.savefig(os.path.join(job_dir, fn_lime))
                        plt.close(fig)
                        lime_files.append(fn_lime)
                    except Exception as plot_err: pass

            except Exception as e:
                print(f"LIME Error: {e}")


        # =========================================================
        # 6. FINAL PREDICTION OUTPUT
        # =========================================================
        print("Generating Final Prediction CSV...")

        preds_logit = model_pipeline.predict(X_final_inference)
        df_preds = pd.DataFrame(preds_logit, columns=output_cols, index=X_final_inference.index)

        # Transform back
        for c in logit_cols: df_preds[c] = sigmoid_transform(df_preds[c])

        # Merge Input Features + Predicted Outputs
        res = pd.concat([X_final_inference.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1)

        return res, feature_stats, shap_files, lime_files, scatter_files, None

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, None, str(e)

def predict_single_dynamic(inputs, job_dir):
    try:
        model = joblib.load(os.path.join(job_dir, 'dynamic_model_pipeline.pkl'))
        with open(os.path.join(job_dir, 'column_config.json'), 'r') as f: conf = json.load(f)
        try:
            with open(os.path.join(job_dir, 'target_transform.json'), 'r') as f: logit_cols = json.load(f)
        except: logit_cols = []

        inp = [c['name'] for c in conf if c['role'] == 'Input']
        df = pd.DataFrame([inputs])
        for c in inp:
            if c not in df.columns: df[c] = np.nan
        df = df[inp]

        pred = model.predict(df)
        out = [c['name'] for c in conf if c['role'] == 'Output']

        res = {}
        for i, c in enumerate(out):
            val = pred[0][i]
            if c in logit_cols: val = sigmoid_transform(val)
            res[c] = float(val)
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None
