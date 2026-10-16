"""
COVID-19 ML Pipeline with 30x Bootstrapping (Chopped Dataset Version)
=====================================================================
Reproducing and extending the methodology of the CS229 COVID-19 paper:
  - 3 chopped datasets: CLIN_chopped.csv, LAB_chopped.csv, CLINLAB_chopped.csv
    (with positive prevalence matched to the paper: ICU ~5.78%, Vent ~3.29%, Death ~1.96%)
  - 3 binary targets:   is_icu, was_ventilated, last.status
  - 6 feature methods:  Raw, SMOTEENN, PCA, LASSO, GUS, FPR
  - 5 models:           LogisticRegression, LightGBM, MLP, SVM, RandomForest
  - 5-fold Stratified CV for hyperparameter tuning (GridSearchCV with AUROC)
  - 30x paired bootstrapping on the test set for robustness testing, 95% CIs,
    and paired statistical significance comparison (matching Fig. 3 of paper).

Outputs:
  1. results_all_bootstrapped.csv — test metrics (AUROC, ACC, AUPRC) and 30x bootstrap
                                    statistics (mean, std, 95% CI) for every combination.
  2. best_models_bootstrapped.csv — optimal model per (Dataset × Target).
  3. bootstrap_scores_30x.csv     — individual scores for all 30 bootstrap iterations
                                    per combination (ready for box plots & paired t-tests).

Usage:
  # Normal execution (30x bootstrapping)
  python covid_ml_pipeline_bootstrapped.py

  # Custom arguments (e.g. in Colab)
  python covid_ml_pipeline_bootstrapped.py --n-bootstrap 30 --cv-folds 5
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    GenericUnivariateSelect,
    f_classif,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")  # Headless mode for server / Colab compatibility
import matplotlib.pyplot as plt

import lightgbm as lgb
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

# ────────────────────────── Default Configuration ──────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.30        # 70 / 30 train-test split as in paper
CV_FOLDS = 5
SCORING = "roc_auc"
N_BOOTSTRAPS = 30       # 30x bootstrapping as in paper (Fig. 3)

DATASET_PATHS = {
    "CLIN":    "CLIN_chopped.csv",
    "LAB":     "LAB_chopped.csv",
    "CLINLAB": "CLINLAB_chopped.csv",
}

TARGET_COLUMNS = {
    "is_icu":          {"pos_label": True,        "col": "is_icu"},
    "was_ventilated":  {"pos_label": "Yes",       "col": "was_ventilated"},
    "last.status":     {"pos_label": "deceased",  "col": "last.status"},
}

OUTPUT_FILE    = "results_all_bootstrapped.csv"
BEST_FILE      = "best_models_bootstrapped.csv"
BOOTSTRAP_FILE = "bootstrap_scores_30x.csv"


# ────────────────────────── Data Loading & Preprocessing ──────────

def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load CSV, encode categoricals, impute missing values, return (df, feature_cols)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file '{path}' not found. Please ensure chopped CSV files exist.")

    df = pd.read_csv(path)

    target_cols = ["is_icu", "was_ventilated", "last.status"]
    feature_cols = [c for c in df.columns if c not in target_cols]

    # Encode target labels to binary 0/1
    df["is_icu"] = df["is_icu"].astype(int)
    df["was_ventilated"] = (df["was_ventilated"] == "Yes").astype(int)
    df["last.status"]    = (df["last.status"] == "deceased").astype(int)

    # Identify non-numeric columns
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # Encode categoricals with LabelEncoder
    for c in cat_cols:
        df[c] = df[c].astype(object)
        df.loc[df[c].isna(), c] = np.nan
        le = LabelEncoder()
        mask = df[c].notna()
        if mask.any():
            df.loc[mask, c] = le.fit_transform(df.loc[mask, c].astype(str))
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df, feature_cols


def get_Xy(df: pd.DataFrame, feature_cols: List[str], target_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, y arrays for a given target."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_name].values.astype(int)
    return X, y


# ────────────────────────── Feature Engineering Methods ───────────

def fe_raw(X_train, y_train, X_test):
    """Raw scaled features."""
    sc = StandardScaler()
    return sc.fit_transform(X_train), y_train, sc.transform(X_test)


def fe_smoteenn(X_train, y_train, X_test):
    """SMOTEENN resampling + scaling."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    try:
        smote_enn = SMOTEENN(random_state=RANDOM_STATE)
        X_tr, y_tr = smote_enn.fit_resample(X_tr, y_train)
    except ValueError:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = smote.fit_resample(X_tr, y_train)
    return X_tr, y_tr, X_te


def fe_pca(X_train, y_train, X_test):
    """PCA (retain 95% variance) + scaling."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    return X_tr, y_train, X_te


def fe_lasso(X_train, y_train, X_test):
    """SMOTEENN + LASSO L1 feature selection (as in the paper)."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    try:
        smote_enn = SMOTEENN(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote_enn.fit_resample(X_tr, y_train)
    except ValueError:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_train)

    lasso = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                               max_iter=5000, random_state=RANDOM_STATE)
    lasso.fit(X_tr_res, y_tr_res)
    selector = SelectFromModel(lasso, prefit=True)
    X_tr_sel = selector.transform(X_tr_res)
    X_te_sel = selector.transform(X_te)
    if X_tr_sel.shape[1] == 0:
        return X_tr_res, y_tr_res, X_te
    return X_tr_sel, y_tr_res, X_te_sel


def fe_gus(X_train, y_train, X_test):
    """Generic Univariate Selection (k-best with f_classif)."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    n_features = max(1, min(X_tr.shape[1], int(0.5 * X_tr.shape[1])))
    gus = GenericUnivariateSelect(score_func=f_classif, mode="k_best", param=n_features)
    X_tr = gus.fit_transform(X_tr, y_train)
    X_te = gus.transform(X_te)
    return X_tr, y_train, X_te


def fe_fpr(X_train, y_train, X_test):
    """False Positive Rate feature selection (alpha=0.05)."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    fpr_sel = GenericUnivariateSelect(score_func=f_classif, mode="fpr", param=0.05)
    X_tr = fpr_sel.fit_transform(X_tr, y_train)
    X_te = fpr_sel.transform(X_te)
    if X_tr.shape[1] == 0:
        sc2 = StandardScaler()
        return sc2.fit_transform(X_train), y_train, sc2.transform(X_test)
    return X_tr, y_train, X_te


FEATURE_METHODS = {
    "Raw":      fe_raw,
    "SMOTEENN": fe_smoteenn,
    "PCA":      fe_pca,
    "LASSO":    fe_lasso,
    "GUS":      fe_gus,
    "FPR":      fe_fpr,
}


# ────────────────────────── Prediction Models & Grids ─────────────

def get_models_and_grids() -> Dict[str, Tuple[Any, Dict]]:
    """Return dict of model_name → (estimator, param_grid)."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver="saga"),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
            },
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31],
            },
        ),
        "MLP": (
            MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, early_stopping=True),
            {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 1e-2],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        ),
    }


# ────────────────────────── Evaluation & 30x Bootstrapping ────────

def generate_paired_bootstrap_indices(y_test: np.ndarray, n_bootstraps: int = N_BOOTSTRAPS,
                                      random_state: int = RANDOM_STATE) -> List[np.ndarray]:
    """
    Generate paired bootstrap index sets across test patients once per target.
    This guarantees every model/feature combo is evaluated on identical resamples,
    enabling paired one-sided t-tests as performed in the reference paper.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    bootstrap_indices = []

    for _ in range(n_bootstraps):
        while True:
            idx = rng.choice(n, size=n, replace=True)
            # Ensure both positive and negative classes are present in the bootstrap draw
            if len(np.unique(y_test[idx])) > 1:
                bootstrap_indices.append(idx)
                break

    return bootstrap_indices


def evaluate_with_bootstrapping(y_test: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                                bootstrap_indices: List[np.ndarray]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Compute standard test metrics plus 30x bootstrap metrics:
      - Point estimates on the full test set: AUROC, ACC, AUPRC
      - Bootstrap statistics: Mean, Std, 95% Confidence Interval (2.5% to 97.5%)
      - Per-iteration replication records for box plots and paired t-tests
    """
    main_auroc = roc_auc_score(y_test, y_prob)
    main_acc   = accuracy_score(y_test, y_pred)
    main_auprc = average_precision_score(y_test, y_prob)

    boot_aurocs = []
    boot_accs   = []
    boot_auprcs = []
    replications = []

    for b_idx, idx in enumerate(bootstrap_indices, start=1):
        y_b_true = y_test[idx]
        y_b_prob = y_prob[idx]
        y_b_pred = y_pred[idx]

        b_auroc = roc_auc_score(y_b_true, y_b_prob)
        b_acc   = accuracy_score(y_b_true, y_b_pred)
        b_auprc = average_precision_score(y_b_true, y_b_prob)

        boot_aurocs.append(b_auroc)
        boot_accs.append(b_acc)
        boot_auprcs.append(b_auprc)

        replications.append({
            "Bootstrap_Iter": b_idx,
            "AUROC": round(b_auroc, 5),
            "ACC":   round(b_acc, 5),
            "AUPRC": round(b_auprc, 5),
        })

    summary_metrics = {
        "AUROC":            main_auroc,
        "ACC":              main_acc,
        "AUPRC":            main_auprc,
        "AUROC_boot_mean":  float(np.mean(boot_aurocs)),
        "AUROC_boot_std":   float(np.std(boot_aurocs)),
        "AUROC_ci_lower":   float(np.percentile(boot_aurocs, 2.5)),
        "AUROC_ci_upper":   float(np.percentile(boot_aurocs, 97.5)),
        "ACC_boot_mean":    float(np.mean(boot_accs)),
        "ACC_boot_std":     float(np.std(boot_accs)),
        "ACC_ci_lower":     float(np.percentile(boot_accs, 2.5)),
        "ACC_ci_upper":     float(np.percentile(boot_accs, 97.5)),
        "AUPRC_boot_mean":  float(np.mean(boot_auprcs)),
        "AUPRC_boot_std":   float(np.std(boot_auprcs)),
        "AUPRC_ci_lower":   float(np.percentile(boot_auprcs, 2.5)),
        "AUPRC_ci_upper":   float(np.percentile(boot_auprcs, 97.5)),
    }

    return summary_metrics, replications


# ────────────────────────── Performance Curves Plotting ───────────

def plot_performance_curves(test_predictions_store: Dict[Tuple[str, str, str, str], Dict[str, Any]],
                            results_df: pd.DataFrame, output_dir: str = "."):
    """
    Generate AUROC (ROC), AUPRC (Precision-Recall), and ACC (Accuracy vs. Threshold) curves
    for the best-performing pipeline of each dataset and target, matching Fig. 2 of the paper.
    
    Produces:
      1. performance_curves_grid.png        — 3x3 publication grid (Targets × [ROC, PR, ACC])
      2. roc_curves_best_models.png         — 1x3 ROC comparison across the 3 targets
      3. pr_curves_best_models.png          — 1x3 PR comparison across the 3 targets
      4. accuracy_curves_best_models.png    — 1x3 Accuracy vs Threshold comparison
      5. overall_best_pipelines_summary.png — The single overall best pipeline for each target
    """
    target_names = ["is_icu", "was_ventilated", "last.status"]
    target_labels_map = {
        "is_icu": "ICU Admission",
        "was_ventilated": "Mechanical Ventilation",
        "last.status": "Mortality (Death)"
    }
    dataset_colors = {
        "CLIN": "#1f77b4",     # Steel Blue
        "LAB": "#ff7f0e",      # Amber Orange
        "CLINLAB": "#2ca02c"   # Forest Green
    }
    dataset_markers = {
        "CLIN": "o",
        "LAB": "s",
        "CLINLAB": "^"
    }

    valid_results = results_df.dropna(subset=["AUROC"])
    if valid_results.empty or not test_predictions_store:
        print("[WARN] No valid predictions available to plot curves.")
        return

    # Find best model per (Dataset, Target) by test AUROC
    best_idx = valid_results.groupby(["Dataset", "Target"])["AUROC"].idxmax()
    best_per_ds_tgt = valid_results.loc[best_idx]

    # Find overall best model per Target
    overall_best_idx = valid_results.groupby("Target")["AUROC"].idxmax()
    overall_best_per_tgt = valid_results.loc[overall_best_idx].set_index("Target")

    print("\n" + "=" * 85)
    print("  GENERATING PERFORMANCE CURVES (AUROC, AUPRC, ACCURACY)")
    print("=" * 85)

    # ──────────────── 1. Comprehensive 3x3 Grid ────────────────
    fig, axes = plt.subplots(3, 3, figsize=(19, 15), dpi=300)

    for row_idx, tgt in enumerate(target_names):
        tgt_display = target_labels_map.get(tgt, tgt)
        ax_roc = axes[row_idx, 0]
        ax_pr  = axes[row_idx, 1]
        ax_acc = axes[row_idx, 2]

        # ROC Subplot setup
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.3, label="Chance (AUC = 0.50)")
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5)
        ax_roc.set_title(f"{tgt_display}: ROC Curves (AUROC)", fontsize=11.5, fontweight="bold")
        ax_roc.grid(True, linestyle=":", alpha=0.5)

        # PR Subplot setup
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.05])
        ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10.5)
        ax_pr.set_ylabel("Precision (PPV)", fontsize=10.5)
        ax_pr.set_title(f"{tgt_display}: Precision-Recall Curves (AUPRC)", fontsize=11.5, fontweight="bold")
        ax_pr.grid(True, linestyle=":", alpha=0.5)

        # Accuracy Subplot setup
        ax_acc.set_xlim([-0.02, 1.02])
        ax_acc.set_ylim([0.70, 1.02])
        ax_acc.set_xlabel("Decision Threshold", fontsize=10.5)
        ax_acc.set_ylabel("Accuracy", fontsize=10.5)
        ax_acc.set_title(f"{tgt_display}: Accuracy vs. Threshold", fontsize=11.5, fontweight="bold")
        ax_acc.axvline(0.5, color="gray", linestyle="--", lw=1.2, label="Default Threshold (0.50)")
        ax_acc.grid(True, linestyle=":", alpha=0.5)

        prevalence = None

        # Plot each dataset's optimal pipeline
        for ds in ["CLIN", "LAB", "CLINLAB"]:
            match = best_per_ds_tgt[(best_per_ds_tgt["Dataset"] == ds) & (best_per_ds_tgt["Target"] == tgt)]
            if match.empty:
                continue
            row = match.iloc[0]
            fe = row["Feature_Eng"]
            model = row["Model"]
            key = (ds, tgt, fe, model)

            if key not in test_predictions_store:
                continue

            data = test_predictions_store[key]
            y_true = data["y_true"]
            y_prob = data["y_prob"]
            auroc_val = data["AUROC"]
            auprc_val = data["AUPRC"]

            if prevalence is None:
                prevalence = float(np.mean(y_true))

            color = dataset_colors.get(ds, "#333333")
            is_overall_best = (tgt in overall_best_per_tgt.index and
                               row["Dataset"] == overall_best_per_tgt.loc[tgt, "Dataset"] and
                               row["Model"] == overall_best_per_tgt.loc[tgt, "Model"] and
                               row["Feature_Eng"] == overall_best_per_tgt.loc[tgt, "Feature_Eng"])

            lw = 2.6 if is_overall_best else 1.8
            label_tag = " ★ TOP" if is_overall_best else ""

            # 1. ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=color, lw=lw,
                        label=f"{ds} ({model}/{fe}): AUC={auroc_val:.3f}{label_tag}")

            # 2. PR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax_pr.plot(recall, precision, color=color, lw=lw,
                       label=f"{ds} ({model}/{fe}): AUPRC={auprc_val:.3f}{label_tag}")

            # 3. Accuracy vs Threshold
            thresholds = np.linspace(0.02, 0.98, 97)
            accs = np.array([accuracy_score(y_true, (y_prob >= t).astype(int)) for t in thresholds])
            peak_idx = int(np.argmax(accs))
            peak_acc = float(accs[peak_idx])
            peak_t = float(thresholds[peak_idx])
            acc_at_50 = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

            ax_acc.plot(thresholds, accs, color=color, lw=lw,
                        label=f"{ds} ({model}): Peak={peak_acc:.3f} (@t={peak_t:.2f}){label_tag}")
            ax_acc.scatter([peak_t], [peak_acc], color=color, marker=dataset_markers.get(ds, "o"),
                           s=55, zorder=5)

        # Baseline prevalence for PR
        if prevalence is not None:
            ax_pr.axhline(prevalence, color="gray", linestyle="--", lw=1.2,
                          label=f"Prevalence ({prevalence:.1%})")

        ax_roc.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
        ax_pr.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
        ax_acc.legend(loc="lower center", fontsize=8.2, framealpha=0.92)

    plt.tight_layout()
    grid_out = os.path.join(output_dir, "performance_curves_grid.png")
    plt.savefig(grid_out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved 3x3 composite curves grid: '{grid_out}'")

    # ──────────────── 2. Standalone 1x3 ROC Curves ────────────────
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    for col_idx, tgt in enumerate(target_names):
        tgt_display = target_labels_map.get(tgt, tgt)
        ax = axes_roc[col_idx]
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.3, label="Chance (AUC = 0.50)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"{tgt_display}: AUROC Curves", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        for ds in ["CLIN", "LAB", "CLINLAB"]:
            match = best_per_ds_tgt[(best_per_ds_tgt["Dataset"] == ds) & (best_per_ds_tgt["Target"] == tgt)]
            if match.empty:
                continue
            row = match.iloc[0]
            key = (ds, tgt, row["Feature_Eng"], row["Model"])
            if key not in test_predictions_store:
                continue
            d = test_predictions_store[key]
            fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
            color = dataset_colors.get(ds, "#333333")
            is_best = (tgt in overall_best_per_tgt.index and
                       row["Dataset"] == overall_best_per_tgt.loc[tgt, "Dataset"] and
                       row["Model"] == overall_best_per_tgt.loc[tgt, "Model"])
            lw = 2.6 if is_best else 1.8
            tag = " ★ TOP" if is_best else ""
            ax.plot(fpr, tpr, color=color, lw=lw,
                    label=f"{ds} ({row['Model']}/{row['Feature_Eng']}): AUC={d['AUROC']:.3f}{tag}")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    roc_out = os.path.join(output_dir, "roc_curves_best_models.png")
    plt.savefig(roc_out, bbox_inches="tight")
    plt.close(fig_roc)
    print(f"  ✓ Saved standalone AUROC curves:    '{roc_out}'")

    # ──────────────── 3. Standalone 1x3 PR Curves ─────────────────
    fig_pr, axes_pr = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    for col_idx, tgt in enumerate(target_names):
        tgt_display = target_labels_map.get(tgt, tgt)
        ax = axes_pr[col_idx]
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{tgt_display}: AUPRC Curves", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        prevalence = None
        for ds in ["CLIN", "LAB", "CLINLAB"]:
            match = best_per_ds_tgt[(best_per_ds_tgt["Dataset"] == ds) & (best_per_ds_tgt["Target"] == tgt)]
            if match.empty:
                continue
            row = match.iloc[0]
            key = (ds, tgt, row["Feature_Eng"], row["Model"])
            if key not in test_predictions_store:
                continue
            d = test_predictions_store[key]
            if prevalence is None:
                prevalence = float(np.mean(d["y_true"]))
            precision, recall, _ = precision_recall_curve(d["y_true"], d["y_prob"])
            color = dataset_colors.get(ds, "#333333")
            is_best = (tgt in overall_best_per_tgt.index and
                       row["Dataset"] == overall_best_per_tgt.loc[tgt, "Dataset"] and
                       row["Model"] == overall_best_per_tgt.loc[tgt, "Model"])
            lw = 2.6 if is_best else 1.8
            tag = " ★ TOP" if is_best else ""
            ax.plot(recall, precision, color=color, lw=lw,
                    label=f"{ds} ({row['Model']}/{row['Feature_Eng']}): AUPRC={d['AUPRC']:.3f}{tag}")

        if prevalence is not None:
            ax.axhline(prevalence, color="gray", linestyle="--", lw=1.2,
                       label=f"Prevalence ({prevalence:.1%})")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    pr_out = os.path.join(output_dir, "pr_curves_best_models.png")
    plt.savefig(pr_out, bbox_inches="tight")
    plt.close(fig_pr)
    print(f"  ✓ Saved standalone AUPRC curves:    '{pr_out}'")

    # ─────────────── 4. Standalone 1x3 Accuracy Curves ───────────
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    for col_idx, tgt in enumerate(target_names):
        tgt_display = target_labels_map.get(tgt, tgt)
        ax = axes_acc[col_idx]
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([0.70, 1.02])
        ax.set_xlabel("Classification Threshold", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{tgt_display}: Accuracy vs. Threshold", fontsize=12, fontweight="bold")
        ax.axvline(0.5, color="gray", linestyle="--", lw=1.2, label="Default Threshold (0.50)")
        ax.grid(True, linestyle=":", alpha=0.5)

        for ds in ["CLIN", "LAB", "CLINLAB"]:
            match = best_per_ds_tgt[(best_per_ds_tgt["Dataset"] == ds) & (best_per_ds_tgt["Target"] == tgt)]
            if match.empty:
                continue
            row = match.iloc[0]
            key = (ds, tgt, row["Feature_Eng"], row["Model"])
            if key not in test_predictions_store:
                continue
            d = test_predictions_store[key]
            y_true, y_prob = d["y_true"], d["y_prob"]
            thresholds = np.linspace(0.02, 0.98, 97)
            accs = np.array([accuracy_score(y_true, (y_prob >= t).astype(int)) for t in thresholds])
            peak_idx = int(np.argmax(accs))
            peak_acc = float(accs[peak_idx])
            peak_t = float(thresholds[peak_idx])

            color = dataset_colors.get(ds, "#333333")
            is_best = (tgt in overall_best_per_tgt.index and
                       row["Dataset"] == overall_best_per_tgt.loc[tgt, "Dataset"] and
                       row["Model"] == overall_best_per_tgt.loc[tgt, "Model"])
            lw = 2.6 if is_best else 1.8
            tag = " ★ TOP" if is_best else ""
            ax.plot(thresholds, accs, color=color, lw=lw,
                    label=f"{ds} ({row['Model']}): Peak={peak_acc:.3f} (@t={peak_t:.2f}){tag}")
            ax.scatter([peak_t], [peak_acc], color=color, marker=dataset_markers.get(ds, "o"), s=55, zorder=5)

        ax.legend(loc="lower center", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    acc_out = os.path.join(output_dir, "accuracy_curves_best_models.png")
    plt.savefig(acc_out, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  ✓ Saved standalone Accuracy curves: '{acc_out}'")

    # ────────── 5. Overall Best Pipeline Summary Panel (1x3) ─────
    fig_best, axes_best = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    for col_idx, tgt in enumerate(target_names):
        tgt_display = target_labels_map.get(tgt, tgt)
        ax = axes_best[col_idx]
        if tgt not in overall_best_per_tgt.index:
            continue
        best_row = overall_best_per_tgt.loc[tgt]
        ds = best_row["Dataset"]
        fe = best_row["Feature_Eng"]
        model = best_row["Model"]
        key = (ds, tgt, fe, model)

        if key not in test_predictions_store:
            continue

        d = test_predictions_store[key]
        y_true, y_prob = d["y_true"], d["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        thresholds = np.linspace(0.02, 0.98, 97)
        accs = np.array([accuracy_score(y_true, (y_prob >= t).astype(int)) for t in thresholds])

        ax.plot(fpr, tpr, color="#1f77b4", lw=2.4, label=f"ROC (AUC = {d['AUROC']:.3f})")
        ax.plot(recall, precision, color="#ff7f0e", lw=2.4, label=f"PR (AUPRC = {d['AUPRC']:.3f})")
        ax.plot(thresholds, accs, color="#2ca02c", lw=2.4, label=f"Accuracy (Max = {np.max(accs):.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.0, alpha=0.7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("Rate / Threshold / Recall", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title(f"{tgt_display}\nBest: {ds} | {model} | {fe}", fontsize=11.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    summary_out = os.path.join(output_dir, "overall_best_pipelines_summary.png")
    plt.savefig(summary_out, bbox_inches="tight")
    plt.close(fig_best)
    print(f"  ✓ Saved overall best pipeline panel:'{summary_out}'")
    print("=" * 85)


# ────────────────────────── Main Execution ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run COVID-19 ML pipeline with 30x bootstrapping.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAPS,
                        help="Number of bootstrap replications (default: 30)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Test set ratio (default: 0.30)")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS,
                        help="CV folds for hyperparameter tuning (default: 5)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    n_boot = args.n_bootstrap
    test_ratio = args.test_size
    cv_k = args.cv_folds
    seed = args.seed

    print("=" * 85)
    print("  COVID-19 ML PIPELINE (CHOPPED DATASETS + 30x BOOTSTRAPPING)")
    print("=" * 85)
    print(f"  Datasets:         {list(DATASET_PATHS.keys())}")
    print(f"  Target Labels:    {list(TARGET_COLUMNS.keys())}")
    print(f"  Feature Methods:  {list(FEATURE_METHODS.keys())}")
    print(f"  Models:           {list(get_models_and_grids().keys())}")
    print(f"  Bootstrapping:    {n_boot}x paired draws per target")
    print(f"  Train/Test split: {int((1-test_ratio)*100)}% train / {int(test_ratio*100)}% test")
    print(f"  CV Folds:         {cv_k}-fold Stratified CV (scoring={SCORING})")
    print("=" * 85)

    all_results = []
    all_bootstrap_records = []
    test_predictions_store: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    total_combos = (len(DATASET_PATHS) * len(TARGET_COLUMNS) *
                    len(FEATURE_METHODS) * len(get_models_and_grids()))
    combo_idx = 0

    for ds_name, ds_path in DATASET_PATHS.items():
        print(f"\n{'━' * 85}")
        print(f"  DATASET: {ds_name}  ({ds_path})")
        print(f"{'━' * 85}")

        df, feature_cols = load_and_preprocess(ds_path)
        print(f"  Loaded {len(df)} samples with {len(feature_cols)} features.")

        for tgt_name, tgt_info in TARGET_COLUMNS.items():
            print(f"\n  ── Target: {tgt_name} ──")
            X, y = get_Xy(df, feature_cols, tgt_info["col"])
            pos_count = int(y.sum())
            neg_count = len(y) - pos_count
            print(f"     Class Distribution: {pos_count} positive ({pos_count/len(y)*100:.2f}%), {neg_count} negative")

            # Stratified train/test split (70/30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=seed, stratify=y
            )

            # Generate paired 30x bootstrap index sets once for this target
            bootstrap_indices = generate_paired_bootstrap_indices(
                y_test, n_bootstraps=n_boot, random_state=seed
            )

            for fe_name, fe_func in FEATURE_METHODS.items():
                try:
                    X_tr_fe, y_tr_fe, X_te_fe = fe_func(X_train.copy(), y_train.copy(), X_test.copy())
                except Exception as e:
                    print(f"     [SKIP] {fe_name}: feature engineering failed — {e}")
                    continue

                models_and_grids = get_models_and_grids()

                for model_name, (estimator, param_grid) in models_and_grids.items():
                    combo_idx += 1
                    t0 = time.time()

                    try:
                        # 5-fold Stratified CV for hyperparameter tuning
                        cv = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
                        grid = GridSearchCV(
                            estimator=estimator,
                            param_grid=param_grid,
                            scoring=SCORING,
                            cv=cv,
                            n_jobs=-1,
                            refit=True,
                            error_score="raise",
                        )
                        grid.fit(X_tr_fe, y_tr_fe)
                        best_model = grid.best_estimator_

                        # Predict on test set
                        y_prob = best_model.predict_proba(X_te_fe)[:, 1]
                        y_pred = best_model.predict(X_te_fe)

                        # Evaluate with 30x bootstrapping
                        metrics, replications = evaluate_with_bootstrapping(
                            y_test, y_prob, y_pred, bootstrap_indices
                        )
                        elapsed = time.time() - t0

                        # Store test set predictions for curve plotting
                        test_predictions_store[(ds_name, tgt_name, fe_name, model_name)] = {
                            "y_true": y_test.copy(),
                            "y_prob": y_prob.copy(),
                            "y_pred": y_pred.copy(),
                            "AUROC":  metrics["AUROC"],
                            "ACC":    metrics["ACC"],
                            "AUPRC":  metrics["AUPRC"],
                        }

                        # Save individual bootstrap draws for box plots
                        for rep in replications:
                            rep_row = {
                                "Dataset":     ds_name,
                                "Target":      tgt_name,
                                "Feature_Eng": fe_name,
                                "Model":       model_name,
                                **rep
                            }
                            all_bootstrap_records.append(rep_row)

                        result = {
                            "Dataset":         ds_name,
                            "Target":          tgt_name,
                            "Feature_Eng":     fe_name,
                            "Model":           model_name,
                            "Best_Params":     str(grid.best_params_),
                            "CV_AUROC":        round(grid.best_score_, 4),
                            "AUROC":           round(metrics["AUROC"], 4),
                            "AUROC_boot_mean": round(metrics["AUROC_boot_mean"], 4),
                            "AUROC_boot_std":  round(metrics["AUROC_boot_std"], 4),
                            "AUROC_95CI":      f"[{metrics['AUROC_ci_lower']:.3f}, {metrics['AUROC_ci_upper']:.3f}]",
                            "ACC":             round(metrics["ACC"], 4),
                            "ACC_boot_mean":   round(metrics["ACC_boot_mean"], 4),
                            "ACC_boot_std":    round(metrics["ACC_boot_std"], 4),
                            "ACC_95CI":        f"[{metrics['ACC_ci_lower']:.3f}, {metrics['ACC_ci_upper']:.3f}]",
                            "AUPRC":           round(metrics["AUPRC"], 4),
                            "AUPRC_boot_mean": round(metrics["AUPRC_boot_mean"], 4),
                            "AUPRC_boot_std":  round(metrics["AUPRC_boot_std"], 4),
                            "AUPRC_95CI":      f"[{metrics['AUPRC_ci_lower']:.3f}, {metrics['AUPRC_ci_upper']:.3f}]",
                            "Time_s":          round(elapsed, 1),
                        }
                        all_results.append(result)

                        print(f"     [{combo_idx}/{total_combos}] "
                              f"{ds_name} | {tgt_name} | {fe_name} | {model_name:<18} "
                              f"AUROC={metrics['AUROC']:.4f} (boot={metrics['AUROC_boot_mean']:.4f}±{metrics['AUROC_boot_std']:.3f}) | "
                              f"AUPRC={metrics['AUPRC']:.4f} | "
                              f"ACC={metrics['ACC']:.4f} ({elapsed:.1f}s)")

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"     [{combo_idx}/{total_combos}] "
                              f"{ds_name} | {tgt_name} | {fe_name} | {model_name:<18} "
                              f"FAILED: {e} ({elapsed:.1f}s)")
                        all_results.append({
                            "Dataset":         ds_name,
                            "Target":          tgt_name,
                            "Feature_Eng":     fe_name,
                            "Model":           model_name,
                            "Best_Params":     "FAILED",
                            "CV_AUROC":        np.nan,
                            "AUROC":           np.nan,
                            "AUROC_boot_mean": np.nan,
                            "AUROC_boot_std":  np.nan,
                            "AUROC_95CI":      "N/A",
                            "ACC":             np.nan,
                            "ACC_boot_mean":   np.nan,
                            "ACC_boot_std":    np.nan,
                            "ACC_95CI":        "N/A",
                            "AUPRC":           np.nan,
                            "AUPRC_boot_mean": np.nan,
                            "AUPRC_boot_std":  np.nan,
                            "AUPRC_95CI":      "N/A",
                            "Time_s":          round(elapsed, 1),
                        })

    # ── Save Results ──────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'=' * 85}")
    print(f"All model summary results saved to '{OUTPUT_FILE}' ({len(results_df)} rows)")

    # Save detailed 30x bootstrap draws
    boot_df = pd.DataFrame(all_bootstrap_records)
    boot_df.to_csv(BOOTSTRAP_FILE, index=False)
    print(f"All 30x bootstrap replications saved to '{BOOTSTRAP_FILE}' ({len(boot_df)} rows)")

    # Best model per (Dataset, Target) based on test AUROC
    valid = results_df.dropna(subset=["AUROC"])
    if not valid.empty:
        idx_best = valid.groupby(["Dataset", "Target"])["AUROC"].idxmax()
        best_df = valid.loc[idx_best].reset_index(drop=True)
        best_df.to_csv(BEST_FILE, index=False)
        print(f"Best models saved to '{BEST_FILE}'")
        print("\n" + "=" * 85)
        print("  BEST MODELS PER DATASET AND TARGET (ON 30% TEST SET)")
        print("=" * 85)
        cols_show = ["Dataset", "Target", "Feature_Eng", "Model", "AUROC", "AUROC_95CI", "AUPRC", "ACC"]
        print(best_df[cols_show].to_string(index=False))

        # Generate and save AUROC, AUPRC, and ACC curves matching Fig. 2 of the paper
        plot_performance_curves(test_predictions_store, results_df)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
