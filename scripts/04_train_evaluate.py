"""
04_train_evaluate.py
--------------------
Extracts features from ComBat site-harmonized functional connectivity matrices
and trains a gradient-boosted classifier to distinguish ASD from controls.

Classification approach
-----------------------
The full 200x200 Fisher z matrix contains 19,900 unique pairwise connections
(upper triangle). Feeding all 19,900 directly to a classifier with 303 subjects
is severely underpowered and would overfit. Instead, this script uses PCA
(Principal Component Analysis) to compress the 19,900-dimensional connectivity
profile of each subject into a compact set of components that capture the most
variance across subjects.

PCA works by finding directions in the high-dimensional feature space along
which subjects vary most. The first component captures the single direction of
greatest variance, the second captures the next most after removing the first,
and so on. Each subject is then represented by their scores on these components
rather than the raw 19,900 values.

Using the top N_COMPONENTS = 50 principal components retains the dominant
structure of the connectivity data while reducing dimensionality to a ratio
that a gradient-boosted classifier can learn from reliably. This is the
standard approach used in published ABIDE classification papers that report
AUC 0.65-0.75 on harmonized data.

Note: PCA is fit on the training set only within each fold, then applied to
the test set. This prevents information leakage from test subjects into the
feature extraction step.

Classifier: gradient-boosted decision trees (scikit-learn GradientBoostingClassifier)
  - builds trees sequentially, each correcting errors of the previous
  - well-suited to small tabular datasets; does not require GPU

Evaluation: 5-fold stratified cross-validation
  - each fold uses ~242 subjects for training and ~61 for testing
  - stratified to preserve ASD/Control ratio across folds
  - all reported metrics are averaged over the 5 held-out test sets

Usage
-----
  python scripts/04_train_evaluate.py

Inputs
------
  data/connectomes_harmonized.npy   from 02_harmonize.py
  data/labels.npy                   from 01_fetch_and_prepare.py
  data/graphs.pkl                   from 03_build_graphs.py (for node importance)
  data/roi_meta.pkl                 from 01_fetch_and_prepare.py

Outputs
-------
  results/cv_results.pkl   fold results, predicted probabilities, node importance
  results/metrics.csv      summary table of mean +/- SD for each metric

References
----------
Di Martino A, et al. (2014). The autism brain imaging data exchange.
  Mol Psychiatry, 19(6):659-667. doi:10.1038/mp.2013.78.

Fortin JP, et al. (2017). Harmonization of multi-site diffusion tensor
  imaging data. NeuroImage, 161:149-170. doi:10.1016/j.neuroimage.2017.08.047.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_COMPONENTS = 50    # PCA components: captures dominant connectivity structure
N_FOLDS      = 5
RANDOM_SEED  = 42

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load harmonized connectomes, labels, graphs, and ROI metadata."""
    connectomes = np.load(DATA_DIR / "connectomes_harmonized.npy")
    labels      = np.load(DATA_DIR / "labels.npy")

    with open(DATA_DIR / "graphs.pkl",   "rb") as f:
        graphs = pickle.load(f)
    with open(DATA_DIR / "roi_meta.pkl", "rb") as f:
        roi_meta = pickle.load(f)

    networks = np.array(roi_meta["networks"])

    return connectomes, labels, graphs, networks


def extract_upper_triangle(connectomes):
    """
    Extract upper triangle values from each connectivity matrix as a
    flat feature vector.

    For a symmetric 200x200 matrix, the upper triangle contains
    200 * 199 / 2 = 19,900 unique pairwise connectivity values.

    Parameters
    ----------
    connectomes : (N, 200, 200) array

    Returns
    -------
    X : (N, 19900) float64 array
    """
    N, R, _ = connectomes.shape
    tri_idx  = np.triu_indices(R, k=1)
    return np.stack([c[tri_idx] for c in connectomes])


def run_cv(X_raw, labels, graphs, networks):
    """
    Run 5-fold stratified cross-validation with PCA + Gradient Boosting.

    PCA is fit on the training set within each fold and applied to both
    training and test sets. This ensures no information from test subjects
    leaks into the feature extraction step.

    Parameters
    ----------
    X_raw    : (N, 19900) raw upper triangle feature matrix
    labels   : (N,) int array
    graphs   : list of graph dicts (for node importance)
    networks : (200,) array

    Returns
    -------
    fold_results : list of per-fold result dicts
    all_probs    : (N,) float array of predicted ASD probabilities
    node_imp     : (200, 5) float array of node importance
    """
    N = len(labels)

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    all_probs    = np.zeros(N)
    node_imp     = np.zeros((graphs[0]["x"].shape[0], 5))

    clf = GradientBoostingClassifier(
        n_estimators  = 200,
        max_depth     = 3,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = RANDOM_SEED,
    )

    for fold, (tr, te) in enumerate(skf.split(X_raw, labels)):
        # Fit PCA and scaler on training set only
        scaler = StandardScaler()
        pca    = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)

        X_tr = pca.fit_transform(scaler.fit_transform(X_raw[tr]))
        X_te = pca.transform(scaler.transform(X_raw[te]))

        clf.fit(X_tr, labels[tr])

        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        all_probs[te] = probs

        acc  = accuracy_score(labels[te], preds)
        auc  = roc_auc_score(labels[te], probs)
        cm   = confusion_matrix(labels[te], preds, labels=[0, 1])
        sens = cm[1, 1] / max(cm[1].sum(), 1)
        spec = cm[0, 0] / max(cm[0].sum(), 1)

        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  Fold {fold + 1}: Acc={acc:.3f}  AUC={auc:.3f}"
              f"  Sens={sens:.3f}  Spec={spec:.3f}"
              f"  (PCA variance explained: {var_explained:.1%})")

        fold_results.append({
            "fold": fold + 1,
            "acc":  acc,
            "auc":  auc,
            "sens": sens,
            "spec": spec,
            "cm":   cm,
            "te_idx": te,
            "pca_var_explained": float(var_explained),
        })

        # Node importance: mean |ASD - Control| difference in raw node features
        asd_feats  = np.stack([graphs[i]["x"] for i in te if labels[i] == 1])
        ctrl_feats = np.stack([graphs[i]["x"] for i in te if labels[i] == 0])
        node_imp  += np.nan_to_num(np.abs(asd_feats.mean(0) - ctrl_feats.mean(0)))

    node_imp /= N_FOLDS

    return fold_results, all_probs, node_imp


def print_summary(fold_results):
    """Print mean +/- SD for each metric across folds."""
    print("\n=== Summary ===")
    rows = []

    metric_labels = {
        "acc":  "Accuracy",
        "auc":  "AUC-ROC",
        "sens": "Sensitivity (ASD)",
        "spec": "Specificity (CTRL)",
    }

    for key, label in metric_labels.items():
        vals   = [r[key] for r in fold_results]
        mu, sd = np.mean(vals), np.std(vals)
        print(f"  {label:25s}: {mu:.3f} +/- {sd:.3f}")
        rows.append({"Metric": label, "Mean": round(mu, 3), "SD": round(sd, 3)})

    mean_var = np.mean([r["pca_var_explained"] for r in fold_results])
    print(f"\n  PCA ({N_COMPONENTS} components) explains {mean_var:.1%} of connectivity variance on average")

    return rows


def save(fold_results, all_probs, labels, node_imp, summary_rows):
    """Save cross-validation results and summary metrics."""
    results = {
        "fold_results": fold_results,
        "all_probs":    all_probs,
        "labels":       labels,
        "node_imp":     node_imp,
    }

    with open(RESULTS_DIR / "cv_results.pkl", "wb") as f:
        pickle.dump(results, f)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\nSaved results/")
    print("  cv_results.pkl: fold results, probabilities, node importance")
    print("  metrics.csv: summary table")
    print("\nNext: python scripts/05_figures.py")


def main():
    connectomes, labels, graphs, networks = load_data()
    N = len(labels)

    print(f"Subjects: {N}  ASD={labels.sum()}  CTRL={(labels == 0).sum()}")

    print("\nExtracting upper triangle features ...")
    X_raw = extract_upper_triangle(connectomes)
    print(f"Feature matrix: {X_raw.shape}  ({X_raw.shape[1]} connectivity values per subject)")
    print(f"PCA will reduce to {N_COMPONENTS} components within each fold")

    print(f"\n5-Fold Cross-Validation (PCA + Gradient Boosting on harmonized connectomes)\n")
    fold_results, all_probs, node_imp = run_cv(X_raw, labels, graphs, networks)

    summary_rows = print_summary(fold_results)
    save(fold_results, all_probs, labels, node_imp, summary_rows)


if __name__ == "__main__":
    main()