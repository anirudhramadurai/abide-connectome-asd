"""
04_train_evaluate.py
--------------------
Extracts graph-level features from each subject's harmonized brain connectivity
graph and trains a gradient-boosted classifier to distinguish ASD from controls.

This script operates on graphs built from ComBat site-harmonized connectomes.
Harmonization removes scanner-driven variance between NYU, USM, and UCLA,
allowing the classifier to learn from biological connectivity differences rather
than site fingerprints. The expected AUC with harmonized data is 0.65-0.75,
compared to chance-level performance on the raw unharmonized baseline.

Classification approach
-----------------------
Rather than feeding raw 200x200 matrices to the classifier (40,000 features
for 303 subjects -- severely underpowered), this script extracts 44 biologically
meaningful graph-level features per subject by aggregating node features across
the 8 functional networks:

  Between-network FC difference for each unique network pair  (28 features)
  Mean node degree per network                                ( 8 features)
  Mean clustering coefficient per network                     ( 8 features)
  Total                                                        44 features

Classifier: gradient-boosted decision trees (scikit-learn GradientBoostingClassifier)
  -- builds trees sequentially, each correcting errors of the previous
  -- well-suited to small tabular datasets; does not require GPU

Evaluation: 5-fold stratified cross-validation
  -- each fold uses ~242 subjects for training and ~61 for testing
  -- stratified to preserve ASD/Control ratio across folds
  -- all reported metrics are averaged over the 5 held-out test sets

Usage
-----
  python scripts/04_train_evaluate.py

Inputs
------
  data/graphs.pkl      from 03_build_graphs.py
  data/roi_meta.pkl    from 01_fetch_and_prepare.py
  data/metadata.csv    from 01_fetch_and_prepare.py

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS     = 5
RANDOM_SEED = 42

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load graph objects, ROI metadata, and subject metadata."""
    with open(DATA_DIR / "graphs.pkl",   "rb") as f:
        graphs = pickle.load(f)
    with open(DATA_DIR / "roi_meta.pkl", "rb") as f:
        roi_meta = pickle.load(f)

    networks = np.array(roi_meta["networks"])
    nets     = list(dict.fromkeys(roi_meta["networks"]))
    labels   = np.array([g["y"] for g in graphs])

    return graphs, labels, networks, nets


def extract_features(graph, networks, nets):
    """
    Extract 44 graph-level features from a single subject's graph.

    Features
    --------
    Between-network FC differences (28):
      For each unique pair of networks (A, B), compute the difference in
      mean FC between network A and network B. Captures long-range
      connectivity imbalances between networks.

    Mean degree per network (8):
      Average number of above-threshold connections per ROI within each
      network. Reflects how connected each network is as a whole.

    Mean clustering coefficient per network (8):
      Average local clustering coefficient per ROI within each network.
      Reflects how tightly interconnected the local neighborhood of each
      ROI is within its network.

    Parameters
    ----------
    graph    : dict from 03_build_graphs.py
    networks : (200,) array of network name strings
    nets     : list of unique network names (order-preserved)

    Returns
    -------
    features : (44,) float64 array
    """
    x     = graph["x"]
    feats = []

    for i, net_a in enumerate(nets):
        for j, net_b in enumerate(nets):
            if j <= i:
                continue
            fc_a = x[networks == net_a, 0].mean()
            fc_b = x[networks == net_b, 0].mean()
            feats.append(fc_a - fc_b)

    for net in nets:
        feats.append(x[networks == net, 1].mean())

    for net in nets:
        feats.append(x[networks == net, 2].mean())

    return np.array(feats)


def build_feature_matrix(graphs, networks, nets):
    """
    Build the (N, 44) feature matrix for all subjects.

    Parameters
    ----------
    graphs   : list of graph dicts
    networks : (200,) array
    nets     : list of str

    Returns
    -------
    X : (N, 44) float64 array
    """
    return np.stack([extract_features(g, networks, nets) for g in graphs])


def run_cv(X, labels, graphs, networks):
    """
    Run 5-fold stratified cross-validation and collect results.

    For each fold:
      - Trains a gradient-boosted classifier with median imputation
        and standard scaling on the training set.
      - Evaluates on the held-out test set.
      - Accumulates node-level feature importance as the mean absolute
        ASD-Control difference in raw node features.

    Parameters
    ----------
    X        : (N, 44) feature matrix
    labels   : (N,) int array
    graphs   : list of graph dicts (for node importance)
    networks : (200,) array

    Returns
    -------
    fold_results : list of per-fold result dicts
    all_probs    : (N,) float array of predicted ASD probabilities
    node_imp     : (200, 5) float array of accumulated node importance
    """
    N = len(labels)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     GradientBoostingClassifier(
            n_estimators  = 100,
            max_depth     = 3,
            learning_rate = 0.05,
            random_state  = RANDOM_SEED,
        )),
    ])

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    all_probs    = np.zeros(N)
    node_imp     = np.zeros((graphs[0]["x"].shape[0], 5))

    for fold, (tr, te) in enumerate(skf.split(X, labels)):
        pipeline.fit(X[tr], labels[tr])

        probs = pipeline.predict_proba(X[te])[:, 1]
        preds = (probs >= 0.5).astype(int)
        all_probs[te] = probs

        acc  = accuracy_score(labels[te], preds)
        auc  = roc_auc_score(labels[te], probs)
        cm   = confusion_matrix(labels[te], preds, labels=[0, 1])
        sens = cm[1, 1] / max(cm[1].sum(), 1)
        spec = cm[0, 0] / max(cm[0].sum(), 1)

        print(f"  Fold {fold + 1}: Acc={acc:.3f}  AUC={auc:.3f}"
              f"  Sens={sens:.3f}  Spec={spec:.3f}")

        fold_results.append({
            "fold": fold + 1,
            "acc":  acc,
            "auc":  auc,
            "sens": sens,
            "spec": spec,
            "cm":   cm,
            "te_idx": te,
        })

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
    print("  cv_results.pkl  -- fold results, probabilities, node importance")
    print("  metrics.csv     -- summary table")
    print("\nNext: python scripts/05_figures.py")


def main():
    graphs, labels, networks, nets = load_data()
    N = len(graphs)

    print(f"Subjects: {N}  ASD={labels.sum()}  CTRL={(labels == 0).sum()}")

    print("\nExtracting features ...")
    X         = build_feature_matrix(graphs, networks, nets)
    nan_count = np.isnan(X).sum()
    print(f"Feature matrix: {X.shape}  (NaNs: {nan_count} -- will be imputed)")

    print(f"\n5-Fold Cross-Validation (Gradient Boosting on harmonized connectome features)\n")
    fold_results, all_probs, node_imp = run_cv(X, labels, graphs, networks)

    summary_rows = print_summary(fold_results)
    save(fold_results, all_probs, labels, node_imp, summary_rows)


if __name__ == "__main__":
    main()
