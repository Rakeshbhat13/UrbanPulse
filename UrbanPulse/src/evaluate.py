"""Evaluation: threshold selection on validation, final metrics on test."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


def select_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Pick threshold maximizing validation F1. Returns threshold and metrics."""
    val_probs = model.predict(X_val, verbose=0).ravel()
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.argmax(f1))

    return {
        "threshold": float(thresholds[best_idx]),
        "val_precision": float(precision[best_idx]),
        "val_recall": float(recall[best_idx]),
        "val_f1": float(f1[best_idx]),
        "val_probs": val_probs,
    }


def evaluate_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    meta_test: list[dict],
) -> dict:
    """Compute test metrics using the frozen threshold."""
    test_probs = model.predict(X_test, verbose=0).ravel()
    test_preds = (test_probs >= threshold).astype(int)

    report = classification_report(
        y_test, test_preds,
        target_names=["Normal", "Abnormal"],
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])

    # Per-clip confusion matrices
    results_df = pd.DataFrame(meta_test)
    results_df["actual"] = y_test
    results_df["probability"] = test_probs
    results_df["predicted"] = test_preds

    per_clip = {}
    for clip_id in sorted(results_df["clip_id"].unique()):
        rows = results_df[results_df["clip_id"] == clip_id]
        per_clip[int(clip_id)] = confusion_matrix(
            rows["actual"], rows["predicted"], labels=[0, 1]
        ).tolist()

    metrics = {
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_clip_cm": per_clip,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "results_df": results_df,
    }

    # Safe AUC computation (needs both classes)
    if len(np.unique(y_test)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_test, test_probs))
        metrics["roc_auc"] = float(roc_auc_score(y_test, test_probs))

    return metrics


def print_report(threshold_info: dict, test_info: dict) -> None:
    """Print a human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print("VALIDATION THRESHOLD SELECTION")
    print(f"{'='*60}")
    print(f"Threshold:  {threshold_info['threshold']:.4f}")
    print(f"Precision:  {threshold_info['val_precision']:.3f}")
    print(f"Recall:     {threshold_info['val_recall']:.3f}")
    print(f"F1:         {threshold_info['val_f1']:.3f}")

    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(test_info["confusion_matrix"]))
    print()
    report = test_info["classification_report"]
    for cls in ["Normal", "Abnormal"]:
        r = report[cls]
        print(f"  {cls:10s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")

    if "pr_auc" in test_info:
        print(f"\n  PR-AUC:  {test_info['pr_auc']:.3f}")
        print(f"  ROC-AUC: {test_info['roc_auc']:.3f}")

    print(f"\nPer-clip confusion matrices:")
    for clip_id, cm in test_info["per_clip_cm"].items():
        print(f"  Clip {clip_id}: {cm}")
