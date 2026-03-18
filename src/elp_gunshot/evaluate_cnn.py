# src/elp_gunshot/evaluate_cnn.py
"""
Generate publication-quality figures from a completed gunshot CNN training run.

Usage:
    python -m elp_gunshot.evaluate_cnn --run_dir runs/model3_nomask_bs64_lr3e-05_e40_20260318_120000
    python -m elp_gunshot.evaluate_cnn --run_dir runs/... --output_dir results/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_fig(fig, output_dir: Path, stem: str) -> None:
    """Save figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_training_curves(history_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history_df["epoch"], history_df["loss"], label="Train loss")
    if "val_loss" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    if "auc" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["auc"], label="Train AUC")
    if "val_auc" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_auc"], label="Val AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Training / Validation AUC")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_fig(fig, output_dir, "training_curves")


def plot_confusion_matrix(cm: dict, output_dir: Path) -> None:
    required_keys = {"tp", "tn", "fp", "fn"}
    missing = required_keys - cm.keys()
    if missing:
        raise ValueError(f"confusion_matrix missing keys: {sorted(missing)}")

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = matrix.sum()
    if total == 0:
        raise ValueError("confusion_matrix has all zeros; cannot plot percentages.")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = 100.0 * count / total
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\\n({pct:.1f}%)", ha="center", va="center", color=color)

    fig.tight_layout()
    save_fig(fig, output_dir, "confusion_matrix")


def plot_roc_curve(preds_df: pd.DataFrame, output_dir: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = preds_df["y_true"].values
    y_score = preds_df["y_score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_fig(fig, output_dir, "roc_curve")


def plot_pr_curve(preds_df: pd.DataFrame, output_dir: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = preds_df["y_true"].values
    y_score = preds_df["y_score"].values

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()
    save_fig(fig, output_dir, "pr_curve")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from a gunshot CNN training run.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Run directory with training artifacts")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to write figures (default: results/figures)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    output_dir: Path = args.output_dir

    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "test_metrics.json"
    preds_path = run_dir / "test_predictions.csv"

    for path in (history_path, metrics_path, preds_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    history_df = pd.read_csv(history_path)
    test_metrics = json.loads(metrics_path.read_text())
    preds_df = pd.read_csv(preds_path)

    required_pred_cols = {"y_true", "y_score"}
    missing_pred_cols = required_pred_cols - set(preds_df.columns)
    if missing_pred_cols:
        raise ValueError(f"test_predictions.csv missing columns: {sorted(missing_pred_cols)}")

    cm = test_metrics.get("confusion_matrix", {})

    print(f"\nMetrics summary - {run_dir.name}")
    print("-" * 40)
    for key, value in test_metrics.items():
        if key != "confusion_matrix":
            print(f"  {key:<20} {value}")
    print("\n  Confusion matrix:")
    print(f"    TP={cm.get('tp')}  FP={cm.get('fp')}")
    print(f"    FN={cm.get('fn')}  TN={cm.get('tn')}")

    print(f"\nGenerating figures -> {output_dir}")
    plot_training_curves(history_df, output_dir)
    plot_confusion_matrix(cm, output_dir)
    plot_roc_curve(preds_df, output_dir)
    plot_pr_curve(preds_df, output_dir)

    print("\nDone. 4 figures saved (PDF + PNG each).")


if __name__ == "__main__":
    main()
