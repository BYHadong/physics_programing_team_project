"""완료된 실험 결과에서 발표용 PNG 그래프를 생성합니다."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from src.config import EXPERIMENTS_DIR
from src.experiment_splits import MAIN, OTHER
from src.tune_hyperparameters import SEARCH_STAGES, read_json, write_artifact

matplotlib.use("Agg")

PERFORMANCE = {
    "hard_siamese_roc_auc": "Hard ROC-AUC",
    "overall_roc_auc": "Overall ROC-AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "unseen_siamese_fpr": "Unseen Siamese FPR",
}
AXES = {
    "c": "Logistic Regression C",
    "main_target_count": "Main training images per fold",
    "neg_aug_passes": "Additional negative augmentation passes",
    "rotation_degrees": "Maximum absolute rotation (degrees)",
    "hflip_probability": "Horizontal flip probability",
}


def save_figure(figure: plt.Figure, output: Path) -> None:
    """기존 결과를 보존하면서 발표용 220 DPI PNG를 저장합니다."""
    try:
        stream = BytesIO()
        figure.savefig(stream, format="png", dpi=220, bbox_inches="tight")
        write_artifact(output, stream.getvalue())
    finally:
        plt.close(figure)


def candidate_graphs(trials: list[dict], output: Path) -> None:
    """동일 단계의 단일 변수 후보에 대해 지표 변화 그래프를 생성합니다."""
    for stage, field, _ in SEARCH_STAGES:
        rows = sorted(
            (row for row in trials if row["stage"] == stage),
            key=lambda row: row["parameters"][field],
        )
        values = [row["parameters"][field] for row in rows]
        figure, axes = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
        ranking = ("hard_siamese_roc_auc", "overall_roc_auc")
        for key, label in PERFORMANCE.items():
            ax = axes[0 if key in ranking else 1]
            ax.plot(values, [r["metrics"][key] for r in rows], "o-", label=label)
        if field == "c":
            axes[1].set_xscale("log")
        axes[1].set_xticks(values, labels=[str(v) for v in values])
        axes[1].set_xlabel(AXES[field])
        axes[0].set_title(f"{AXES[field]} | development 5-fold OOF")
        for ax in axes:
            ax.set_ylabel("Score")
            ax.grid(alpha=0.22)
            ax.legend(loc="best", ncol=2)
        save_figure(figure, output / f"{stage}.png")


def model_comparison(reports: dict, output: Path) -> None:
    """같은 분할에서 측정된 E00과 최종 후보의 지표를 비교합니다."""
    held = reports["reserved_group_holdout"]
    metrics = (
        {key: row["metrics"] for key, row in held.items()}
        if held
        else reports["development_oof_tuning"]
    )
    labels = ["Baseline E00", f"Selected {reports['selected_experiment']}"]
    figure, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(PERFORMANCE))
    for i, model in enumerate(("baseline", "final")):
        readings = [metrics[model][metric] for metric in PERFORMANCE]
        bars = ax.bar(x + (i - 0.5) * 0.34, readings, width=0.34, label=labels[i])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, PERFORMANCE.values(), rotation=18, ha="right")
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score")
    ax.set_title("Reserved group holdout" if held else "Development OOF (used for tuning)")
    ax.legend()
    save_figure(figure, output / "baseline_vs_final.png")


def curves_and_confusion(trials: list[dict], reports: dict, output: Path) -> None:
    """평가 분할의 Hard/Overall ROC, PR 및 최종 혼동 행렬을 저장합니다."""
    held = reports["reserved_group_holdout"]
    by_id = {row["experiment"]: row for row in trials}
    selected = by_id[reports["selected_experiment"]]
    title = "Reserved group holdout" if held else "Development OOF (used for tuning)"
    roc_fig, roc_axes = plt.subplots(1, 2, figsize=(10, 4.3))
    pr_fig, pr_ax = plt.subplots(figsize=(6, 4.6))
    for model, trial in (("baseline", trials[0]), ("final", selected)):
        data = held[model]["predictions"] if held else trial["oof_predictions"]
        metrics = held[model]["metrics"] if held else trial["metrics"]
        classes, scores = np.asarray(data["classes"]), np.asarray(data["scores"])
        y = classes == MAIN
        for ax, label, mask, key in (
            (roc_axes[0], "Hard", y | (classes == OTHER), "hard_siamese_roc_auc"),
            (roc_axes[1], "Overall", np.ones(len(y), dtype=bool), "overall_roc_auc"),
        ):
            fpr, tpr, _ = roc_curve(y[mask], scores[mask])
            ax.plot(fpr, tpr, label=f"{trial['experiment']} (AUC {metrics[key]:.3f})")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title=label)
        precision, recall, _ = precision_recall_curve(y, scores)
        pr_ax.plot(
            recall,
            precision,
            label=f"{trial['experiment']} (AP {metrics['average_precision']:.3f})",
        )
    for ax in roc_axes:
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        ax.legend(loc="lower right")
    roc_fig.suptitle(f"ROC | {title}")
    save_figure(roc_fig, output / "roc_curve.png")
    pr_ax.set(title=f"Precision–Recall | {title}", xlabel="Recall", ylabel="Precision")
    pr_ax.legend()
    save_figure(pr_fig, output / "precision_recall_curve.png")
    matrix = (held["final"]["metrics"] if held else selected["metrics"])["confusion_matrix"]
    figure, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["Other cats", "Main"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix | {title}")
    save_figure(figure, output / "confusion_matrix.png")


def fold_graph(trials: list[dict], reports: dict, output: Path) -> None:
    """기준과 선정 후보의 개발 Fold별 Hard ROC-AUC를 표시합니다."""
    selected = next(row for row in trials if row["experiment"] == reports["selected_experiment"])
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for title, row in (("Baseline", trials[0]), ("Selected", selected)):
        metric = "hard_siamese_roc_auc"
        data = row["fold_results"]
        summary = row["fold_summary"][metric]
        ax.plot(
            [f["fold"] for f in data],
            [f["metrics"][metric] for f in data],
            "o-",
            label=f"{title} {row['experiment']} ({summary['mean']:.3f} ± {summary['std']:.3f})",
        )
    ax.set(
        title="Development Fold stability",
        xlabel="Fold",
        ylabel="Hard ROC-AUC",
        xticks=[f["fold"] for f in selected["fold_results"]],
    )
    ax.legend()
    save_figure(figure, output / "fold_comparison.png")


def generate_graphs(results: Path, output: Path) -> None:
    """저장된 결과에서 단일 변수, 모델 비교 및 평가용 그래프를 생성합니다."""
    trials = read_json(results / "hyperparameter_results.json")
    reports = read_json(results / "final_model_metrics.json")
    candidate_graphs(trials, output)
    model_comparison(reports, output)
    curves_and_confusion(trials, reports, output)
    fold_graph(trials, reports, output)


def main() -> None:
    """실험 결과 및 그래프 경로를 받아 PNG 생성기를 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ("", ".", ".."):
        parser.error("올바른 run-id를 지정하세요.")
    generate_graphs(args.output / "results" / args.run_id, args.output / "graphs" / args.run_id)


if __name__ == "__main__":
    main()
