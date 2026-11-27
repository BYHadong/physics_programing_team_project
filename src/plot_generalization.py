"""선정 결과와 평가 원자료를 읽어 발표용 일반화 비교 PNG를 만듭니다."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.analyze_results import AXES, curves_and_confusion, save_figure
from src.config import COLAB_BACKUP_DIR, IMAGENET_MEAN, IMAGENET_STD
from src.dataset import build_transforms
from src.embedding_cache import SeededDataset, deterministic_seed, preserve_rng
from src.experiment_splits import load_manifest
from src.generalization_study import SEARCH
from src.linear_probe import Hyperparameters, training_recipe
from src.tune_hyperparameters import read_json

METRICS = {
    "hard_siamese_roc_auc": "Hard ROC-AUC",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1",
    "unseen_siamese_fpr": "Siamese FPR",
}


def plot_gap(trial: dict, output: Path) -> None:
    """동일 임계값에서 Fold별 원본 학습과 검증 성능을 그립니다."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    folds = trial["fold_results"]
    x = [fold["fold"] for fold in folds]
    for ax, (key, label) in zip(axes.flat, METRICS.items()):
        for split, display in (("train_clean", "Train originals"), ("validation", "Validation")):
            ax.plot(x, [row[split][key] for row in folds], "o-", label=display)
        ax.set(title=label, xlabel="Development fold", xticks=x, ylim=(-0.025, 1.025))
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"Train–validation gap | {trial['experiment']}")
    save_figure(fig, output / "overfitting_gap.png")


def plot_candidates(rows: list[dict], output: Path) -> None:
    """단계별 단일 변수 후보에서 AUC, F1, FPR 변화와 원본 증강 비율을 그립니다."""
    for parameter, _ in SEARCH:
        items = sorted(
            (row for row in rows if row["stage"] == parameter),
            key=lambda row: row["parameters"][parameter] or 0,
        )
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
        x = [row["parameters"][parameter] or 0 for row in items]
        for key in ("hard_siamese_roc_auc", "f1", "unseen_siamese_fpr"):
            ax.plot(x, [row["metrics"][key] for row in items], "o-", label=METRICS[key])
        x_label = AXES[parameter]
        if parameter in ("main_target_count", "neg_aug_passes"):
            ratio = (
                "positive_multiplier" if parameter == "main_target_count" else "negative_multiplier"
            )
            ax.set_xticks(
                x,
                [
                    f"{value if value or parameter != 'main_target_count' else 'No aug'}\n{np.mean([f[ratio] for f in item['fold_results']]):.1f}×"
                    for value, item in zip(x, items)
                ],
            )
            x_label += "\nMean training count / originals"
        if parameter == "c":
            ax.set_xscale("log")
            ax.set_xticks(x, [str(value) for value in x])
        ax.set(
            xlabel=x_label,
            title=f"Development OOF · n={items[0]['metrics']['total_count']}",
            ylim=(-0.025, 1.025),
        )
        ax.legend()
        save_figure(fig, output / f"{parameter}.png")


def plot_comparison(comparisons: dict, output: Path) -> None:
    """같은 개발 및 고정 평가에서 E00·E15·선정 모델을 비교합니다."""
    splits = ["oof", "holdout"] if comparisons["final"]["holdout"] else ["oof"]
    fig, axes = plt.subplots(
        1, len(splits), figsize=(6.1 * len(splits), 4.6), squeeze=False, layout="constrained"
    )
    x = np.arange(len(METRICS))
    for ax, split in zip(axes.flat, splits):
        for i, (label, row) in enumerate(comparisons.items()):
            scores = [row[split][key] for key in METRICS]
            ax.bar(x + (i - 1) * 0.25, scores, 0.24, label=f"{label.upper()} ({row['experiment']})")
        ax.set(
            title="Development OOF" if split == "oof" else "Reserved group holdout",
            xticks=x,
            xticklabels=list(METRICS.values()),
            ylim=(0, 1.06),
        )
        ax.tick_params(axis="x", labelrotation=16)
        ax.legend()
    save_figure(fig, output / "baseline_comparison.png")


def plot_resolution(rows: list[dict], output: Path) -> None:
    """해상도별 정확도와 관측한 캐시 포함 시간·GPU 피크를 그립니다."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, key, title in zip(
        axes.flat,
        (
            "hard_siamese_roc_auc",
            "f1",
            "unseen_siamese_fpr",
            "embedding_seconds",
            "classifier_fit_seconds",
            "peak_gpu_mib",
        ),
        (
            "Hard ROC-AUC",
            "F1",
            "Siamese FPR",
            "Feature extraction (s)",
            "Logistic fit (s)",
            "Peak GPU allocation (MiB)",
        ),
    ):
        values = [
            (str(row["image_size"]), value)
            for row in rows
            if (value := row.get(key, row["metrics"].get(key))) is not None
        ]
        if values:
            x, y = zip(*values)
            ax.bar(x, y, color="#276b88")
        else:
            ax.text(0.5, 0.5, "Unavailable", ha="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Input pixels")

    fig.suptitle(f"Development OOF | {rows[0]['experiment']} fixed conditions")
    save_figure(fig, output / "resolution_comparison.png")


def plot_data_ablation(ablation: dict, output: Path) -> None:
    """E15 양성 원본 그룹을 줄였을 때 같은 검증 Fold의 성능을 그립니다."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for key, label in (("full", "All training groups"), ("reduced", "One group removed")):
        folds = ablation[f"{key}_folds"]
        indices = [row["fold"] for row in folds]
        axes[0].plot(
            indices, [row["train_clean"]["main_count"] for row in folds], "o-", label=label
        )
        axes[1].plot(
            indices, [row["validation"]["hard_siamese_roc_auc"] for row in folds], "o-", label=label
        )
    axes[0].set(title="Main originals per training fold", xlabel="Development fold", xticks=indices)
    axes[1].set(title="Validation Hard ROC-AUC", xlabel="Development fold", xticks=indices)
    axes[0].legend()
    save_figure(fig, output / "positive_data_ablation.png")


def plot_diagnostics(trials: list[dict], selected: dict, config: dict, output: Path) -> None:
    """전체 후보·Fold 클래스·임계값·클래스별 평가 비교 PNG를 생성합니다."""
    n = selected["metrics"]["total_count"]
    caption = f"{selected['experiment']} | Development OOF · n={n} · seed={config['seed']}"
    keys = ("hard_siamese_roc_auc", "f1", "unseen_siamese_fpr")
    files = {
        "all_candidates.png": (trials, [r["experiment"] for r in trials], keys),
        "threshold_comparison.png": (
            list(selected["policies"].values()),
            list(selected["policies"]),
            ("precision", "recall", "unseen_siamese_fpr"),
        ),
        "fold_class_distribution.png": (
            [f["validation"] for f in selected["fold_results"]],
            [f"Fold {f['fold']}" for f in selected["fold_results"]],
            ("main_count", "other_count", "etc_count"),
        ),
        "fold_performance.png": (
            [f["validation"] for f in selected["fold_results"]],
            [f"Fold {f['fold']}" for f in selected["fold_results"]],
            keys,
        ),
        "class_performance.png": (
            [
                row
                for label, row in selected["policies"].items()
                if label in ("fixed_0_5", config["policy"])
            ],
            [label for label in selected["policies"] if label in ("fixed_0_5", config["policy"])],
            ("recall", "other_accuracy", "etc_accuracy"),
        ),
    }
    for name, (rows, labels, metrics) in files.items():
        path = output / name
        if path.is_file():
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.32), 4.3), layout="constrained")
        x = np.arange(len(rows))
        for offset, metric in enumerate(metrics):
            values = []
            for row in rows:
                row = row.get("metrics", row)
                value = row.get(metric)
                if metric == "etc_count":
                    value = row["total_count"] - row["main_count"] - row["other_count"]
                values.append(np.nan if value is None else value)
            ax.bar(
                x + (offset - 1) * 0.25,
                values,
                0.24,
                label=METRICS.get(metric, metric.replace("_", " ").title()),
            )
        ax.set(xticks=x, xticklabels=labels, title=caption)
        ax.tick_params(axis="x", rotation=55 if len(rows) > 5 else 0)
        ax.legend()
        save_figure(fig, path)


def plot_augmentation(results: Path, output: Path, processed: Path, selected: dict) -> None:
    """선정 설정에서 얼굴·귀·눈 주변을 육안으로 비교할 학습 샘플을 그립니다."""
    path = output / "augmentation_review.png"
    if path.is_file():
        return
    plan = read_json(results / "fold_indices.json")
    records = load_manifest(processed / "manifest.json", require_other_ids=False)
    params = Hyperparameters(**selected["parameters"])
    recipes = training_recipe(records, plan["folds"][0]["train"], params)
    indices = [i for i in recipes[0][0]][:2]
    examples = [
        [(i, aug) for group, aug, label in recipes if label == 1 and i in group][:5]
        for i in indices
    ]
    config = read_json(results / "experiment_config.json")
    fig, axes = plt.subplots(len(examples), 5, figsize=(11, 2.5 * len(examples)), squeeze=False)
    for axes_in_row, entries in zip(axes, examples):
        for ax in axes_in_row:
            ax.axis("off")
        for ax, (index, aug) in zip(axes_in_row, entries):
            flip = random.Random(deterministic_seed(index, aug, "flip")).random()
            transform = build_transforms(
                image_size=selected["image_size"],
                is_train=aug > 0,
                rotation_degrees=params.rotation_degrees,
                fixed_hflip=flip < params.hflip_probability,
            )
            dataset = SeededDataset(
                [records[index]], [index], aug, processed_dir=processed, transform=transform
            )
            with preserve_rng():
                tensor = dataset[0][0]
            pixels = tensor.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
            ax.imshow(np.clip(pixels, 0, 1))
            ax.set_title(f"#{index} · {'original' if aug == 0 else f'pass {aug}'}")
    fig.suptitle(f"{selected['experiment']} | Fold 1 training · seed {config['seed']}")
    save_figure(fig, path)


def generate(results: Path, output: Path, processed: Path | None = None) -> None:
    """완료된 실험의 그래프를 만들고 기존 출력을 보존합니다."""
    trials = read_json(results / "hyperparameter_results.json")
    report = read_json(results / "final_model_metrics.json")
    compare = report["comparisons"]
    selected = next(row for row in trials if row["experiment"] == report["selected_experiment"])
    if processed is not None:
        plot_augmentation(results, output, processed, selected)
    plot_diagnostics(trials, selected, read_json(results / "experiment_config.json"), output)
    if not (output / "overfitting_gap.png").is_file():
        plot_gap(selected, output)
    missing_candidates = [row for row in trials if not (output / f"{row['stage']}.png").is_file()]
    plot_candidates(missing_candidates, output)
    if not (output / "baseline_comparison.png").is_file():
        plot_comparison(compare, output)
    ablation_path = results / "data_ablation.json"
    if ablation_path.is_file() and not (output / "positive_data_ablation.png").is_file():
        ablation = read_json(ablation_path)
        if "reduced_folds" in ablation:
            plot_data_ablation(ablation, output)
    if not (output / "resolution_comparison.png").is_file():
        by_id = {row["experiment"]: row for row in trials}
        names = read_json(results / "best_hyperparameters.json")["resolution_experiments"]
        sizes = [by_id[name] for name in names]
        plot_resolution(sizes, output)

    held = None
    if compare["final"]["holdout"] is not None:
        held = {
            model: {
                "predictions": compare[model]["holdout_predictions"],
                "metrics": compare[model]["holdout"],
            }
            for model in ("baseline", "final")
        }
    if any(
        not (output / name).is_file()
        for name in ("roc_curve.png", "precision_recall_curve.png", "confusion_matrix.png")
    ):
        curves_and_confusion(trials, {**report, "reserved_group_holdout": held}, output)
    print(f"[그래프] {output}")


def main() -> None:
    """Drive에 저장된 새 실험 결과로 필요한 그래프를 다시 만듭니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COLAB_BACKUP_DIR)
    parser.add_argument("--run-id", default="dinov2-final-review-1121-01")
    parser.add_argument("--processed", type=Path)
    args = parser.parse_args()
    generate(
        args.output / "results" / args.run_id, args.output / "graphs" / args.run_id, args.processed
    )


if __name__ == "__main__":
    main()
