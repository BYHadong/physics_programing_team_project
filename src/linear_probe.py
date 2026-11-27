"""동결 특징 기반 LogisticRegression 학습, 임계값 보정 및 평가를 제공합니다."""

from __future__ import annotations

import math
from time import perf_counter
import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.config import (
    DEFAULT_HFLIP_PROBABILITY,
    DEFAULT_ROTATION_DEGREES,
    LOGISTIC_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    PREPROCESSED_SIZE,
    RANDOM_SEED,
    TARGET_UNSEEN_FPR,
)
from src.embedding_cache import EmbeddingCache
from src.experiment_splits import MAIN, OTHER

THRESHOLD_POLICIES = ("max_f1", "precision_recall_balance", "min_siamese_fpr", "target_siamese_fpr")
METRIC_DEFINITIONS = {
    "hard_siamese_roc_auc": "ROC-AUC for MAIN positive and OTHER negative",
    "overall_roc_auc": "ROC-AUC for MAIN positive and OTHER + ETC negative",
    "accuracy": "(TP + TN) / (TP + TN + FP + FN)",
    "balanced_accuracy": "(sensitivity + specificity) / 2",
    "precision": "TP / (TP + FP), 0 when there is no predicted positive",
    "recall": "TP / (TP + FN)",
    "f1": "2 * precision * recall / (precision + recall), 0 for a zero denominator",
    "unseen_siamese_fpr": "predicted MAIN among true OTHER / true OTHER",
    "average_precision": "recall-weighted precision over thresholds, sklearn AP",
    "confusion_matrix": "[[TN, FP], [FN, TP]], positive is MAIN",
    "threshold": "predict MAIN when score >= threshold",
    "fold_std": "population standard deviation (ddof=0), undefined folds omitted",
    "null": "metric undefined when a required class is absent",
}


@dataclass(frozen=True)
class Hyperparameters:
    """대칭 7회 기준 설정 또는 증강 목표 수량 기반 학습 조건을 나타냅니다."""

    c: float = 1.0
    main_target_count: int | None = None
    positive_aug_passes: int = 7
    neg_aug_passes: int = 7
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES
    hflip_probability: float = DEFAULT_HFLIP_PROBABILITY

    def __post_init__(self) -> None:
        """유효한 정규화 강도, 수량, 회전 범위, 반전 확률을 확인합니다."""
        if not math.isfinite(self.c) or self.c <= 0:
            raise ValueError("LogisticRegression C는 유한한 양수여야 합니다.")
        if self.main_target_count is not None and (
            type(self.main_target_count) is not int or self.main_target_count < 1
        ):
            raise ValueError("양성 목표 수량은 양의 정수여야 합니다.")
        for passes in (self.positive_aug_passes, self.neg_aug_passes):
            if type(passes) is not int or passes < 0:
                raise ValueError("추가 증강 패스는 0 이상의 정수여야 합니다.")
        if not 0 <= self.rotation_degrees <= 180 or not 0 <= self.hflip_probability <= 1:
            raise ValueError("회전 범위는 0~180도, 좌우 반전 확률은 0~1이어야 합니다.")


def training_recipe(
    records: list[dict], train: list[int], params: Hyperparameters
) -> list[tuple[list[int], int, int]]:
    """학습 원본 배치별 인덱스, 증강 패스 번호, 이진 라벨을 반환합니다."""
    positive = sorted(i for i in train if records[i]["class_name"] == MAIN)
    negative = sorted(i for i in train if records[i]["class_name"] != MAIN)
    if not positive or not negative:
        raise ValueError("Linear Probe 학습에 양성 및 음성 이미지가 모두 필요합니다.")
    target = params.main_target_count
    if target is None:
        positive_recipe = [(positive, aug, 1) for aug in range(params.positive_aug_passes + 1)]
    else:
        # 원본은 모두 유지하고 증가하는 목표 수량이 이전 세트의 상위 집합이 되게 합니다.
        positive_recipe = [(positive, 0, 1)]
        remaining = target - len(positive)
        rng = random.Random(RANDOM_SEED)
        aug = 1
        while remaining > 0:
            order = positive.copy()
            rng.shuffle(order)
            positive_recipe.append((order[:remaining], aug, 1))
            remaining -= len(positive)
            aug += 1
    return positive_recipe + [(negative, aug, 0) for aug in range(params.neg_aug_passes + 1)]


def fit_probe(
    cache: EmbeddingCache, train: list[int], params: Hyperparameters, timing: dict | None = None
) -> tuple[LogisticRegression, np.ndarray, dict]:
    """학습 인덱스에서만 증강하여 L2 분류기, 프로토타입, 증강 수량을 계산합니다."""
    matrices, labels = [], []
    for indices, aug_pass, label in training_recipe(cache.records, train, params):
        matrix = cache.get(indices, aug_pass, params.rotation_degrees, params.hflip_probability)
        matrices.append(matrix)
        labels.append(np.full(len(indices), label, dtype=np.int64))
    features, y = np.vstack(matrices), np.concatenate(labels)
    classifier = LogisticRegression(
        C=params.c,
        solver=LOGISTIC_SOLVER,
        class_weight=LOGISTIC_CLASS_WEIGHT,
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    # lbfgs의 기본 L2를 유지합니다. 최신 sklearn에서 폐기 예정인 penalty 인자는 생략합니다.
    started = perf_counter()
    classifier.fit(features, y)
    if timing is not None:
        timing["classifier_fit_seconds"] = timing.get("classifier_fit_seconds", 0.0) + (
            perf_counter() - started
        )
    prototype = features[y == 1].mean(axis=0)
    norm = np.linalg.norm(prototype)
    if norm > 0:
        prototype /= norm
    return (
        classifier,
        prototype,
        {
            "positive_original": sum(cache.records[i]["class_name"] == MAIN for i in train),
            "negative_original": sum(cache.records[i]["class_name"] != MAIN for i in train),
            "positive_training": int(y.sum()),
            "negative_training": int((1 - y).sum()),
        },
    )


def probe_scores(
    features: np.ndarray,
    classifier: LogisticRegression,
    prototype: np.ndarray,
    method: str,
) -> np.ndarray:
    """기존 두 판별 방식의 점수를 반환하며 신규 실험은 Linear Probe만 호출합니다."""
    if method == "prototype_cosine":
        return features @ prototype
    return expit(
        np.asarray(features, dtype=np.float64) @ classifier.coef_[0]
        + float(classifier.intercept_[0])
    )


def fold_scores(
    cache: EmbeddingCache,
    development: list[int],
    folds: list[dict],
    params: Hyperparameters,
    method: str = "linear_probe",
) -> tuple[np.ndarray, list[dict]]:
    """학습과 겹치지 않는 클린 검증 점수와 Fold별 학습 장수를 반환합니다."""
    position = {index: offset for offset, index in enumerate(development)}
    scores = np.full(len(development), np.nan)
    fold_counts = []
    for fold in folds:
        classifier, prototype, counts = fit_probe(cache, fold["train"], params)
        validation = fold["validation"]
        if validation:
            scores[[position[i] for i in validation]] = probe_scores(
                cache.get(validation), classifier, prototype, method
            )
        fold_counts.append(counts)
    if np.isnan(scores).any():
        raise ValueError("OOF 예측에서 누락된 검증 인덱스가 있습니다.")
    return scores, fold_counts


def evaluate(classes: list[str] | np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """고정 임계값과 순위 점수에서 README와 동일한 AUC 및 추가 지표를 반환합니다."""
    classes = np.asarray(classes, dtype=str)
    y = classes == MAIN
    other = classes == OTHER
    hard = y | other
    pred = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[False, True]).ravel())
    positive, negative = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive if positive else None
    return {
        "threshold": float(threshold),
        "hard_siamese_roc_auc": float(roc_auc_score(y[hard], scores[hard]))
        if positive and other.any()
        else None,
        "overall_roc_auc": float(roc_auc_score(y, scores)) if positive and negative else None,
        "accuracy": (tp + tn) / len(y) if len(y) else None,
        "balanced_accuracy": float((recall + tn / negative) / 2)
        if recall is not None and negative
        else None,
        "precision": precision if len(y) else None,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if positive else None,
        "unseen_siamese_fpr": float(pred[other].mean()) if other.any() else None,
        "average_precision": float(average_precision_score(y, scores))
        if positive and negative
        else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "main_count": positive,
        "other_count": int(other.sum()),
        "total_count": len(y),
        "other_accuracy": float((~pred[other]).mean()) if other.any() else None,
        "etc_accuracy": float((~pred[~hard]).mean()) if (~hard).any() else None,
        "majority_baseline_accuracy": max(positive, negative) / len(y) if len(y) else None,
    }


def thresholds_from_validation(
    classes: list[str], scores: np.ndarray, fpr_targets: tuple[float, ...] = ()
) -> dict[str, dict]:
    """내부 검증 점수에서 네 정책의 임계값과 해당 지표를 계산합니다."""
    classes = np.asarray(classes, dtype=str)
    main, other = classes == MAIN, classes == OTHER
    if not main.any() or not other.any() or not np.isfinite(scores).all():
        raise ValueError("임계값 보정에 유한한 양성·다른 샴고양이 검증 점수가 필요합니다.")
    thresholds = np.append(np.unique(scores), np.nextafter(max(1.0, float(scores.max())), np.inf))
    total_positive = len(scores) - np.searchsorted(np.sort(scores), thresholds)
    true_positive = main.sum() - np.searchsorted(np.sort(scores[main]), thresholds)
    other_fp = other.sum() - np.searchsorted(np.sort(scores[other]), thresholds)
    precision = np.divide(
        true_positive, total_positive, out=np.zeros(len(thresholds)), where=total_positive > 0
    )
    recall, fpr = true_positive / main.sum(), other_fp / other.sum()
    f1 = 2 * true_positive / (total_positive + main.sum())
    included = np.arange(len(thresholds))

    def pick(candidates: np.ndarray, *criteria: np.ndarray) -> float:
        """우선순위대로 동률을 해소해 선택한 검증 임계값을 반환합니다."""
        order = np.lexsort(tuple(arr[candidates] for arr in (thresholds, *reversed(criteria))))
        return float(thresholds[candidates[order[-1]]])

    overlap = included[true_positive > 0]
    chosen = {
        # README는 F1 동률일 때 오름차순으로 처음 등장한 임계값을 사용합니다.
        "max_f1": pick(included, f1, -thresholds),
        "precision_recall_balance": pick(overlap, -abs(precision - recall), f1, -fpr),
        "min_siamese_fpr": pick(included, -fpr, recall, precision),
        "target_siamese_fpr": pick(included[fpr <= TARGET_UNSEEN_FPR], recall, precision, -fpr),
    }
    targets = {"target_siamese_fpr": TARGET_UNSEEN_FPR}
    for target in fpr_targets:
        if not 0 <= target <= 1 or not math.isfinite(target):
            raise ValueError("FPR 목표는 0~1의 유한한 비율이어야 합니다.")
        name = f"fpr_le_{target:g}"
        targets[name] = target
        chosen[name] = pick(included[fpr <= target], recall, precision, -fpr)
    results = {name: evaluate(classes, scores, threshold) for name, threshold in chosen.items()}
    for name, target in targets.items():
        results[name].update(
            target_fpr=target,
            target_supports_positive_recall=bool(results[name]["recall"]),
            fpr_resolution=1 / int(other.sum()),
        )
    return results


def checkpoint_payload(
    backbone_key: str,
    backbone_id: str,
    image_size: int,
    method: str,
    classifier: LogisticRegression,
    prototype: np.ndarray,
    threshold: float,
    metadata: dict,
) -> dict:
    """기존 예측 인터페이스와 호환되는 직렬화 가능한 체크포인트를 만듭니다."""
    return {
        **metadata,
        "backbone_key": backbone_key,
        "backbone_id": backbone_id,
        "backbone_frozen": True,
        "image_size": image_size,
        "preprocessed_size": PREPROCESSED_SIZE,
        "method": method,
        "classifier": "LogisticRegression",
        "classifier_settings": {
            "penalty": "l2",
            "class_weight": LOGISTIC_CLASS_WEIGHT,
            "solver": LOGISTIC_SOLVER,
            "max_iter": LOGISTIC_MAX_ITER,
            "random_state": RANDOM_SEED,
        },
        "threshold": threshold,
        "prototype": torch.from_numpy(prototype),
        "linear_coef": torch.from_numpy(classifier.coef_[0]),
        "linear_intercept": float(classifier.intercept_[0]),
    }
