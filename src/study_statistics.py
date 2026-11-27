"""고정 평가 코호트의 그룹 단위 불확실성을 추정합니다."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import RANDOM_SEED
from src.experiment_splits import MAIN, OTHER
from src.linear_probe import evaluate

METRIC_KEYS = ("hard_siamese_roc_auc", "f1", "balanced_accuracy", "unseen_siamese_fpr")


def cluster_intervals(
    records: list[dict],
    groups: dict[str, str],
    rows: list[int],
    scores: np.ndarray,
    threshold: float,
    draws: int = 250,
) -> dict:
    """클래스별 촬영·개체를 복원 추출하여 조건부 95% 구간을 계산합니다."""
    classes = np.asarray([records[i]["class_name"] for i in rows])
    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        strata[classes[index]][groups[records[row]["processed_rel_path"]]].append(index)
    counts = {name: len(grouped) for name, grouped in strata.items()}
    if any(counts.get(cls, 0) < 2 for cls in (MAIN, OTHER)):
        return {"group_counts": counts, "reason": "필수 클래스의 Holdout 그룹이 두 개 미만입니다."}
    rng = np.random.default_rng(RANDOM_SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        indices = []
        for groups_in_class in strata.values():
            blocks = list(groups_in_class.values())
            for i in rng.integers(len(blocks), size=len(blocks)):
                indices.extend(blocks[i])
        result = evaluate(classes[indices], scores[indices], threshold)
        for key in METRIC_KEYS:
            samples[key].append(result[key])
    return {
        "group_counts": counts,
        "draws": draws,
        "scope": "Fixed model and threshold; class-stratified group bootstrap",
        "percentile_95_intervals": {
            key: np.quantile(values, (0.025, 0.975)).tolist() for key, values in samples.items()
        },
    }
