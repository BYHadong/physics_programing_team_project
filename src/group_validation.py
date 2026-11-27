"""촬영·개체 그룹을 보존하면서 평가에 사용할 수 있는 분할을 찾습니다."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.config import RANDOM_SEED
from src.experiment_splits import CLASSES, MAIN, OTHER

REQUIRED_CLASSES = {MAIN, OTHER}


def group_ids(records: list[dict], groups: dict[str, str], indices: list[int]) -> set[str]:
    """선택된 행에 속하는 그룹 ID를 반환합니다."""
    return {groups[records[i]["processed_rel_path"]] for i in indices}


def check_partition(records: list[dict], groups: dict[str, str], *parts: list[int]) -> None:
    """분할 간 원본 인덱스 또는 그룹이 겹치면 실패합니다."""
    seen = set()
    for indices in parts:
        identifiers = group_ids(records, groups, indices)
        if len(set(indices)) != len(indices) or seen.intersection(identifiers):
            raise ValueError("학습·검증·테스트 중 둘 이상의 분할에 동일 그룹이나 사진이 있습니다.")
        seen.update(identifiers)


def buckets_by_class(records: list[dict], groups: dict[str, str], indices: list[int]) -> dict:
    """클래스 안에서 분리할 수 없는 사진 그룹을 수집합니다."""
    buckets = {name: defaultdict(list) for name in CLASSES}
    for index in indices:
        record = records[index]
        buckets[record["class_name"]][groups[record["processed_rel_path"]]].append(index)
    return buckets


def class_counts(records: list[dict], indices: list[int]) -> dict[str, int]:
    """분할의 클래스별 수량을 0을 포함해 반환합니다."""
    counts = Counter(records[index]["class_name"] for index in indices)
    return {name: counts[name] for name in CLASSES}


def complete_fold(records: list[dict], indices: list[int]) -> bool:
    """양성과 다른 샴고양이가 모두 있는지를 확인합니다."""
    return REQUIRED_CLASSES.issubset({records[index]["class_name"] for index in indices})


def distribution_score(records: list[dict], partitions: list[list[int]]) -> float:
    """클래스별 전체 수량 중 각 분할이 차지하는 비율의 편차를 반환합니다."""
    counts = np.asarray([list(class_counts(records, part).values()) for part in partitions])
    total = counts.sum(axis=0)
    active = total > 0
    return float((counts[:, active] / total[active]).std(axis=0).mean())


def balanced_folds(buckets: dict, indices: list[int], count: int) -> list[dict]:
    """혼합 클래스 그룹이 없는 경우 그룹을 쪼개지 않고 최소 지지수를 확보합니다."""
    folds: list[list[int]] = [[] for _ in range(count)]
    for groups in buckets.values():
        loads = [0] * count
        for group in sorted(groups.values(), key=lambda subset: (-len(subset), subset[0])):
            target = min(range(count), key=lambda i: (loads[i], len(folds[i]), i))
            folds[target].extend(group)
            loads[target] += len(group)
    return [
        {"train": sorted(set(indices).difference(part)), "validation": sorted(part)}
        for part in folds
    ]


def choose_folds(
    records: list[dict],
    groups: dict[str, str],
    indices: list[int],
    maximum: int = 5,
) -> tuple[list[dict], list[dict], str | None]:
    """가능한 Fold 수부터 두 sklearn 방식을 비교하고 필요한 경우 그룹 균형 분할을 씁니다."""
    buckets = buckets_by_class(records, groups, indices)
    cap = min(maximum, *(len(buckets[name]) for name in REQUIRED_CLASSES))
    y = [records[i]["class_name"] for i in indices]
    g = [groups[records[i]["processed_rel_path"]] for i in indices]
    x = np.zeros(len(indices))
    comparisons = []
    # 5-Fold 요청에서 클래스 지지수가 부족한 단계도 보고서에 남깁니다.
    for count in range(maximum, max(1, cap), -1):
        for method in ("GroupKFold", "StratifiedGroupKFold"):
            comparisons.append(
                {
                    "method": method,
                    "fold_count": count,
                    "valid": False,
                    "reason": "필수 클래스의 독립 그룹 수가 Fold 수보다 적습니다.",
                }
            )

    for count in range(cap, 1, -1):
        partitions = []
        for name, options in (
            ("GroupKFold", [GroupKFold(n_splits=count)]),
            (
                "StratifiedGroupKFold",
                [
                    StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=RANDOM_SEED + i)
                    for i in range(12)
                ],
            ),
        ):
            trials = []
            for splitter in options:
                folds = [
                    {"train": [indices[i] for i in train], "validation": [indices[i] for i in val]}
                    for train, val in splitter.split(x, y, g)
                ]
                valid = all(
                    complete_fold(records, fold["train"])
                    and complete_fold(records, fold["validation"])
                    for fold in folds
                )
                score = distribution_score(records, [fold["validation"] for fold in folds])
                trials.append((not valid, score, folds))
            invalid, score, folds = min(trials, key=lambda trial: trial[:2])
            comparisons.append(
                {
                    "method": name,
                    "fold_count": count,
                    "valid": not invalid,
                    "imbalance": score,
                    "candidate_count": len(trials),
                    "validation_counts": [class_counts(records, f["validation"]) for f in folds],
                }
            )
            if not invalid:
                partitions.append((score, name, folds))

        if not partitions:
            # 현재 감사 정책은 그룹을 한 클래스 안에서만 합칩니다.
            folds = balanced_folds(buckets, indices, count)
            valid = all(
                complete_fold(records, f["validation"]) and complete_fold(records, f["train"])
                for f in folds
            )
            comparisons.append(
                {
                    "method": "BalancedGroupAssignment",
                    "fold_count": count,
                    "valid": valid,
                    "validation_counts": [class_counts(records, f["validation"]) for f in folds],
                }
            )
            if valid:
                score = distribution_score(records, [f["validation"] for f in folds])
                partitions.append((score, "BalancedGroupAssignment", folds))
        if partitions:
            _, method, selected = min(partitions, key=lambda row: row[:2])
            for fold in selected:
                check_partition(records, groups, fold["train"], fold["validation"])
            return selected, comparisons, method
    return [], comparisons, None


def make_plan(
    records: list[dict],
    groups: dict[str, str],
    excluded: list[int],
    maximum: int = 5,
    prior: list[int] | None = None,
    prior_related_groups: set[str] | None = None,
) -> dict:
    """고정 Holdout을 먼저 확보하고 개발 데이터에서 유효한 그룹 CV를 선택합니다."""
    eligible = sorted(set(range(len(records))).difference(excluded))
    buckets = buckets_by_class(records, groups, eligible)
    availability = {name: len(value) for name, value in buckets.items()}
    held = []
    excluded = sorted(set(excluded))
    prior_status = "이전 Holdout 목록 없이 현재 확인된 그룹에서 분할했습니다."
    if prior:
        prior_set = set(prior)
        if not prior_set <= set(range(len(records))):
            raise ValueError("이전 Holdout 인덱스가 현재 Manifest 범위를 벗어납니다.")
        protected = group_ids(records, groups, prior).union(prior_related_groups or ())
        held = [i for i in eligible if groups[records[i]["processed_rel_path"]] in protected]
        if set(held) == prior_set:
            prior_status = "이전에 사용한 Holdout을 현재 그룹 규칙에서 재사용합니다."
        else:
            prior_status = "이전 Holdout에 연결된 그룹 전체를 개발 데이터에서 격리했습니다."
        if not complete_fold(records, held):
            excluded = sorted(set(excluded).union(held))
            eligible = sorted(set(eligible).difference(held))
            held = []
            prior_status += " 필수 클래스가 부족하여 이 그룹에서 공동 평가를 계산할 수 없습니다."
    if not prior and not held and min(availability[cls] for cls in REQUIRED_CLASSES) >= 3:
        # 양성·다른 샴고양이의 개발 그룹을 최소 두 개씩 남깁니다.
        for name, grouped in buckets.items():
            keep = max(2, min(maximum, len(grouped) - 1)) if name in REQUIRED_CLASSES else 1
            budget = len(grouped) - keep
            remaining = list(grouped.values())
            selected: list[int] = []
            goal = sum(map(len, remaining)) * 0.2
            for _ in range(max(0, budget)):
                subset = min(
                    remaining, key=lambda part: (abs(len(selected) + len(part) - goal), part[0])
                )
                if selected and abs(len(selected) - goal) <= abs(
                    len(selected) + len(subset) - goal
                ):
                    break
                selected.extend(subset)
                remaining.remove(subset)
            held.extend(selected)
    development = sorted(set(eligible).difference(held))
    held = sorted(held)
    folds, comparisons, method = choose_folds(records, groups, development, maximum)
    for fold in folds:
        check_partition(records, groups, fold["train"], fold["validation"], held, excluded)
    check_partition(records, groups, development, held, excluded)
    return {
        "development": development,
        "holdout": held,
        "excluded": excluded,
        "groups": groups,
        "eligible_group_counts": availability,
        "development_group_counts": {
            name: len(items)
            for name, items in buckets_by_class(records, groups, development).items()
        },
        "folds": folds,
        "selected_method": method,
        "splitter_comparison": comparisons,
        "fold_counts": [
            {part: class_counts(records, ix) for part, ix in fold.items()} for fold in folds
        ],
        "holdout_counts": class_counts(records, held),
        "prior_holdout_status": prior_status,
        "prior_holdout_overlap": len(set(prior).intersection(held)) if prior is not None else None,
        "prior_holdout_eligible_indices": sorted(set(prior or []).intersection(held)),
        "independent_external_dataset": False,
        "reason": None
        if folds
        else "양성·다른 샴고양이 각각 두 개 이상의 검증 가능한 그룹이 필요합니다.",
        "holdout_reason": None
        if held
        else "개발 CV에 필요한 그룹 수를 남기면 고정 Holdout을 구성할 수 없습니다.",
    }
