"""촬영·개체를 격리해 DINOv2 Linear Probe를 비교하고 새 결과를 저장합니다."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.config import (
    BATCH_SIZE,
    COLAB_BACKUP_DIR,
    DINO_BACKBONE_ID,
    DINO_BACKBONE_KEY,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from src.embedding_cache import EmbeddingCache, fingerprint
from src.experiment_splits import MAIN, OTHER, load_manifest, source_index
from src.group_validation import buckets_by_class, choose_folds, make_plan
from src.linear_probe import (
    Hyperparameters,
    checkpoint_payload,
    evaluate,
    fit_probe,
    probe_scores,
    thresholds_from_validation,
)
from src.study_audit import scan_dataset, similarity_reviews
from src.study_data import INTEGRITY
from src.study_history import history_audit, legacy_rows, restore_legacy
from src.prepare_dataset import record_keys
from src.train_baseline import load_backbone, set_seed
from src.study_statistics import cluster_intervals
from src.tune_hyperparameters import (
    METHOD,
    class_names,
    read_json,
    summarize_folds,
    write_artifact,
    write_csv,
    write_json,
)

POLICIES = ("max_f1", "precision_recall_balance", "target_siamese_fpr")
FPR_TARGETS = (0.01, 0.05, 0.10)
RESOLUTIONS = (224, 336, 448)
SEARCH = (
    ("c", (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)),
    ("main_target_count", (None, 250, 500, 750, 1000)),
    ("neg_aug_passes", (0, 1, 3, 5, 7)),
    ("rotation_degrees", (0, 5, 10, 15, 20, 30)),
    ("hflip_probability", (0.0, 0.25, 0.5, 0.75, 1.0)),
)
RANK_WEIGHTS = {
    "hard_siamese_roc_auc": 0.50,
    "f1": 0.18,
    "unseen_siamese_fpr": 0.15,
    "balanced_accuracy": 0.07,
    "recall": 0.05,
    "fold_hard_auc_std": 0.05,
}


def study_metrics(classes: list[str], scores: np.ndarray, threshold: float) -> dict:
    """공통 지표와 두 샴고양이 클래스만 포함한 정확도를 계산합니다."""
    result = evaluate(classes, scores, threshold)
    labels = np.asarray(classes)
    hard = np.isin(labels, [MAIN, OTHER])
    result["hard_siamese_accuracy"] = (
        float(((scores >= threshold)[hard] == (labels == MAIN)[hard]).mean())
        if hard.any()
        else None
    )
    return result


def policies_from_oof(
    classes: list[str], scores: np.ndarray, fpr_targets: tuple[float, ...] = FPR_TARGETS
) -> dict:
    """세 정책을 개발 데이터 OOF 점수로만 보정하고 동일 지표를 계산합니다."""
    existing = thresholds_from_validation(classes, scores, fpr_targets)
    policies = {
        policy: {**row, **study_metrics(classes, scores, row["threshold"])}
        for policy, row in existing.items()
        if policy != "min_siamese_fpr"
    }
    policies["fixed_0_5"] = study_metrics(classes, scores, 0.5)
    return policies


def evaluate_cv(
    cache: EmbeddingCache,
    plan: dict,
    params: Hyperparameters,
    policy: str,
    fpr_targets: tuple[float, ...] = FPR_TARGETS,
) -> dict:
    """Fold별 학습·검증의 원본 점수를 저장하고 OOF로 임계값을 결정합니다."""
    development = plan["development"]
    offsets = {index: i for i, index in enumerate(development)}
    oof = np.full(len(development), np.nan)
    raw_scores = []
    timings = {}
    with threadpool_limits(limits=1):
        for number, fold in enumerate(plan["folds"], 1):
            train, val = fold["train"], fold["validation"]
            print(
                f"[Fold {number}/{len(plan['folds'])}] 학습 {len(train)} · 검증 {len(val)}",
                flush=True,
            )
            classifier, prototype, train_counts = fit_probe(cache, train, params, timings)
            train_scores = probe_scores(cache.get(train), classifier, prototype, METHOD)
            val_scores = probe_scores(cache.get(val), classifier, prototype, METHOD)
            oof[[offsets[i] for i in val]] = val_scores
            raw_scores.append((train_scores, val_scores, train_counts))
    classes = class_names(cache.records, development)
    policies = policies_from_oof(classes, oof, fpr_targets)
    threshold = policies[policy]["threshold"]
    folds = []
    for number, (fold, scores) in enumerate(zip(plan["folds"], raw_scores), 1):
        train_scores, val_scores, counts = scores
        train_metrics = study_metrics(
            class_names(cache.records, fold["train"]), train_scores, threshold
        )
        val_metrics = study_metrics(
            class_names(cache.records, fold["validation"]), val_scores, threshold
        )
        folds.append(
            {
                "fold": number,
                **counts,
                "positive_multiplier": counts["positive_training"] / counts["positive_original"],
                "negative_multiplier": counts["negative_training"] / counts["negative_original"],
                "train_clean": train_metrics,
                "validation": val_metrics,
            }
        )
    return {
        "metrics": policies[policy],
        **timings,
        "policies": policies,
        "fold_results": folds,
        "fold_summary": {
            split: summarize_folds([{"metrics": fold[split]} for fold in folds])
            for split in ("train_clean", "validation")
        },
        "oof_predictions": {"indices": development, "classes": classes, "scores": oof.tolist()},
    }


def score_trial(result: dict) -> float:
    """사전에 지정한 가중치로 개발 OOF 모델 순위를 계산합니다."""
    values = result["metrics"]
    std = result["fold_summary"]["validation"]["hard_siamese_roc_auc"]["std"]
    scores = {
        **values,
        "unseen_siamese_fpr": 1 - values["unseen_siamese_fpr"],
        "fold_hard_auc_std": 1 - std,
    }
    return sum(weight * scores[metric] for metric, weight in RANK_WEIGHTS.items())


def resolution_batches(
    model: torch.nn.Module, device: torch.device, resolutions: tuple[int, ...] = RESOLUTIONS
) -> dict[int, int]:
    """동결 백본의 실제 forward로 해상도 및 사용할 배치 크기를 확인합니다."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("해상도 비교는 동결된 eval 백본이 필요합니다.")
    batches = {}
    preferences = {224: BATCH_SIZE, 336: min(BATCH_SIZE, 8), 448: min(BATCH_SIZE, 4)}
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE는 1 이상이어야 합니다.")
    for size in resolutions:
        preferred = preferences[size]
        batch = preferred if device.type == "cuda" else 1
        while batch:
            try:
                with (
                    torch.inference_mode(),
                    torch.amp.autocast("cuda", enabled=device.type == "cuda"),
                ):
                    features = model(torch.zeros((batch, 3, size, size), device=device))
                    if (
                        features.ndim != 2
                        or features.shape[0] != batch
                        or not torch.isfinite(features).all()
                    ):
                        raise ValueError(f"{size}px에서 임베딩 형식이 예상과 다릅니다.")
                    del features
                break
            except torch.cuda.OutOfMemoryError:
                batch //= 2
            if batch:
                torch.cuda.empty_cache()
        if not batch:
            raise RuntimeError(f"{size}px에서 한 장도 처리할 CUDA 메모리가 없습니다.")
        batches[size] = batch
        print(f"[입력 확인] {size}×{size} · 배치 {batch}", flush=True)
    # CPU도 기존 224px 배치 설정을 사용해야 호환 캐시를 조회할 수 있습니다.
    return batches if device.type == "cuda" else {size: preferences[size] for size in resolutions}


def previous_holdout(
    path: Path | None, manifest_hash: str, records: list[dict], processed: Path
) -> list[int] | None:
    """동일 Manifest의 예전 Holdout 위치만 비교용으로 읽습니다."""
    if path is None:
        return None
    config_path = path.with_name("experiment_config.json")
    if not config_path.is_file():
        raise ValueError("이전 Holdout의 experiment_config.json이 필요합니다.")
    prior_hash = read_json(config_path).get("manifest_sha256")
    held = read_json(path)["holdout"]
    if manifest_hash == prior_hash:
        return held
    provenance = processed / INTEGRITY
    if provenance.is_file():
        origin = read_json(provenance).get("archived_manifest", {})
        if prior_hash == origin.get("sha256") and prior_hash is not None:
            current = {key: i for i, key in enumerate(record_keys(records))}
            try:
                keys = origin["original_keys"]
                if not set(held) <= set(range(len(keys))):
                    raise ValueError("이전 Holdout 인덱스가 이전 Manifest의 범위를 벗어납니다.")
                return [current[keys[i]] for i in held]
            except (KeyError, IndexError) as error:
                raise ValueError("복구본에서 과거 Holdout 원본을 찾지 못했습니다.") from error
    raise ValueError("이전 Holdout Manifest와 현재 데이터의 대응을 검증할 수 없습니다.")


class Study:
    """동일 실험 안에서 후보를 저장하고 이미 완료된 학습을 재개합니다."""

    def __init__(
        self,
        directory: Path,
        signature: str,
        plan: dict,
        policy: str,
        caches: dict[int, EmbeddingCache],
        device: torch.device,
        fpr_targets: tuple[float, ...] = FPR_TARGETS,
    ) -> None:
        """분할, 캐시 및 후보 결과의 이름 공간을 고정합니다."""
        self.directory, self.signature, self.plan = directory, signature, plan
        self.policy, self.caches, self.device = policy, caches, device
        self.rows: list[dict] = []
        self.started = perf_counter()
        self.planned = 0
        self.new_fits = 0
        self.resumed = 0
        self.fpr_targets = fpr_targets

    def trial(self, name: str, params: Hyperparameters, size: int, stage: str) -> dict:
        """동일 설정의 결과를 재사용하고 새 학습의 경과 시간과 CUDA 피크를 기록합니다."""
        context = {
            "signature": self.signature,
            "experiment": name,
            "stage": stage,
            "parameters": asdict(params),
            "image_size": size,
        }
        path = self.directory / name / "result.json"
        if path.is_file():
            row = read_json(path)
            if any(row.get(key) != value for key, value in context.items()):
                raise ValueError(f"기존 후보 설정과 다릅니다. 새 RUN_ID가 필요합니다. {path}")
            self.resumed += 1
            print(f"[{name}] 완료 결과 사용", flush=True)
        else:
            previous = next(
                (
                    row
                    for row in self.rows
                    if row["image_size"] == size and row["parameters"] == context["parameters"]
                ),
                None,
            )
            start = perf_counter()
            print(f"[{name}] {stage} · {size}px · {context['parameters']}", flush=True)
            if previous:
                row = {
                    **previous,
                    "reused_trial": previous["experiment"],
                    "peak_gpu_mib": None,
                    "features_reused": 0,
                    "features_extracted": 0,
                    "embedding_seconds": 0.0,
                    "classifier_fit_seconds": 0.0,
                    "oom_retries": 0,
                }
            else:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                cache = self.caches[size]
                fields = {
                    "features_reused": "hits",
                    "features_extracted": "extracted",
                    "embedding_seconds": "extraction_seconds",
                    "oom_retries": "oom_retries",
                }
                before = {key: getattr(cache, attr) for key, attr in fields.items()}
                row = evaluate_cv(cache, self.plan, params, self.policy, self.fpr_targets)
                row.update(
                    {key: getattr(cache, attr) - before[key] for key, attr in fields.items()}
                )
                self.new_fits += 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                row["peak_gpu_mib"] = (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else None
                )
                row["reused_trial"] = None
            row.update(
                context,
                batch_size=self.caches[size].batch_size,
                elapsed_seconds=perf_counter() - start,
            )
            row["selection_score"] = score_trial(row)
            write_json(path, row)
        self.rows.append(row)
        lead = max(self.rows, key=score_trial)["experiment"]
        remaining = self.planned - len(self.rows)
        accumulated = perf_counter() - self.started
        started_rows = [r for r in self.rows if not r["reused_trial"] and r["features_extracted"]]
        estimate = (
            np.mean([r["elapsed_seconds"] for r in started_rows]) * remaining
            if started_rows and remaining
            else 0
        )
        print(
            f"[진행 {len(self.rows)}/{self.planned}] 최대 {remaining}개 남음 · 상위 {lead} · "
            f"캐시 {row['features_reused']}건 · 신규 {row['features_extracted']}건 · "
            f"실험 {row['elapsed_seconds']:.1f}초 · 누적 {accumulated:.1f}초 · "
            f"남은 시간 추정 {estimate:.0f}초 · GPU 피크 {row['peak_gpu_mib']} MiB · {path}",
            flush=True,
        )
        return row

    def search(
        self, legacy: dict[str, Hyperparameters], budget: int, resolutions: tuple[int, ...]
    ) -> tuple[dict, list[dict]]:
        """기존 24개 ID를 먼저 재평가하고 예산 안에서 순차·상호작용 탐색을 실행합니다."""
        measures = [(field, value) for field, values in SEARCH for value in values]
        self.planned = len(legacy) + min(budget, len(measures) + 3) + len(resolutions) - 1
        for name, params in legacy.items():
            self.trial(name, params, 224, "legacy")
        best = max(self.rows, key=score_trial)
        spent, alternatives = 0, {}
        for field, values in SEARCH:
            anchor = Hyperparameters(**best["parameters"])
            candidates = []
            for value in values[: max(0, budget - spent)]:
                settings = {field: value}
                if field == "main_target_count":
                    settings["positive_aug_passes"] = 0 if value is None else 7
                params = replace(anchor, **settings)
                spent += 1
                candidates.append(self.trial(f"T{spent:02d}", params, 224, field))
            best = max([best, *candidates], key=score_trial)
            alternatives[field] = sorted(candidates, key=score_trial, reverse=True)
        anchor = Hyperparameters(**best["parameters"])
        for first, second in (
            ("c", "neg_aug_passes"),
            ("main_target_count", "rotation_degrees"),
            ("hflip_probability", "rotation_degrees"),
        ):
            if spent >= budget:
                break
            changes = {}
            for field in (first, second):
                other = next(
                    (
                        row
                        for row in alternatives[field]
                        if row["parameters"][field] != getattr(anchor, field)
                    ),
                    None,
                )
                if other:
                    changes[field] = other["parameters"][field]
            if len(changes) != 2:
                continue
            if "main_target_count" in changes:
                changes["positive_aug_passes"] = 0 if changes["main_target_count"] is None else 7
            params = replace(anchor, **changes)
            if any(row["parameters"] == asdict(params) for row in self.rows):
                continue
            spent += 1
            best = max(
                [best, self.trial(f"T{spent:02d}", params, 224, "interaction")], key=score_trial
            )
        params = Hyperparameters(**best["parameters"])
        self.planned = len(self.rows) + len(resolutions) - 1
        print(f"[해상도] {best['experiment']} 조건 고정 · 전체 확정 {self.planned}행", flush=True)
        resolution_rows = [best] + [
            self.trial(f"R{size}", params, size, "resolution")
            for size in resolutions
            if size != 224
        ]
        return max(resolution_rows, key=score_trial), resolution_rows


def compare_final(
    study: Study,
    selected: dict,
    results: Path,
    checkpoint: Path,
    manifest_hash: str,
) -> None:
    """선정이 끝난 후 E00·E15·선정 모델을 같은 개발 및 고정 평가 분할에서 비교합니다."""
    report_path = results / "final_model_metrics.json"
    if (report_path.is_file() and read_json(report_path)["signature"] != study.signature) or (
        checkpoint.is_file()
        and torch.load(checkpoint, map_location="cpu", weights_only=True)["run_signature"]
        != study.signature
    ):
        raise ValueError("기존 최종 결과가 현재 설정과 다릅니다. 새 RUN_ID를 사용하세요.")
    if report_path.is_file() and checkpoint.is_file():
        return
    development, holdout = study.plan["development"], study.plan["holdout"]
    prior_indices = set(study.plan["prior_holdout_eligible_indices"])
    prior_positions = [position for position, row in enumerate(holdout) if row in prior_indices]
    report = {}
    trained = {}
    by_id = {row["experiment"]: row for row in study.rows}
    for label, row in (("baseline", by_id["E00"]), ("e15", by_id["E15"]), ("final", selected)):
        cache = study.caches[row["image_size"]]
        threshold = row["metrics"]["threshold"]
        name = row["experiment"]
        if name not in trained:
            started = perf_counter()
            params = Hyperparameters(**row["parameters"])
            with threadpool_limits(limits=1):
                clf, proto, counts = fit_probe(cache, development, params)
                held_classes = class_names(cache.records, holdout)
                scores = probe_scores(cache.get(holdout), clf, proto, METHOD) if holdout else None
            trained[name] = (clf, proto, counts, held_classes, scores, perf_counter() - started)
        clf, proto, counts, classes, scores, elapsed = trained[name]
        report[label] = {
            "experiment": name,
            "image_size": row["image_size"],
            "training_counts": counts,
            "oof": row["metrics"],
            "oof_policies": row["policies"],
            "holdout": study_metrics(classes, scores, threshold) if scores is not None else None,
            "holdout_predictions": {"classes": classes, "scores": scores.tolist()}
            if scores is not None
            else None,
            "previous_holdout_subset": study_metrics(
                [classes[i] for i in prior_positions], scores[prior_positions], threshold
            )
            if scores is not None and prior_positions
            else None,
            "refit_seconds": elapsed,
        }
        if label == "final" and not checkpoint.is_file():
            saved = checkpoint_payload(
                DINO_BACKBONE_KEY,
                DINO_BACKBONE_ID,
                row["image_size"],
                METHOD,
                clf,
                proto,
                threshold,
                {
                    "experiment": name,
                    "hyperparameters": row["parameters"],
                    "threshold_policy": study.policy,
                    "fpr_targets": list(study.fpr_targets),
                    "train_seed": RANDOM_SEED,
                    "manifest_sha256": manifest_hash,
                    "training_indices": development,
                    "holdout_indices": holdout,
                    "dataset_digest": cache.identity["dataset_and_manifest"],
                    "embedding_cache_key": cache.key,
                    "backbone_weights_sha256": cache.identity["weights_sha256"],
                    "study_groups": study.plan["groups"],
                    "study_excluded": study.plan["excluded"],
                    "study_quarantined": study.plan.get("quarantined_other_copies", []),
                    "study_photo_hashes": list(cache.photo_hashes or []),
                    "batch_size": cache.batch_size,
                    "run_signature": study.signature,
                },
            )
            buffer = BytesIO()
            torch.save(saved, buffer)
            write_artifact(checkpoint, buffer.getvalue())
    if not report_path.is_file():
        write_csv(
            results / "baseline_comparison.csv",
            [
                {"model": label, "experiment": row["experiment"], "split": split, **row[split]}
                for label, row in report.items()
                for split in ("oof", "holdout", "previous_holdout_subset")
                if row[split] is not None
            ],
        )
        write_json(
            report_path,
            {
                "signature": study.signature,
                "selected_experiment": selected["experiment"],
                "threshold_policy": study.policy,
                "holdout_index_count": len(holdout),
                "independent_external_dataset": False,
                "prior_holdout_status": study.plan["prior_holdout_status"],
                "prior_holdout_overlap": study.plan["prior_holdout_overlap"],
                "comparisons": report,
                "holdout_uncertainty": cluster_intervals(
                    study.caches[selected["image_size"]].records,
                    study.plan["groups"],
                    holdout,
                    trained[selected["experiment"]][4],
                    selected["metrics"]["threshold"],
                )
                if holdout
                else None,
            },
        )


def compare_positive_data(study: Study, path: Path) -> None:
    """E15 Fold마다 확인된 양성 촬영 그룹 하나를 제외해 데이터량 영향을 비교합니다."""
    if path.is_file():
        return
    reduced, removed = [], []
    cache = study.caches[224]
    for fold in study.plan["folds"]:
        groups = buckets_by_class(cache.records, study.plan["groups"], fold["train"])[MAIN]
        if len(groups) < 2:
            write_json(path, {"reason": "일부 학습 Fold에 확인된 양성 그룹이 하나뿐입니다."})
            return
        group = min(groups.values(), key=lambda rows: (len(rows), rows[0]))
        reduced.append({**fold, "train": sorted(set(fold["train"]).difference(group))})
        removed.append(group)
    start = perf_counter()
    baseline = next(row for row in study.rows if row["experiment"] == "E15")
    result = evaluate_cv(
        cache,
        {**study.plan, "folds": reduced},
        Hyperparameters(**baseline["parameters"]),
        study.policy,
        study.fpr_targets,
    )
    write_json(
        path,
        {
            "scope": "development OOF, E15 only, same validation indices",
            "removed_indices_by_fold": removed,
            "full_oof": baseline["metrics"],
            "reduced_oof": result["metrics"],
            "full_folds": baseline["fold_results"],
            "reduced_folds": result["fold_results"],
            "elapsed_seconds": perf_counter() - start,
        },
    )


def nested_diagnostic(study: Study, legacy: dict[str, Hyperparameters], path: Path) -> None:
    """고정된 E00·E15 두 후보에 한해 바깥 Fold와 분리된 모델 선택을 검증합니다."""
    if path.is_file():
        return
    cache = study.caches[224]
    groups = study.plan["groups"]
    inner_plans = []
    for fold in study.plan["folds"]:
        inner, _, _ = choose_folds(cache.records, groups, fold["train"], 3)
        if not inner:
            write_json(
                path,
                {
                    "reason": "외부 학습 Fold의 독립 그룹이 부족하여 내부 Group CV를 만들 수 없습니다."
                },
            )
            return
        inner_plans.append(inner)
    report = []
    for fold, inner in zip(study.plan["folds"], inner_plans):
        train, validation = fold["train"], fold["validation"]
        inner_plan = {"development": train, "folds": inner}
        trials = {
            name: evaluate_cv(cache, inner_plan, legacy[name], study.policy, study.fpr_targets)
            for name in ("E00", "E15")
        }
        name = max(trials, key=lambda candidate: score_trial(trials[candidate]))
        threshold = trials[name]["metrics"]["threshold"]
        with threadpool_limits(limits=1):
            classifier, prototype, _ = fit_probe(cache, train, legacy[name])
            scores = probe_scores(cache.get(validation), classifier, prototype, METHOD)
        report.append(
            {
                "selected": name,
                "outer_validation": validation,
                "inner_folds": inner,
                "metrics": study_metrics(class_names(cache.records, validation), scores, threshold),
            }
        )
    write_json(
        path,
        {
            "scope": "Nested Group CV within development, predeclared E00 and E15 only",
            "used_for_final_selection": False,
            "folds": report,
            "summary": summarize_folds(report),
        },
    )


def export_tables(results: Path, rows: list[dict], resolutions: list[dict], best: dict) -> None:
    """모든 후보와 Fold의 원자료를 CSV와 JSON으로 저장합니다."""
    write_json(results / "hyperparameter_results.json", rows)
    table = [
        {
            "experiment": row["experiment"],
            "stage": row["stage"],
            "image_size": row["image_size"],
            "batch_size": row["batch_size"],
            **row["parameters"],
            **row["metrics"],
            "selection_score": row["selection_score"],
            "fold_hard_auc_mean": row["fold_summary"]["validation"]["hard_siamese_roc_auc"]["mean"],
            "fold_hard_auc_std": row["fold_summary"]["validation"]["hard_siamese_roc_auc"]["std"],
            "elapsed_seconds": row["elapsed_seconds"],
            "peak_gpu_mib": row["peak_gpu_mib"],
            "reused_trial": row["reused_trial"],
            "embedding_seconds": row["embedding_seconds"],
            "classifier_fit_seconds": row["classifier_fit_seconds"],
            "features_extracted": row["features_extracted"],
            "features_reused": row["features_reused"],
            "oom_retries": row["oom_retries"],
        }
        for row in rows
    ]
    write_csv(results / "hyperparameter_results.csv", table)
    write_csv(results / "legacy_reevaluation.csv", [r for r in table if r["stage"] == "legacy"])
    write_csv(results / "additional_search.csv", [r for r in table if r["stage"] != "legacy"])
    write_csv(
        results / "fold_results.csv",
        [
            {
                "experiment": row["experiment"],
                "fold": fold["fold"],
                "split": split,
                "positive_training": fold["positive_training"],
                "negative_training": fold["negative_training"],
                "positive_multiplier": fold["positive_multiplier"],
                "negative_multiplier": fold["negative_multiplier"],
                **fold[split],
            }
            for row in rows
            for fold in row["fold_results"]
            for split in ("train_clean", "validation")
        ],
    )
    write_csv(
        results / "threshold_comparison.csv",
        [
            {"experiment": row["experiment"], "policy": policy, **metrics}
            for row in rows
            for policy, metrics in row["policies"].items()
        ],
    )
    write_csv(
        results / "resolution_comparison.csv",
        [
            {
                "experiment": row["experiment"],
                "image_size": row["image_size"],
                "batch_size": row["batch_size"],
                "elapsed_seconds": row["elapsed_seconds"],
                "peak_gpu_mib": row["peak_gpu_mib"],
                "reused_trial": row["reused_trial"],
                "embedding_seconds": row["embedding_seconds"],
                "classifier_fit_seconds": row["classifier_fit_seconds"],
                "features_extracted": row["features_extracted"],
                "features_reused": row["features_reused"],
                "oom_retries": row["oom_retries"],
                **row["metrics"],
            }
            for row in resolutions
        ],
    )
    write_json(
        results / "best_hyperparameters.json",
        {
            "experiment": best["experiment"],
            "image_size": best["image_size"],
            "resolution_experiments": [row["experiment"] for row in resolutions],
            "parameters": best["parameters"],
            "validation_metrics": best["metrics"],
            "validation_threshold_policies": best["policies"],
            "rank_weights": RANK_WEIGHTS,
        },
    )


def run(args: argparse.Namespace) -> None:
    """감사, 고정 분할, 후보 비교 및 최종 평가를 새 RUN_ID에 저장합니다."""
    start = perf_counter()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("RUN_ID에는 영문·숫자·밑줄·하이픈만 사용할 수 있습니다.")
    set_seed(RANDOM_SEED)
    results = args.output / "results" / args.run_id
    manifest = args.processed / "manifest.json"
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    records = load_manifest(manifest, require_other_ids=False)
    legacy_dir = args.legacy_results or args.output / "results" / "dinov2-1121-01"
    targets = tuple(args.fpr_targets)
    policy_names = set(POLICIES).union(f"fpr_le_{target:g}" for target in targets)
    if args.threshold_policy not in policy_names:
        raise ValueError(f"임계값 정책은 {sorted(policy_names)} 중에서 지정하세요.")
    if args.additional_budget < 0 or any(not 0 <= target <= 1 for target in targets):
        raise ValueError("추가 예산과 FPR 목표를 확인하세요.")
    resolutions = (224, *sorted(set(args.input_sizes).difference({224})))
    legacy, origins = {}, {}
    try:
        legacy, origins = restore_legacy(legacy_dir)
    except ValueError as error:
        if not args.audit_only:
            raise
        print(f"[기존 설정] {error}", flush=True)
    quarantine_other_copies = getattr(args, "quarantine_other_copies", False)
    digest, groups, excluded, hashes, audit = scan_dataset(
        records,
        args.processed,
        source_index(args.source),
        manifest_hash,
        args.sessions,
        args.review,
        quarantine_other_copies=quarantine_other_copies,
    )
    integrity = args.processed / INTEGRITY
    if integrity.is_file():
        entries = read_json(integrity)["sources"]
        for key, row in zip(record_keys(records), audit["file_hashes"]):
            if entries.get(key) != {
                "raw_sha256": row["original_sha256"],
                "crop_sha256": row["crop_sha256"],
            }:
                raise ValueError(f"준비 후 원본 또는 전처리 크롭이 변경되었습니다. {row['index']}")
    old_split = legacy_dir / "fold_indices.json"
    prior = args.prior_split or (old_split if old_split.is_file() else None)
    prior_indices = previous_holdout(prior, manifest_hash, records, args.processed)
    prior_set = set(prior_indices or ())
    quarantine = audit["other_copy_quarantine"]
    prior_related_groups = {
        group for i, group in quarantine["retained_split_groups_by_index"].items() if i in prior_set
    }
    plan = make_plan(
        records,
        groups,
        excluded,
        prior=prior_indices,
        prior_related_groups=prior_related_groups,
    )
    plan["quarantined_other_copies"] = quarantine["indices"]
    config = {
        "dataset_sha256": digest,
        "manifest_sha256": manifest_hash,
        "groups_sha256": fingerprint(plan),
        "raw_audit_sha256": fingerprint(audit),
        "seed": RANDOM_SEED,
        "policy": args.threshold_policy,
        "rank_weights": RANK_WEIGHTS,
        "fpr_targets": targets,
        "legacy_sources": origins,
        "legacy_conditions": legacy_rows(legacy),
        "quarantine_other_copies": quarantine_other_copies,
        "additional_budget": args.additional_budget,
        "search": SEARCH,
        "resolutions": resolutions,
        "code": {
            source.name: sha256(source.read_bytes()).hexdigest()
            for source in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    write_json(results / "experiment_config.json", config)
    write_json(results / "data_audit.json", audit)
    write_json(results / "fold_indices.json", plan)
    write_json(results / "splitter_comparison.json", plan["splitter_comparison"])
    if legacy:
        write_csv(results / "legacy_conditions.csv", legacy_rows(legacy))
        if not (results / "historical_metrics_audit.json").exists():
            write_json(results / "historical_metrics_audit.json", history_audit(legacy_dir))
    print(
        f"[그룹] 확인된 양성 {audit['verified_main_group_count']}개 · 평가 제외 {len(excluded)}장"
    )
    print(
        f"[분할] {plan['selected_method']} · {len(plan['folds'])} Fold · Holdout {len(plan['holdout'])}장"
    )
    for number, counts in enumerate(plan["fold_counts"], 1):
        print(f"[Fold {number}] {counts}", flush=True)
    if audit["exact_or_review_conflicts"]:
        raise ValueError(
            "동일 사진의 라벨 또는 Manifest 경로 충돌이 있습니다. data_audit.json을 확인하세요."
        )
    if quarantine["indices"]:
        print(
            f"[복제 격리] {len(quarantine['indices'])}개 행 제외 · "
            f"충돌 {len(quarantine['resolved_conflicts'])}건 처리",
            flush=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_backbone(DINO_BACKBONE_ID, device)
    batches = resolution_batches(model, device, resolutions)
    photo_hashes = [(row["original_sha256"], row["crop_sha256"]) for row in audit["file_hashes"]]
    caches = {
        size: EmbeddingCache(
            args.output / "embeddings",
            records,
            args.processed,
            digest,
            DINO_BACKBONE_ID,
            model,
            size,
            batches[size],
            device,
            photo_hashes=photo_hashes,
        )
        for size in resolutions
    }
    signature = fingerprint({"config": config, "caches": {s: c.key for s, c in caches.items()}})
    write_json(
        results / "runtime_config.json",
        {
            "signature": signature,
            "batches": batches,
            "environment": caches[224].identity["execution_environment"],
        },
    )
    environment = caches[224].identity["execution_environment"]
    write_artifact(
        results / "environment_versions.txt",
        ("\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n").encode(),
    )
    review_dir = results / "similarity_review"
    review_summary = review_dir / "summary.json"
    if not review_summary.is_file():
        print("[단계] 원본 임베딩과 유사 후보 감사", flush=True)
        clean = caches[224].get(list(range(len(records))))
        print(
            f"[캐시 224px] 원본 재사용 {caches[224].hits} · 신규 {caches[224].extracted}",
            flush=True,
        )
        review_lists = similarity_reviews(
            records,
            hashes,
            clean,
            plan,
            audit["reviewed_pairs"],
        )
        summary = {}
        for name, rows in review_lists.items():
            if rows:
                write_csv(review_dir / f"{name}.csv", rows)
            summary[name] = {
                "pairs": len(rows),
                "cross_fold": sum(row["cross_fold"] for row in rows),
                "cross_holdout": sum(row["cross_holdout"] for row in rows),
                "cross_class": sum(row["left_class"] != row["right_class"] for row in rows),
                "unreviewed_cross_partition": sum(
                    (row["cross_fold"] or row["cross_holdout"]) and not row["review_decision"]
                    for row in rows
                ),
            }
        strong = [
            row
            for row in review_lists["dhash_le_4"]
            if row["cosine"] >= 0.98 and (row["cross_fold"] or row["cross_holdout"])
        ]
        if strong:
            write_csv(review_dir / "priority_cross_partition.csv", strong)
            write_json(
                review_dir / "priority_image_list.json",
                sorted({row[field] for row in strong for field in ("left", "right")}),
            )
        summary["priority_cross_partition"] = {
            "pairs": len(strong),
            "unreviewed": sum(not row["review_decision"] for row in strong),
        }
        write_json(review_summary, summary)
    pending = read_json(review_summary)["priority_cross_partition"]["unreviewed"]
    if pending:
        print(f"[감사] 분할 경계의 우선 검토 후보 {pending}쌍 · {review_dir}")
    if args.audit_only:
        print(f"[감사 완료] {results}")
        return
    if not plan["folds"]:
        raise ValueError(f"{plan['reason']} · 확인 결과는 {results}에 있습니다.")
    if pending:
        raise ValueError("우선 검토 후보를 확인한 뒤 --review와 새 RUN_ID로 실행하세요.")

    study = Study(
        args.output / "experiments" / args.run_id,
        signature,
        plan,
        args.threshold_policy,
        caches,
        device,
        targets,
    )
    best, resolution_rows = study.search(legacy, args.additional_budget, resolutions)
    export_tables(results, study.rows, resolution_rows, best)
    compare_positive_data(study, results / "data_ablation.json")
    nested_diagnostic(study, legacy, results / "nested_group_diagnostic.json")
    checkpoint = args.output / "checkpoints" / args.run_id / "final_model.pt"
    compare_final(study, best, results, checkpoint, manifest_hash)
    timing = results / "timing.json"
    if not timing.is_file():
        write_json(
            timing,
            {
                "elapsed_seconds": perf_counter() - start,
                "recorded_candidate_seconds": sum(row["elapsed_seconds"] for row in study.rows),
                "report_rows": len(study.rows),
                "cumulative_unique_cv": sum(row["reused_trial"] is None for row in study.rows),
                "cv_trained_in_this_process": study.new_fits,
                "rows_resumed_in_this_process": study.resumed,
                "scope": "this process including cache hits",
            },
        )
    print(
        f"[완료] {len(study.rows)}개 행 · 이 실행에서 {study.new_fits}개 CV 재학습 · "
        f"{study.resumed}개 완료 행 재사용 · {perf_counter() - start:.1f}초",
        flush=True,
    )
    print(f"[최종 모델] {checkpoint}", flush=True)


def main() -> None:
    """Colab 또는 로컬의 준비된 데이터에서 새 일반화 실험을 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--processed", type=Path, default=OUTPUTS_DIR / "processed_1121")
    parser.add_argument("--output", type=Path, default=COLAB_BACKUP_DIR)
    parser.add_argument("--run-id", default="dinov2-final-review-1121-01")
    parser.add_argument("--threshold-policy", default="max_f1")
    parser.add_argument("--fpr-targets", type=float, nargs="+", default=FPR_TARGETS)
    parser.add_argument("--legacy-results", type=Path)
    parser.add_argument("--additional-budget", type=int, default=31)
    parser.add_argument(
        "--input-sizes", type=int, nargs="+", choices=RESOLUTIONS, default=RESOLUTIONS
    )
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--prior-split", type=Path)
    parser.add_argument(
        "--quarantine-other-copies",
        action="store_true",
        help="다른 샴고양이 개체 사이의 바이트 동일 원본을 격리하고 연결 그룹을 함께 분할",
    )
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
