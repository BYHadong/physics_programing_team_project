"""DINOv2 Linear Probe 기준 실험부터 고정 Holdout 검증까지 실행합니다."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.config import (
    BACKBONE_INPUT_SIZE,
    BATCH_SIZE,
    DINO_BACKBONE_ID,
    DINO_BACKBONE_KEY,
    EXPERIMENTS_DIR,
    GROUP_CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SOURCE_DATA_DIR,
    TARGET_UNSEEN_FPR,
)
from src.embedding_cache import EmbeddingCache, fingerprint
from src.experiment_splits import audit_dataset, load_manifest, source_index, split_plan
from src.linear_probe import (
    METRIC_DEFINITIONS,
    THRESHOLD_POLICIES,
    Hyperparameters,
    checkpoint_payload,
    evaluate,
    fit_probe,
    fold_scores,
    probe_scores,
    thresholds_from_validation,
)
from src.train_baseline import load_backbone, set_seed

METHOD = "linear_probe"
SEARCH_STAGES = (
    ("regularization", "c", (0.01, 0.05, 0.1, 0.5, 1.0)),
    ("negative_augmentation", "neg_aug_passes", (1, 3, 5, 7)),
    ("positive_augmentation", "main_target_count", (250, 500, 750)),
    ("rotation", "rotation_degrees", (5.0, 15.0, 30.0)),
    ("horizontal_flip", "hflip_probability", (0.0, 0.25, 0.5, 0.75, 1.0)),
)
RANK_ORDER = ("hard_siamese_roc_auc", "lower_unseen_siamese_fpr", "f1", "lower_fold_hard_std")


def read_json(path: Path) -> dict:
    """저장된 실험 JSON 파일을 읽습니다."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifact(path: Path, data: bytes) -> None:
    """미완성 파일을 숨기고 기존 결과와 다른 내용의 덮어쓰기를 방지합니다."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"기존 결과와 내용이 다릅니다. 새 run-id를 사용하세요: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_bytes(data)
    temp.replace(path)


def write_json(path: Path, data: object) -> None:
    """JSON 실험 결과를 UTF-8로 저장합니다."""
    write_artifact(
        path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """실험 또는 Fold별 지표를 Excel에서 읽을 수 있는 CSV로 저장합니다."""
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    write_artifact(path, stream.getvalue().encode("utf-8-sig"))


def metric_rank(trial: dict) -> tuple[float, ...]:
    """Hard ROC-AUC를 우선하고 FPR, F1, Fold 표준편차 순으로 동률을 해소합니다."""
    metrics = trial["metrics"]
    std = trial["fold_summary"]["hard_siamese_roc_auc"]["std"]
    return (
        metrics["hard_siamese_roc_auc"],
        -metrics["unseen_siamese_fpr"],
        metrics["f1"],
        -std if std is not None else -float("inf"),
    )


def summarize_folds(folds: list[dict]) -> dict:
    """Fold별 수치 지표의 평균, 모집단 표준편차 및 유효 Fold 수를 반환합니다."""
    keys = set(folds[0]["metrics"]) - {"confusion_matrix"}
    summary = {}
    for key in sorted(keys):
        values = [fold["metrics"][key] for fold in folds if fold["metrics"][key] is not None]
        summary[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "valid_folds": len(values),
        }
    return summary


def class_names(records: list[dict], indices: list[int]) -> list[str]:
    """원본 인덱스 순서에 맞는 클래스 목록을 반환합니다."""
    return [records[i]["class_name"] for i in indices]


def cv_trial(cache: EmbeddingCache, plan: dict, params: Hyperparameters, policy: str) -> dict:
    """5-Fold OOF 점수, 네 임계값 정책, Fold별 지표를 반환합니다."""
    development = plan["development"]
    folds = plan["folds"]
    with threadpool_limits(limits=1):
        scores, counts = fold_scores(cache, development, folds, params, METHOD)
    classes = class_names(cache.records, development)
    policies = thresholds_from_validation(classes, scores)
    threshold = policies[policy]["threshold"]
    lookup = dict(zip(development, scores))
    fold_results = []
    for number, (fold, sizes) in enumerate(zip(folds, counts), start=1):
        validation = fold["validation"]
        fold_results.append(
            {
                "fold": number,
                "training_counts": sizes,
                "metrics": evaluate(
                    class_names(cache.records, validation),
                    np.asarray([lookup[i] for i in validation]),
                    threshold,
                ),
            }
        )
    return {
        "parameters": asdict(params),
        "metrics": policies[policy],
        "threshold_policies": policies,
        "fold_results": fold_results,
        "fold_summary": summarize_folds(fold_results),
        "oof_predictions": {
            "indices": development,
            "classes": classes,
            "scores": scores.tolist(),
        },
    }


class ExperimentRunner:
    """고정 데이터와 설정으로 실험을 실행하고 완료된 결과를 불러옵니다."""

    def __init__(
        self, output: Path, signature: str, cache: EmbeddingCache, plan: dict, policy: str
    ) -> None:
        """실험 파일 경로, 분할, 캐시 및 선택할 임계값 정책을 지정합니다."""
        self.output, self.signature, self.cache = output, signature, cache
        self.plan, self.policy = plan, policy
        self.trials: list[dict] = []

    def run_trial(
        self,
        params: Hyperparameters,
        stage: str,
        field: str | None = None,
        anchor: str | None = None,
    ) -> dict:
        """실험 번호와 설정을 검증하여 완료 결과를 재사용하거나 새로 측정합니다."""
        exp_id = f"E{len(self.trials):02d}"
        path = self.output / exp_id / "result.json"
        context = {
            "run_signature": self.signature,
            "experiment": exp_id,
            "stage": stage,
            "field": field,
            "anchor": anchor,
            "parameters": asdict(params),
        }
        if path.exists():
            result = read_json(path)
            if any(result.get(k) != v for k, v in context.items()):
                raise ValueError(f"완료 실험과 현재 설정이 다릅니다: {path}")
            print(f"[{exp_id}] 완료 결과 재개", flush=True)
        else:
            start = time.perf_counter()
            print(f"[{exp_id}] {stage} {asdict(params)}", flush=True)
            duplicate = next((r for r in self.trials if r["parameters"] == asdict(params)), None)
            result = (
                duplicate.copy()
                if duplicate
                else cv_trial(self.cache, self.plan, params, self.policy)
            )
            result.update(context)
            result["reused_trial"] = duplicate["experiment"] if duplicate else None
            result["elapsed_seconds"] = time.perf_counter() - start
            write_json(path, result)
        metrics = result["metrics"]
        print(
            f"[{exp_id}] Hard AUC={metrics['hard_siamese_roc_auc']:.4f}, "
            f"FPR={metrics['unseen_siamese_fpr']:.2%}, F1={metrics['f1']:.4f}",
            flush=True,
        )
        self.trials.append(result)
        return result

    def search(self, space: dict[str, list]) -> dict:
        """각 후보 단계에서 한 변수만 바꾸고 상위 대안을 제한적으로 조합합니다."""
        incumbent = self.trials[0]
        trials_by_field = {}
        for stage, field, _ in SEARCH_STAGES:
            params = Hyperparameters(**incumbent["parameters"])
            reference = incumbent["experiment"]
            stage_trials = [
                self.run_trial(replace(params, **{field: value}), stage, field, reference)
                for value in space[field]
            ]
            trials_by_field[field] = stage_trials
            incumbent = max([incumbent, *stage_trials], key=metric_rank)

        combined_fields = set()
        positive_arm = max(trials_by_field["main_target_count"], key=metric_rank)
        positive_params = Hyperparameters(**positive_arm["parameters"])
        # 양성 장수 선택과 별도로 같은 양성 장수에서 음성 증강 비대칭을 한 번 검증합니다.
        for trial in sorted(trials_by_field["neg_aug_passes"], key=metric_rank, reverse=True):
            config = replace(positive_params, neg_aug_passes=trial["parameters"]["neg_aug_passes"])
            if any(r["parameters"] == asdict(config) for r in self.trials):
                continue
            combined = self.run_trial(
                config, "combination", "neg_aug_passes", positive_arm["experiment"]
            )
            incumbent = max([incumbent, combined], key=metric_rank)
            combined_fields.add("neg_aug_passes")
            break

        single_factor_trials = [row for rows in trials_by_field.values() for row in rows]
        for trial in sorted(single_factor_trials, key=metric_rank, reverse=True):
            field = trial["field"]
            if field in combined_fields:
                continue
            value = trial["parameters"][field]
            params = Hyperparameters(**incumbent["parameters"])
            config = replace(params, **{field: value})
            if any(r["parameters"] == asdict(config) for r in self.trials):
                continue
            combined = self.run_trial(config, "combination", field, incumbent["experiment"])
            incumbent = max([incumbent, combined], key=metric_rank)
            combined_fields.add(field)
            if len(combined_fields) == 3:
                break
        return incumbent


def search_space(path: Path | None) -> dict[str, list]:
    """기본 후보를 구성하고 외부 JSON의 필드별 후보를 검증합니다."""
    space = {field: list(default) for _, field, default in SEARCH_STAGES}
    if path is not None:
        custom = read_json(path)
        if not isinstance(custom, dict) or not custom.keys() <= space.keys():
            raise ValueError("탐색 설정에는 지원되는 하이퍼파라미터 이름만 사용하세요.")
        space.update(custom)
    for field, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} 후보는 비어 있지 않은 목록이어야 합니다.")
        for value in values:
            replace(Hyperparameters(), **{field: value})
    return space


def export_search(results_dir: Path, trials: list[dict], best: dict) -> None:
    """종합 CSV, JSON 및 선정 근거를 보존합니다."""
    write_json(results_dir / "hyperparameter_results.json", trials)
    write_csv(
        results_dir / "hyperparameter_results.csv",
        [
            {
                "experiment": r["experiment"],
                "stage": r["stage"],
                "field": r["field"],
                "anchor": r["anchor"],
                **r["parameters"],
                **r["metrics"],
                "fold_hard_auc_mean": r["fold_summary"]["hard_siamese_roc_auc"]["mean"],
                "fold_hard_auc_std": r["fold_summary"]["hard_siamese_roc_auc"]["std"],
                **{
                    f"fold_{name}_{stat}": values[stat]
                    for name, values in r["fold_summary"].items()
                    for stat in ("mean", "std", "valid_folds")
                },
                "target_1_22_percent_fpr_recall": r["threshold_policies"]["target_siamese_fpr"][
                    "recall"
                ],
                "target_1_22_percent_achieved_fpr": r["threshold_policies"]["target_siamese_fpr"][
                    "unseen_siamese_fpr"
                ],
                "elapsed_seconds": r["elapsed_seconds"],
                "reused_trial": r["reused_trial"],
            }
            for r in trials
        ],
    )
    write_csv(
        results_dir / "fold_results.csv",
        [
            {
                "experiment": trial["experiment"],
                "fold": row["fold"],
                **row["training_counts"],
                **row["metrics"],
            }
            for trial in trials
            for row in trial["fold_results"]
        ],
    )
    write_json(
        results_dir / "best_hyperparameters.json",
        {
            "experiment": best["experiment"],
            "parameters": best["parameters"],
            "rank_order": RANK_ORDER,
            "validation_metrics": best["metrics"],
            "validation_threshold_policies": best["threshold_policies"],
            "scope": "Best among evaluated candidates on the development folds.",
        },
    )


def fit_final(
    runner: ExperimentRunner,
    baseline: dict,
    best: dict,
    results_dir: Path,
    checkpoint_path: Path,
) -> None:
    """튜닝에 사용하지 않은 Holdout으로 E00과 개발 데이터 재학습 모델을 평가합니다."""
    final_metrics = results_dir / "final_model_metrics.json"
    checkpoint_exists = checkpoint_path.is_file()
    report_exists = final_metrics.is_file()
    if checkpoint_exists:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            saved.get("run_signature") != runner.signature
            or saved.get("hyperparameters") != best["parameters"]
        ):
            raise ValueError("기존 최종 체크포인트와 현재 실험 설정이 다릅니다.")
    if report_exists and read_json(final_metrics).get("run_signature") != runner.signature:
        raise ValueError("기존 최종 평가 결과와 현재 실험 설정이 다릅니다.")
    if report_exists and checkpoint_exists:
        print(f"[최종 모델] 완료 결과 확인: {checkpoint_path}", flush=True)
        return
    start = time.perf_counter()
    plan, cache = runner.plan, runner.cache
    dev, holdout = plan["development"], plan["holdout"]
    final_scores = {}
    payload = None
    with threadpool_limits(limits=1):
        for label, trial in (("final", best), ("baseline", baseline)):
            if label == "baseline" and best["experiment"] == baseline["experiment"]:
                final_scores[label] = final_scores["final"]
                continue
            params = Hyperparameters(**trial["parameters"])
            classifier, prototype, counts = fit_probe(cache, dev, params)
            threshold = trial["metrics"]["threshold"]
            scores = (
                probe_scores(cache.get(holdout), classifier, prototype, METHOD) if holdout else []
            )
            classes = class_names(cache.records, holdout)
            final_scores[label] = {
                "experiment": trial["experiment"],
                "training_counts": counts,
                "metrics": evaluate(classes, scores, threshold) if holdout else None,
                "threshold_policies_fixed_on_development": {
                    name: evaluate(classes, scores, row["threshold"])
                    for name, row in trial["threshold_policies"].items()
                }
                if holdout
                else None,
                "predictions": {"indices": holdout, "classes": classes, "scores": list(scores)},
            }
            if label == "final":
                payload = checkpoint_payload(
                    DINO_BACKBONE_KEY,
                    DINO_BACKBONE_ID,
                    BACKBONE_INPUT_SIZE,
                    METHOD,
                    classifier,
                    prototype,
                    threshold,
                    {
                        "hyperparameters": asdict(params),
                        "threshold_policy": runner.policy,
                        "manifest_sha256": read_json(results_dir / "experiment_config.json")[
                            "manifest_sha256"
                        ],
                        "main_target_count": params.main_target_count,
                        "metrics": trial["metrics"],
                        "training_indices": dev,
                        "holdout_indices": holdout,
                        "dataset_digest": cache.identity["dataset_and_manifest"],
                        "embedding_cache_key": cache.key,
                        "run_signature": runner.signature,
                        "backbone_weights_sha256": cache.identity["weights_sha256"],
                    },
                )
    rows = []
    for label, trial in (("baseline", baseline), ("final", best)):
        rows.append(
            {
                "model": label,
                "split": "development_oof_tuning",
                "experiment": trial["experiment"],
                **trial["metrics"],
            }
        )
        if holdout:
            rows.append(
                {
                    "model": label,
                    "split": "reserved_group_holdout",
                    "experiment": trial["experiment"],
                    **final_scores[label]["metrics"],
                }
            )
    write_csv(results_dir / "baseline_comparison.csv", rows)
    if not checkpoint_exists:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(f".{uuid.uuid4().hex}.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
    if not report_exists:
        write_json(
            final_metrics,
            {
                "run_signature": runner.signature,
                "threshold_policy": runner.policy,
                "selected_experiment": best["experiment"],
                "development_oof_tuning": {
                    "baseline": baseline["metrics"],
                    "final": best["metrics"],
                },
                "reserved_group_holdout": final_scores if holdout else None,
                "independent_external_dataset": False,
                "holdout_index_count": len(holdout),
                "elapsed_seconds": time.perf_counter() - start,
                "limitations": [
                    "The reserved groups are from the same dataset.",
                    "Main photos without capture evidence may share an unrecognized scene.",
                    "Near-duplicate flags are candidates for review.",
                ]
                + (
                    []
                    if holdout
                    else ["No reserved split was feasible. OOF metrics guided tuning."]
                ),
            },
        )
    print(f"[최종 모델] {checkpoint_path}", flush=True)


def run(args: argparse.Namespace) -> None:
    """설정 검증, Drive 저장, 재개, 단계적 탐색 및 최종 재학습을 실행합니다."""
    set_seed(RANDOM_SEED)
    space = search_space(args.search_space)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("run-id는 1~80자의 영문·숫자·밑줄·하이픈을 사용하세요.")
    manifest_path = args.manifest or args.processed / "manifest.json"
    records = load_manifest(manifest_path)
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    dataset_digest, groups, audit = audit_dataset(
        records, args.processed, source_index(args.source), manifest_hash
    )
    plan = split_plan(records, groups, audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_backbone(DINO_BACKBONE_ID, device)
    cache = EmbeddingCache(
        args.output / "embeddings",
        records,
        args.processed,
        dataset_digest,
        DINO_BACKBONE_ID,
        model,
        img_size,
        BATCH_SIZE,
        device,
    )
    paths = {
        "results": args.output / "results" / args.run_id,
        "experiments": args.output / "experiments" / args.run_id,
        "checkpoint": args.output / "checkpoints" / args.run_id / "final_model.pt",
    }
    source_hashes = {
        name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "config.py",
            "tune_hyperparameters.py",
            "experiment_splits.py",
            "linear_probe.py",
        )
    }
    config = {
        "run_id": args.run_id,
        "seed": RANDOM_SEED,
        "folds": GROUP_CV_FOLDS,
        "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_digest,
        "split_fingerprint": fingerprint(plan),
        "source_sha256": source_hashes,
        "cache_fingerprint": cache.key,
        "cache_identity": cache.identity,
        "baseline": asdict(Hyperparameters()),
        "backbone": DINO_BACKBONE_ID,
        "method": METHOD,
        "backbone_frozen": True,
        "classifier": "LogisticRegression (L2)",
        "threshold_policy": args.threshold_policy,
        "target_unseen_siamese_fpr": TARGET_UNSEEN_FPR,
        "search_space": space,
        "rank_order": RANK_ORDER,
        "metric_definitions": METRIC_DEFINITIONS,
    }
    result_dir = paths["results"]
    write_json(result_dir / "experiment_config.json", config)
    write_json(result_dir / "fold_indices.json", plan)
    write_json(result_dir / "data_audit.json", audit)
    env = cache.identity["execution_environment"]
    write_artifact(
        result_dir / "environment_versions.txt",
        (
            "\n".join(f"{k}={v}" for k, v in env.items() if k != "packages")
            + "\n"
            + "\n".join(f"{k}=={v}" for k, v in env["packages"].items())
            + "\n"
        ).encode(),
    )
    runner = ExperimentRunner(
        paths["experiments"],
        fingerprint(config),
        cache,
        plan,
        args.threshold_policy,
    )
    baseline = runner.run_trial(Hyperparameters(), "baseline")
    if args.through == "baseline":
        return
    best = runner.search(space)
    export_search(result_dir, runner.trials, best)
    if args.through == "final":
        fit_final(runner, baseline, best, result_dir, paths["checkpoint"])


def main() -> None:
    """명령행 인자를 읽고 요청한 단계까지 실험을 진행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
    )
    parser.add_argument("--through", choices=("baseline", "search", "final"), default="final")
    parser.add_argument("--search-space", type=Path, help="하이퍼파라미터 후보 목록 JSON")
    parser.add_argument("--threshold-policy", choices=THRESHOLD_POLICIES, default="max_f1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
