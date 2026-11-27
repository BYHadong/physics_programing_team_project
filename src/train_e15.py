"""고정된 E15 설정으로 DINOv2 Linear Probe를 학습하고 평가합니다."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter

import torch
from threadpoolctl import threadpool_limits

from src.config import (
    BATCH_SIZE,
    DINO_BACKBONE_ID,
    DINO_BACKBONE_KEY,
    EXPERIMENTS_DIR,
    GROUP_CV_FOLDS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SOURCE_DATA_DIR,
)
from src.embedding_cache import EmbeddingCache, fingerprint
from src.experiment_splits import audit_dataset, load_manifest, source_index, split_plan
from src.linear_probe import (
    THRESHOLD_POLICIES,
    Hyperparameters,
    checkpoint_payload,
    evaluate,
    fit_probe,
    probe_scores,
)
from src.train_baseline import load_backbone, set_seed
from src.tune_hyperparameters import (
    METHOD,
    class_names,
    cv_trial,
    read_json,
    write_artifact,
    write_json,
)

E15 = Hyperparameters(
    c=0.01,
    main_target_count=750,
    neg_aug_passes=5,
    rotation_degrees=30.0,
    hflip_probability=0.5,
)


def run(args: argparse.Namespace) -> Path:
    """같은 그룹 분할과 캐시를 사용해 E15만 학습하고 체크포인트를 반환합니다."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("run-id는 1~80자의 영문·숫자·밑줄·하이픈을 사용하세요.")
    set_seed(RANDOM_SEED)

    manifest = args.processed / "manifest.json"
    records = load_manifest(manifest)
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    digest, groups, audit = audit_dataset(
        records, args.processed, source_index(args.source), manifest_hash
    )
    plan = split_plan(records, groups, audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_backbone(DINO_BACKBONE_ID, device)
    cache = EmbeddingCache(
        args.output / "embeddings",
        records,
        args.processed,
        digest,
        DINO_BACKBONE_ID,
        model,
        image_size,
        BATCH_SIZE,
        device,
    )
    config = {
        "run_id": args.run_id,
        "experiment": "E15",
        "parameters": asdict(E15),
        "threshold_policy": args.threshold_policy,
        "seed": RANDOM_SEED,
        "folds": GROUP_CV_FOLDS,
        "backbone": DINO_BACKBONE_ID,
        "method": METHOD,
        "manifest_sha256": manifest_hash,
        "split_fingerprint": fingerprint(plan),
        "embedding_cache_key": cache.key,
        "execution_environment": cache.identity["execution_environment"],
        "source_sha256": {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "train_e15.py",
                "config.py",
                "tune_hyperparameters.py",
                "experiment_splits.py",
                "linear_probe.py",
            )
        },
    }
    signature = fingerprint(config)
    results = args.output / "results" / args.run_id
    checkpoint = args.output / "checkpoints" / args.run_id / "final_model.pt"
    report_path = results / "final_model_metrics.json"
    write_json(results / "experiment_config.json", config)
    write_json(results / "fold_indices.json", plan)
    write_json(results / "data_audit.json", audit)
    print(f"[E15] 원본 {len(records)}장 · {device} · 고정 설정 {asdict(E15)}", flush=True)

    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("run_signature") != signature:
            raise ValueError("기존 E15 체크포인트와 현재 실험 설정이 다릅니다.")
    if report_path.is_file():
        if read_json(report_path).get("run_signature") != signature:
            raise ValueError("기존 E15 평가 결과와 현재 실험 설정이 다릅니다.")
        if checkpoint.is_file():
            print(f"[E15] 완료된 체크포인트 사용 · {checkpoint}", flush=True)
            return checkpoint

    validation_path = results / "e15_validation.json"
    if validation_path.is_file():
        validation = read_json(validation_path)
        if validation.get("run_signature") != signature:
            raise ValueError("기존 E15 검증 결과와 현재 실험 설정이 다릅니다.")
        print("[E15] 저장한 5-Fold 검증 점수 사용", flush=True)
    else:
        print("[E15] 5-Fold 검증과 임계값 보정", flush=True)
        start = perf_counter()
        validation = cv_trial(cache, plan, E15, args.threshold_policy)
        validation.update(
            experiment="E15", run_signature=signature, elapsed_seconds=perf_counter() - start
        )
        write_json(validation_path, validation)

    start = perf_counter()
    train, holdout = plan["development"], plan["holdout"]
    threshold = validation["metrics"]["threshold"]
    print("[E15] 최종 Linear Probe 학습", flush=True)
    with threadpool_limits(limits=1):
        classifier, prototype, counts = fit_probe(cache, train, E15)
        holdout_report = None
        if holdout:
            scores = probe_scores(cache.get(holdout), classifier, prototype, METHOD)
            classes = class_names(records, holdout)
            holdout_report = {
                "metrics": evaluate(classes, scores, threshold),
                "predictions": {"indices": holdout, "classes": classes, "scores": scores.tolist()},
            }

    if not checkpoint.is_file():
        payload = checkpoint_payload(
            DINO_BACKBONE_KEY,
            DINO_BACKBONE_ID,
            image_size,
            METHOD,
            classifier,
            prototype,
            threshold,
            {
                "experiment": "E15",
                "hyperparameters": asdict(E15),
                "main_target_count": E15.main_target_count,
                "threshold_policy": args.threshold_policy,
                "metrics": validation["metrics"],
                "manifest_sha256": manifest_hash,
                "training_indices": train,
                "holdout_indices": holdout,
                "dataset_digest": digest,
                "embedding_cache_key": cache.key,
                "run_signature": signature,
                "backbone_weights_sha256": cache.identity["weights_sha256"],
            },
        )
        buffer = BytesIO()
        torch.save(payload, buffer)
        write_artifact(checkpoint, buffer.getvalue())

    if not report_path.is_file():
        write_json(
            report_path,
            {
                "run_signature": signature,
                "selected_experiment": "E15",
                "threshold_policy": args.threshold_policy,
                "threshold": threshold,
                "training_counts": counts,
                "development_oof": validation["metrics"],
                "reserved_group_holdout": {"final": holdout_report} if holdout_report else None,
                "holdout_index_count": len(holdout),
                "independent_external_dataset": False,
                "elapsed_seconds": perf_counter() - start,
            },
        )
    print(f"[E15] 최종 체크포인트 · {checkpoint}", flush=True)
    return checkpoint


def main() -> None:
    """전처리 위치와 출력 경로를 받아 E15 단독 학습을 시작합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--processed", type=Path, default=OUTPUTS_DIR / "processed_1121")
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--run-id", default="dinov2-e15-1121-01")
    parser.add_argument("--threshold-policy", choices=THRESHOLD_POLICIES, default="max_f1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
