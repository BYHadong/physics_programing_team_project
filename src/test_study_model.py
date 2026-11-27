"""저장한 DINOv2 모델의 단일 이미지와 그룹 테스트를 독립적으로 실행합니다."""

from __future__ import annotations

import argparse
import re
from secrets import randbits
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from src.config import COLAB_BACKUP_DIR, OUTPUTS_DIR
from src.experiment_splits import load_manifest, source_index
from src.predict import predict_image, verified_crop_record
from src.study_audit import original_pixel_sha256
from src.test_random_samples import (
    manifest_sample_items,
    run_random_sample_test,
    sample_per_class,
    study_sample_records,
)
from src.tune_hyperparameters import write_json

MODES = ("single", "random", "strict", "reserved-all", "reproducibility", "external")


def run_test(args: argparse.Namespace) -> Path:
    """선택한 테스트를 실행하고 고유 폴더에 결과를 저장합니다."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("RUN_ID에는 영문·숫자·밑줄·하이픈만 사용할 수 있습니다.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{args.mode}-{stamp}-{uuid4().hex[:8]}"
    directory = args.output / "graphs" / args.run_id / "sample_tests" / name
    manifest = args.processed / "manifest.json"
    checkpoint = args.checkpoint or args.output / "checkpoints" / args.run_id / "final_model.pt"
    summary = directory / "results.json"
    if args.mode == "single":
        if args.image is None:
            raise ValueError("single 모드에 --image가 필요합니다.")
        record = None
        if manifest.is_file():
            try:
                record = verified_crop_record(args.image, args.processed, args.source)
            except ValueError:
                pass  # 새 외부 이미지는 기존 단일 이미지 전처리를 사용합니다.
        prediction = predict_image(
            args.image,
            checkpoint,
            return_crop=True,
            processed_record=record,
            processed_dir=args.processed,
        )
        crop = prediction.pop("square_img")
        directory.mkdir(parents=True, exist_ok=True)
        crop.save(directory / "predictions.png")
        write_json(summary, prediction)
        print(f"{prediction['prediction']} · score {prediction['score']:.4f}")
    elif args.mode == "reproducibility":
        if args.test_seed is None:
            raise ValueError("고정 시드 재현성 확인에 정수 --test-seed가 필요합니다.")
        records = load_manifest(manifest, require_other_ids=False)
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            records = study_sample_records(records, manifest, saved)
        items = manifest_sample_items(records, args.source)
        first, second = (
            sample_per_class(args.source, args.n_per_class, args.test_seed, items) for _ in range(2)
        )
        write_json(
            summary,
            {
                "test_seed": args.test_seed,
                "same_sample_order": first == second,
                "scope": "image selection only",
                "samples": [
                    {"image_path": str(path), "class": cls, "individual": identity}
                    for path, cls, identity in first
                ],
            },
        )
    else:
        strict, reserved = args.mode == "strict", args.mode == "reserved-all"
        external = args.mode == "external"
        source = args.source
        count = args.n_per_class
        if external:
            if args.external_source is None:
                raise ValueError("external 모드에 --external-source가 필요합니다.")
            source = args.external_source
            originals = source_index(args.source)
            new = source_index(source)
            baseline_hashes = {original_pixel_sha256(path) for path in originals.values()}
            repeated = [
                key for key, path in new.items() if original_pixel_sha256(path) in baseline_hashes
            ]
            if repeated:
                raise ValueError(
                    f"외부 평가와 기존 데이터에 같은 원본 {len(repeated)}장이 있습니다."
                )
            if not {"main_siamese_cat", "other_siamese_cat"} <= {k.split("/")[0] for k in new}:
                raise ValueError("외부 평가에는 특정 고양이와 다른 샴고양이가 모두 필요합니다.")
            count = len(new)
            write_json(
                directory / "external_audit.json",
                {
                    "source": str(source),
                    "external_images": len(new),
                    "exact_pixel_sha256_overlap": 0,
                    "session_independence_requires_confirmation": True,
                },
            )
        if strict and not args.strict_development:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            reserved = bool(saved.get("holdout_indices"))
        if args.mode == "reserved-all":
            count = len(load_manifest(manifest, require_other_ids=False))
        run_random_sample_test(
            source_dir=source,
            checkpoint_path=checkpoint,
            n_per_class=count,
            seed=randbits(32) if args.test_seed is None else args.test_seed,
            strict_holdout=strict,
            output_json=summary,
            output_grid=directory / "predictions.png",
            processed_dir=args.processed,
            cache_dir=args.output / "embeddings",
            reuse_processed=not external,
            reserved_cohort=reserved,
            external_cohort=external,
        )
    print(f"[테스트 저장] {directory}", flush=True)
    return directory


def main() -> None:
    """학습과 별개로 사용할 테스트 모드와 샘플링 인자를 받습니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--processed", type=Path, default=OUTPUTS_DIR / "processed_1121")
    parser.add_argument("--output", type=Path, default=COLAB_BACKUP_DIR)
    parser.add_argument("--run-id", default="dinov2-final-review-1121-01")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--external-source", type=Path)
    parser.add_argument("--n-per-class", type=int, default=10)
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--test-seed", type=int, default=123)
    seed.add_argument("--random-seed", action="store_const", const=None, dest="test_seed")
    parser.add_argument("--strict-development", action="store_true")
    run_test(parser.parse_args())


if __name__ == "__main__":
    main()
