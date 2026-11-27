"""
main_siamese_cat, other_siamese_cat, etc_cats 각 클래스에서 N장(기본 10장)씩
랜덤으로 추출하여 모델 판별 성능을 종합 테스트하는 CLI 스크립트.
"""

import argparse
import json
import random
import secrets
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CROP_INTEGRITY_FILENAME,
    EXPERIMENTS_DIR,
    GROUP_CV_FOLDS,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SOURCE_DATA_DIR,
    YOLO_WEIGHTS_PATH,
)
from src.dataset import VALID_IMAGE_EXTS
from src.embedding_cache import EmbeddingCache, fingerprint
from src.experiment_splits import (
    MAIN,
    OTHER,
    load_manifest,
    matching_record,
    processed_path,
    source_index,
    source_key,
    strict_training_indices,
)
from src.group_validation import check_partition, choose_folds
from src.linear_probe import (
    Hyperparameters,
    checkpoint_payload,
    evaluate,
    fit_probe,
    fold_scores,
    thresholds_from_validation,
)
from src.predict import load_checkpoint_backbone, predict_image
from src.preprocess import collect_source_files
from src.study_audit import file_sha256, processed_digest, scan_dataset
from src.train_baseline import set_seed
from src.tune_hyperparameters import write_artifact

matplotlib.use("Agg")


CLASS_ORDER = ["main_siamese_cat", "other_siamese_cat", "etc_cats"]
DEFAULT_TEST_SEED = 123
CLASS_ABBR = {
    "main_siamese_cat": "MAIN",
    "other_siamese_cat": "OTHER",
    "etc_cats": "ETC",
}
ROW_LABELS = {
    "main_siamese_cat": "Main 사진",
    "other_siamese_cat": "Other 사진",
    "etc_cats": "ETC 사진",
}


def sample_per_class(
    source_dir: Path = SOURCE_DATA_DIR,
    n_per_class: int = 10,
    seed: Optional[int] = DEFAULT_TEST_SEED,
    items: Optional[List[Tuple[Path, str, str]]] = None,
    main_dir: Optional[Path] = None,
) -> List[Tuple[Path, str, str]]:
    """
    원본 데이터 디렉토리에서 클래스별로 n_per_class장씩 중복 없이 랜덤 추출합니다.
    main_dir이 지정되면 main_siamese_cat 샘플은 해당 폴더에서 추출합니다.

    매개변수:
        source_dir: 클래스 폴더가 포함된 원본 데이터 루트 경로
        n_per_class: 클래스당 추출할 이미지 장수
        seed: 난수 시드 (None이면 매 실행마다 다른 랜덤 추출)
        items: 사전 지정한 샘플 후보 원본 목록
        main_dir: main_siamese_cat 대체 외부 테스트 폴더 경로 (선택)

    반환값:
        (파일 경로, class_name, individual_id) 튜플 리스트
    """
    if n_per_class < 1:
        raise ValueError("클래스별 샘플 수는 1장 이상이어야 합니다.")
    all_items = collect_source_files(Path(source_dir)) if items is None else items
    if not all_items:
        raise FileNotFoundError(f"샘플링할 이미지를 찾을 수 없습니다: {source_dir}")

    grouped: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)
    for item in all_items:
        grouped[item[1]].append(item)

    if main_dir is not None:
        main_path = Path(main_dir)
        if not main_path.exists() or not main_path.is_dir():
            raise FileNotFoundError(f"지정한 main 테스트 폴더를 찾을 수 없습니다: {main_path}")
        custom_main = [
            (fp, MAIN, MAIN)
            for fp in sorted(main_path.rglob("*"))
            if fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS
        ]
        if not custom_main:
            raise FileNotFoundError(f"지정한 main 테스트 폴더에 이미지가 없습니다: {main_path}")
        grouped[MAIN] = custom_main

    rng = random.Random(seed)
    sampled: List[Tuple[Path, str, str]] = []

    ordered_classes = [c for c in CLASS_ORDER if c in grouped] + [
        c for c in sorted(grouped.keys()) if c not in CLASS_ORDER
    ]
    for cls_name in ordered_classes:
        cls_items = grouped[cls_name]
        k = min(n_per_class, len(cls_items))
        sampled.extend(rng.sample(cls_items, k))

    return sampled


def prepare_holdout(
    records: List[Dict], processed_dir: Path, source_dir: Path, manifest_hash: str
) -> Tuple[str, Dict[str, str], list[int]]:
    """원본 EXIF와 전처리 내용을 대조해 검증용 그룹과 캐시 키를 반환합니다."""
    digest, groups, excluded, _, audit = scan_dataset(
        records, processed_dir, source_index(source_dir), manifest_hash
    )
    if audit["exact_or_review_conflicts"]:
        raise ValueError("Strict Holdout 원본에 동일 사진 또는 개체 ID 충돌이 있습니다.")
    return digest, groups, excluded


def group_sample_pool(
    records: List[Dict], groups: Dict[str, str], source_dir: Path, n: int, seed: Optional[int]
) -> List[Tuple[Path, str, str]]:
    """내부 보정용 개체·촬영 그룹을 남기는 Strict Holdout 후보를 반환합니다."""
    raw = source_index(source_dir)
    buckets = defaultdict(lambda: defaultdict(list))
    for record in records:
        cls = record["class_name"]
        path = raw.get(source_key(record["source_path"], cls))
        if path is not None:
            buckets[cls][groups[record["processed_rel_path"]]].append(
                (path, cls, record.get("individual_id", cls))
            )
    rng = random.Random(seed)
    pool = []
    for cls in CLASS_ORDER:
        available = list(buckets[cls].values())
        rng.shuffle(available)
        keep = 2 if cls in (MAIN, OTHER) else 1
        if len(available) <= keep:
            raise ValueError(f"Strict Holdout 샘플링에 필요한 {cls} 독립 그룹이 부족합니다.")
        selected = []
        for group in available[:-keep]:
            selected.extend(group)
            if len(selected) >= n:
                break
        pool.extend(selected)
    return pool


def study_partition(base_ckpt: Dict, records: List[Dict], folder: Path, manifest_hash: str) -> str:
    """새 체크포인트의 저장 분할이 동일한 데이터에 안전하게 적용되는지 확인합니다."""
    digest = processed_digest(records, folder, manifest_hash)
    if (
        base_ckpt.get("manifest_sha256") != manifest_hash
        or base_ckpt.get("dataset_digest") != digest
    ):
        raise ValueError("체크포인트의 Manifest 또는 전처리 이미지가 현재 데이터와 다릅니다.")
    check_partition(
        records,
        base_ckpt["study_groups"],
        base_ckpt["training_indices"],
        base_ckpt["holdout_indices"],
        base_ckpt["study_excluded"],
    )
    return digest


def fit_holdout_checkpoint(
    sampled_items: List[Tuple[Path, str, str]],
    base_ckpt: Dict,
    device: torch.device,
    backbone_model: Optional[torch.nn.Module] = None,
    manifest_path: Path = PROCESSED_DATA_DIR / "manifest.json",
    processed_dir: Path = PROCESSED_DATA_DIR,
    *,
    source_dir: Path = SOURCE_DATA_DIR,
    cache_dir: Path = EXPERIMENTS_DIR / "embeddings",
    prepared: Optional[Tuple[List[Dict], str, Dict[str, str], list[int]]] = None,
) -> Dict:
    """
    원본과 해당 그룹을 제외하고 저장 설정으로 재학습 및 임계값을 보정합니다.

    sampled_items는 테스트 원본, base_ckpt는 모델·증강 조건, device와 backbone_model은
    특징 추출 장치와 재사용 모델입니다. manifest_path, processed_dir, source_dir에서
    데이터를 대조하고 cache_dir에 특징을 저장합니다. prepared는 사전 계산한 감사 결과입니다.
    반환값은 내부 검증으로 임계값을 다시 정한 추론용 임시 체크포인트입니다.
    """
    set_seed(RANDOM_SEED)
    if base_ckpt.get("train_seed", RANDOM_SEED) != RANDOM_SEED:
        raise ValueError("Strict Holdout의 TRAIN_SEED가 저장된 모델과 다릅니다.")
    if prepared is None:
        records = load_manifest(manifest_path, require_other_ids="study_groups" not in base_ckpt)
        manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
        if "study_groups" in base_ckpt:
            digest = study_partition(base_ckpt, records, processed_dir, manifest_hash)
            groups = base_ckpt["study_groups"]
            excluded = base_ckpt["study_excluded"]
        else:
            digest, groups, excluded = prepare_holdout(
                records, processed_dir, source_dir, manifest_hash
            )
    else:
        records, digest, groups, excluded = prepared
    train = sorted(
        set(strict_training_indices(records, sampled_items, groups)).difference(excluded)
    )
    if "study_groups" in base_ckpt:
        train = sorted(set(train).intersection(base_ckpt["training_indices"]))
    folds, _, _ = choose_folds(records, groups, train, GROUP_CV_FOLDS)
    if not folds or any(
        not {MAIN, OTHER}.issubset({records[i]["class_name"] for i in fold["train"]})
        for fold in folds
    ):
        raise ValueError("Strict Holdout 제외 후 내부 검증용 양성·샴고양이 그룹이 부족합니다.")

    model = (
        backbone_model
        if backbone_model is not None
        else load_checkpoint_backbone(base_ckpt, device)
    )
    img_size = int(base_ckpt["image_size"])
    stored = base_ckpt.get("hyperparameters")
    params = (
        Hyperparameters(**stored)
        if stored
        else Hyperparameters(
            main_target_count=base_ckpt.get("main_target_count", 500),
            neg_aug_passes=base_ckpt.get("neg_aug_passes", 1),
            hflip_probability=0.5 if base_ckpt.get("enable_hflip", True) else 0.0,
        )
    )
    cache = EmbeddingCache(
        cache_dir,
        records,
        processed_dir,
        digest,
        base_ckpt["backbone_id"],
        model,
        img_size,
        int(base_ckpt.get("batch_size", BATCH_SIZE)),
        device,
        photo_hashes=base_ckpt.get("study_photo_hashes"),
    )
    policy = base_ckpt.get("threshold_policy", "max_f1")
    targets = tuple(base_ckpt.get("fpr_targets", ()))
    method = base_ckpt["method"]
    signature = fingerprint(
        {
            "cache": cache.key,
            "params": asdict(params),
            "method": method,
            "policy": policy,
            "targets": targets,
            "train": train,
            "folds": folds,
            "original_training": base_ckpt.get("training_indices"),
            "original_run": base_ckpt.get("run_signature"),
            "code": {
                name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "test_random_samples.py",
                    "linear_probe.py",
                    "group_validation.py",
                    "config.py",
                )
            },
        }
    )
    path = cache_dir.parent / "strict_checkpoints" / f"{signature}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("strict_signature") != signature or saved["strict_training_indices"] != train:
            raise ValueError("Strict Holdout 캐시의 학습 범위가 현재 분할과 다릅니다.")
        print(f"[Strict Holdout] 동일 학습 그룹 체크포인트 재사용 · {path}", flush=True)
        return {**saved, "strict_cache_reused": True}
    with threadpool_limits(limits=1):
        scores, _ = fold_scores(cache, train, folds, params, method)
        policies = thresholds_from_validation(
            [records[i]["class_name"] for i in train], scores, targets
        )
        classifier, prototype, counts = fit_probe(cache, train, params)
    tuning_indices = base_ckpt.get("training_indices")
    if base_ckpt.get("dataset_digest") != digest:
        tuning_indices = None
    trained_data_overlap = (
        bool(set(tuning_indices).difference(train)) if tuning_indices is not None else None
    )
    print(f"[Strict Holdout] {len(records) - len(train)}장 제외, {len(train)}장 재학습")
    saved = checkpoint_payload(
        base_ckpt["backbone_key"],
        base_ckpt["backbone_id"],
        img_size,
        method,
        classifier,
        prototype,
        policies[policy]["threshold"],
        {
            **base_ckpt,
            "hyperparameters": asdict(params),
            "threshold_policy": policy,
            "threshold_calibration": "OOF on strict training data only",
            "strict_threshold_policies": policies,
            "strict_fold_indices": folds,
            "strict_training_indices": train,
            "strict_training_counts": counts,
            "strict_signature": signature,
            "strict_cache_reused": False,
            "checkpoint_tuning_overlap": trained_data_overlap,
            "legacy_hyperparameter_defaults_used": stored is None,
        },
    )
    buffer = BytesIO()
    torch.save(saved, buffer)
    write_artifact(path, buffer.getvalue())
    return saved


def save_result_grid(results: List[Dict], output_path: Path) -> None:
    """클래스별 줄에 크롭 이미지, 판별 점수와 정답 여부를 표시한 PNG를 저장합니다."""
    if not results:
        return

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    grouped = defaultdict(list)
    for result in results:
        grouped[result["true_class"]].append(result)

    order = [cls for cls in CLASS_ORDER if cls in grouped]
    order.extend(cls for cls in grouped if cls not in CLASS_ORDER)
    cols = min(10, max(map(len, grouped.values())))
    row_samples = [
        (cls, grouped[cls][start : start + cols])
        for cls in order
        for start in range(0, len(grouped[cls]), cols)
    ]
    rows = len(row_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4 + 1.2, rows * 2.8), squeeze=False)

    for row_axes, (cls, images) in zip(axes, row_samples):
        for ax in row_axes:
            ax.axis("off")

        true_abbr = CLASS_ABBR.get(cls, cls)
        for ax, result in zip(row_axes, images):
            image = result.get("square_img")
            if image is not None:
                ax.imshow(image)

            pred_abbr = "MAIN" if result["pred_is_main"] else "OTHER"
            status = "OK" if result["correct"] else "WRONG"
            color = "#1b7f3a" if result["pred_is_main"] else "#c92a2a"
            ax.set_title(
                f"T:{true_abbr} | P:{pred_abbr}\np={result['probability']:.2f} ({status})",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                pad=4,
            )
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        row_axes[0].set_ylabel(
            ROW_LABELS.get(cls, true_abbr),
            rotation=0,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            labelpad=12,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def manifest_sample_items(records: list[dict], source_dir: Path) -> list[tuple[Path, str, str]]:
    """전처리된 Manifest 행에 대응하는 원본만 샘플 후보로 반환합니다."""
    originals = source_index(source_dir)
    try:
        return [
            (
                originals[source_key(record["source_path"], record["class_name"])],
                record["class_name"],
                record.get("individual_id", record["class_name"]),
            )
            for record in records
        ]
    except KeyError as error:
        raise ValueError("전처리 Manifest에 대응하는 원본 이미지가 없습니다.") from error


def study_sample_records(records: list[dict], manifest_path: Path, checkpoint: dict) -> list[dict]:
    """체크포인트의 복제 격리 목록과 Manifest를 대조해 랜덤 표본 후보를 반환합니다."""
    quarantined = set(checkpoint.get("study_quarantined", ()))
    if not quarantined:
        return records
    if checkpoint["manifest_sha256"] != sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("복제 격리 목록과 현재 Manifest가 다릅니다.")
    return [record for i, record in enumerate(records) if i not in quarantined]


def run_random_sample_test(
    source_dir: Path = SOURCE_DATA_DIR,
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_baseline.pt",
    n_per_class: int = 10,
    seed: Optional[int] = DEFAULT_TEST_SEED,
    strict_holdout: bool = False,
    save_grid: bool = True,
    output_json: Path = OUTPUTS_DIR / "random_sample_test_results.json",
    output_grid: Path = OUTPUTS_DIR / "random_sample_test_grid.png",
    *,
    processed_dir: Path = PROCESSED_DATA_DIR,
    cache_dir: Path = EXPERIMENTS_DIR / "embeddings",
    reuse_processed: bool = False,
    reserved_cohort: bool = False,
    external_cohort: bool = False,
    main_dir: Optional[Path] = None,
) -> Dict:
    """
    클래스당 n_per_class장을 랜덤 추출하여 모델 추론 및 종합 지표를 계산·출력합니다.
    """
    if output_json.exists() or (save_grid and output_grid.exists()):
        raise FileExistsError("이미 존재하는 랜덤 테스트 결과에 새 결과를 저장할 수 없습니다.")
    seed = secrets.randbits(32) if seed is None else seed
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
            "먼저 `python src/train_baseline.py`를 실행하세요."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    manifest_path = processed_dir / "manifest.json"
    records, manifest_hash = [], ""
    has_quarantine = (
        bool(base_ckpt.get("study_quarantined")) and not external_cohort and main_dir is None
    )
    if strict_holdout or reuse_processed or reserved_cohort or has_quarantine:
        records = load_manifest(manifest_path, require_other_ids="study_groups" not in base_ckpt)
        manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()

    prepared, items = None, None
    validated_digest = None
    if (strict_holdout or reserved_cohort) and "study_groups" in base_ckpt:
        validated_digest = study_partition(base_ckpt, records, processed_dir, manifest_hash)
    needs_refit = strict_holdout and not (reserved_cohort and validated_digest)
    if needs_refit:
        if validated_digest is not None:
            digest, groups = validated_digest, base_ckpt["study_groups"]
            excluded = base_ckpt["study_excluded"]
        else:
            digest, groups, excluded = prepare_holdout(
                records, processed_dir, source_dir, manifest_hash
            )
        prepared = records, digest, groups, excluded
    if reserved_cohort:
        held = base_ckpt.get("holdout_indices")
        if not held or base_ckpt.get("manifest_sha256") != manifest_hash:
            raise ValueError("이 체크포인트와 Manifest에 고정 Holdout 목록이 없습니다.")
        items = manifest_sample_items([records[i] for i in held], source_dir)
    elif prepared is not None:
        trainable = base_ckpt["training_indices"] if validated_digest else range(len(records))
        excluded = set(prepared[3])
        eligible = [records[i] for i in trainable if i not in excluded]
        items = group_sample_pool(eligible, prepared[2], source_dir, n_per_class, seed)
    elif reuse_processed or has_quarantine:
        eligible = study_sample_records(records, manifest_path, base_ckpt)
        items = manifest_sample_items(eligible, source_dir)
    sampled_items = sample_per_class(
        source_dir=source_dir, n_per_class=n_per_class, seed=seed, items=items, main_dir=main_dir
    )
    yolo_model = None
    if not reuse_processed:
        from ultralytics import YOLO

        yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
    backbone_model = load_checkpoint_backbone(base_ckpt, device)

    active_ckpt = (
        fit_holdout_checkpoint(
            sampled_items=sampled_items,
            base_ckpt=base_ckpt,
            device=device,
            backbone_model=backbone_model,
            manifest_path=manifest_path,
            processed_dir=processed_dir,
            source_dir=source_dir,
            cache_dir=cache_dir,
            prepared=prepared,
        )
        if needs_refit
        else base_ckpt
    )

    mode_label = (
        "Strict Holdout (테스트 샘플 및 해당 개체 제외 후 재학습)"
        if needs_refit
        else "Strict Holdout (격리된 체크포인트 사용)"
        if strict_holdout
        else "저장된 체크포인트로 추론"
    )
    print("\n" + "=" * 96)
    print(f"  랜덤 샘플 테스트 | 모드: {mode_label}")
    print(
        f"  모델: {active_ckpt['backbone_key']} ({active_ckpt['method']}) | "
        f"임계값: {active_ckpt['threshold']:.4f} | 시드: {seed} | 총 {len(sampled_items)}장"
    )
    print("=" * 96)
    if not strict_holdout and not reserved_cohort and not external_cohort:
        print("[안내] 일반 랜덤 샘플에 기존 학습 이미지가 포함될 수 있습니다.")
    if needs_refit:
        print(
            "[안내] 재학습과 임계값 보정은 테스트 그룹을 제외합니다. "
            "기존 튜닝에 사용된 그룹이면 탐색 단계와의 독립성은 보장되지 않습니다."
        )

    results: List[Dict] = []
    class_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    tp = fp = fn = tn = 0

    header = (
        f"{'No':>2} | {'실제 클래스(개체)':<22} | {'파일명':<28} | "
        f"{'YOLO':<5} | {'예측':<8} | {'확률':>7} | {'유사도':>6} | {'결과'}"
    )
    print(header)
    print("-" * 96)

    sidecar = processed_dir / CROP_INTEGRITY_FILENAME
    integrity = (
        json.loads(sidecar.read_text("utf-8"))["sources"]
        if reuse_processed and sidecar.is_file()
        else None
    )
    for idx, (img_path, true_class, individual_id) in enumerate(sampled_items, start=1):
        record = matching_record(img_path, records, true_class) if reuse_processed else None
        if record is not None and integrity is not None:
            key = source_key(img_path, true_class)
            proof = integrity.get(key)
            if (
                not proof
                or file_sha256(img_path) != proof["raw_sha256"]
                or file_sha256(processed_path(processed_dir, record["processed_rel_path"]))
                != proof["crop_sha256"]
            ):
                raise ValueError("테스트 원본 또는 기존 크롭이 준비한 이미지와 다릅니다.")
        pred = predict_image(
            image_path=img_path,
            checkpoint_path=checkpoint_path,
            ckpt=active_ckpt,
            yolo_model=yolo_model,
            backbone_model=backbone_model,
            device=device,
            return_crop=save_grid,
            processed_record=record,
            processed_dir=processed_dir,
        )

        true_is_main = true_class == "main_siamese_cat"
        pred_is_main = bool(pred["is_main_cat"])
        correct = true_is_main == pred_is_main

        if true_is_main and pred_is_main:
            tp += 1
        elif (not true_is_main) and pred_is_main:
            fp += 1
        elif true_is_main and (not pred_is_main):
            fn += 1
        else:
            tn += 1

        class_stats[true_class]["total"] += 1
        if correct:
            class_stats[true_class]["correct"] += 1

        true_label_disp = (
            f"other({individual_id})" if true_class == "other_siamese_cat" else true_class
        )
        if len(true_label_disp) > 22:
            true_label_disp = true_label_disp[:19] + "..."
        fname = img_path.name
        if len(fname) > 28:
            fname = fname[:25] + "..."
        confidence = pred["yolo_conf"]
        yolo_str = (
            f"{confidence:.2f}"
            if confidence is not None
            else ("탐지" if pred["yolo_detected"] else "미탐지")
        )
        pred_disp = "내 고양이" if pred_is_main else "다른고양"
        ox = "O" if correct else "X"

        print(
            f"{idx:>2} | {true_label_disp:<22} | {fname:<28} | "
            f"{yolo_str:<5} | {pred_disp:<8} | {pred['probability'] * 100:>6.2f}% | "
            f"{pred['cosine_similarity']:>6.4f} | {ox}"
        )

        rec = {
            "index": idx,
            "image_path": str(img_path),
            "true_class": true_class,
            "individual_id": individual_id,
            "true_is_main": true_is_main,
            "pred_is_main": pred_is_main,
            "correct": correct,
            "probability": round(float(pred["probability"]), 4),
            "cosine_similarity": round(float(pred["cosine_similarity"]), 4),
            "score": float(pred["score"]),
            "yolo_detected": bool(pred["yolo_detected"]),
            "yolo_conf": pred["yolo_conf"],
        }
        if save_grid and "square_img" in pred:
            rec["square_img"] = pred["square_img"]
        results.append(rec)

    if save_grid:
        save_result_grid(results, output_grid)

    total = len(results)
    total_correct = tp + tn
    overall_acc = (total_correct / total) if total > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("  클래스별 정확도 요약")
    print("=" * 60)
    for cls_name in CLASS_ORDER:
        if cls_name in class_stats:
            st = class_stats[cls_name]
            acc = st["correct"] / st["total"] * 100.0 if st["total"] > 0 else 0.0
            print(f"  - {cls_name:<20s}: {st['correct']:>2d} / {st['total']:<2d} ({acc:.1f}%)")
    print("-" * 60)
    print(f"  전체 정확도 (Accuracy) : {total_correct} / {total} ({overall_acc * 100:.2f}%)")
    print(f"  Precision / Recall / F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"  Confusion Matrix       : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("=" * 60)

    serializable_results = [{k: v for k, v in r.items() if k != "square_img"} for r in results]
    hard = [row["correct"] for row in results if row["true_class"] in (MAIN, OTHER)]
    metrics = evaluate(
        [r["true_class"] for r in results],
        np.asarray([r["score"] for r in results]),
        float(active_ckpt["threshold"]),
    )
    metrics["hard_siamese_accuracy"] = sum(hard) / len(hard) if hard else None
    summary_payload = {
        "mode": "strict_holdout" if strict_holdout else "full_checkpoint",
        "backbone_key": active_ckpt["backbone_key"],
        "experiment": active_ckpt.get("experiment"),
        "checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "method": active_ckpt["method"],
        "threshold": float(active_ckpt["threshold"]),
        "sample_source": (
            "external_dataset"
            if external_cohort
            else "reserved_group_holdout"
            if reserved_cohort
            else "project_dataset"
        ),
        "refitted": bool(needs_refit and not active_ckpt.get("strict_cache_reused")),
        "strict_cache_reused": bool(needs_refit and active_ckpt.get("strict_cache_reused")),
        "strict_holdout": {
            key: active_ckpt[key]
            for key in (
                "hyperparameters",
                "threshold_policy",
                "threshold_calibration",
                "strict_threshold_policies",
                "strict_fold_indices",
                "strict_training_indices",
                "strict_training_counts",
                "checkpoint_tuning_overlap",
                "legacy_hyperparameter_defaults_used",
            )
        }
        if needs_refit
        else None,
        "seed": seed,
        "test_seed": seed,
        "refit_seed": RANDOM_SEED if needs_refit else None,
        "n_per_class": n_per_class,
        "total_samples": total,
        "class_accuracy": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
            }
            for k, v in class_stats.items()
        },
        "metrics": metrics,
        "samples": serializable_results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"[JSON 결과 저장] {output_json}")
    if save_grid:
        print(f"[그리드 이미지]  {output_grid}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="main, other, etc 각 클래스별 N장 랜덤 추출 모델 테스트 스크립트"
    )
    parser.add_argument("--n-per-class", type=int, default=10, help="클래스별 샘플 수 (기본 10)")
    parser.add_argument("--seed", type=int, help="명시 시 --random과 함께 써도 시드 고정")
    parser.add_argument(
        "--random",
        action="store_true",
        help="지정 시 매 실행마다 생성한 랜덤 시드로 샘플 추출",
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR, help="원본 데이터 폴더")
    parser.add_argument(
        "--main-dir",
        type=Path,
        default=None,
        help="main_siamese_cat 대신 테스트할 외부 폴더 경로 (예: 데이터/main_siamese_cat_test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_baseline.pt",
        help="체크포인트 파일 경로",
    )
    parser.add_argument(
        "--strict-holdout",
        action="store_true",
        help="테스트 샘플과 해당 other 개체를 학습에서 완전 제외한 Holdout 재학습 모델로 평가",
    )
    parser.add_argument("--no-grid", action="store_true", help="시각화 그리드 PNG 저장 생략")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=EXPERIMENTS_DIR / "embeddings")
    parser.add_argument("--reuse-processed", action="store_true", help="기존 YOLO 크롭으로 평가")
    parser.add_argument(
        "--reserved-cohort", action="store_true", help="체크포인트의 고정 Holdout에서만 추출"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "random_sample_tests")
    parser.add_argument("--output-json", type=Path, help="명시적 JSON 결과 경로")
    args = parser.parse_args()

    effective_seed = args.seed if args.seed is not None or args.random else DEFAULT_TEST_SEED
    mode = "strict_holdout" if args.strict_holdout else "random_samples"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = args.output_dir / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_random_sample_test(
        source_dir=args.source,
        checkpoint_path=args.checkpoint,
        n_per_class=args.n_per_class,
        seed=effective_seed,
        strict_holdout=args.strict_holdout,
        save_grid=not args.no_grid,
        output_json=args.output_json or run_dir / f"{mode}.json",
        output_grid=run_dir / f"{mode}_grid.png",
        processed_dir=args.processed,
        cache_dir=args.embedding_cache,
        reuse_processed=args.reuse_processed,
        reserved_cohort=args.reserved_cohort,
        main_dir=args.main_dir,
    )


if __name__ == "__main__":
    main()
