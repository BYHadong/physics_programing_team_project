"""
5-Fold Group CV 기반 사전학습 백본 비교 및 베이스라인 학습 모듈.

비교 모델:
- megadescriptor (BVRA/MegaDescriptor-S-224)
- dinov2 (vit_small_patch14_dinov2.lvd142m)
- convnext (convnext_tiny.fb_in22k_ft_in1k)

비교 방식:
- prototype_cosine: main_siamese_cat 임베딩 평균 프로토타입과의 코사인 유사도
- linear_probe: 온라인 증강 임베딩 기반 class_weight='balanced' 로지스틱 회귀
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from src.config import (
        BACKBONE_INPUT_SIZE,
        BACKBONE_MODELS,
        BATCH_SIZE,
        CHECKPOINTS_DIR,
        GDRIVE_CATEGORIES,
        OUTPUTS_DIR,
        PREPROCESSED_SIZE,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
    )
    from src.dataset import (
        ManifestCatDataset,
        build_transforms,
        create_group_folds,
    )
except ImportError:
    from config import (
        BACKBONE_INPUT_SIZE,
        BACKBONE_MODELS,
        BATCH_SIZE,
        CHECKPOINTS_DIR,
        GDRIVE_CATEGORIES,
        OUTPUTS_DIR,
        PREPROCESSED_SIZE,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
    )
    from dataset import (
        ManifestCatDataset,
        build_transforms,
        create_group_folds,
    )


def set_seed(seed: int = RANDOM_SEED) -> None:
    """실험 재현성을 위해 Python, NumPy, PyTorch 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model_name_or_id: str, device: torch.device) -> Tuple[torch.nn.Module, int]:
    """
    timm 사전학습 백본을 임베딩 추출기(num_classes=0) 형태로 로드하고 가중치를 동결합니다.

    매개변수:
        model_name_or_id: BACKBONE_MODELS 키 또는 timm 모델 식별자
        device: 연산 장치 (cuda / cpu)

    반환값:
        (동결된 eval 모델, 입력 이미지 크기)
    """
    model_id = BACKBONE_MODELS.get(model_name_or_id, model_name_or_id)
    target_size = BACKBONE_INPUT_SIZE

    try:
        model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
    except TypeError:
        model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=0,
        )

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, target_size


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: ManifestCatDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    데이터셋의 모든 샘플에 대해 L2 정규화된 특징 임베딩 행렬 (N, D)을 추출합니다.

    매개변수:
        model: 동결된 백본 모델
        dataset: ManifestCatDataset 인스턴스
        batch_size: 배치 크기
        device: 연산 장치

    반환값:
        L2 정규화된 float32 NumPy 배열 (shape: [N, D])
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: List[np.ndarray] = []
    use_amp = device.type == "cuda"

    for images, _, _, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            feats = model(images)
        feats = F.normalize(feats.float(), p=2, dim=1)
        chunks.append(feats.cpu().numpy())

    return np.vstack(chunks).astype(np.float32)


def extract_augmented_main_embeddings(
    model: torch.nn.Module,
    main_records: List[Dict],
    processed_dir: Path,
    target_count: int,
    img_size: int,
    batch_size: int,
    device: torch.device,
    enable_hflip: bool = True,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    main_siamese_cat 학습 레코드에 클린 + 온라인 증강(RandomResizedCrop, Flip, Rotation, ColorJitter,
    GaussianBlur, JPEG 압축)을 적용해 정확히 target_count(기본 500장)개의 임베딩을 추출합니다.

    매개변수:
        model: 동결된 백본 모델
        main_records: 학습 세트의 main_siamese_cat 레코드 목록
        processed_dir: 전처리 이미지 폴더 경로
        target_count: 목표로 하는 총 양성(main) 학습 임베딩 수
        img_size: 모델 입력 해상도
        batch_size: 배치 크기
        device: 연산 장치
        enable_hflip: 좌우 반전 증강 사용 여부
        seed: 증강 샘플링 재현용 난수 시드

    반환값:
        shape (target_count, D)의 L2 정규화된 float32 임베딩 행렬
    """
    if not main_records:
        raise ValueError("증강할 main_siamese_cat 학습 레코드가 비어 있습니다.")

    classes = sorted(list(GDRIVE_CATEGORIES.keys()))
    eval_tf = build_transforms(image_size=img_size, is_train=False)
    clean_ds = ManifestCatDataset(
        main_records, processed_dir=processed_dir, transform=eval_tf, classes=classes
    )
    clean_embs = extract_embeddings(model, clean_ds, batch_size=batch_size, device=device)

    if target_count <= len(main_records):
        return clean_embs[:target_count]

    need = target_count - len(main_records)
    rng = random.Random(seed)
    aug_records: List[Dict] = []
    while len(aug_records) < need:
        shuffled = list(main_records)
        rng.shuffle(shuffled)
        aug_records.extend(shuffled)
    aug_records = aug_records[:need]

    set_seed(seed)
    train_tf = build_transforms(image_size=img_size, is_train=True, enable_hflip=enable_hflip)
    aug_ds = ManifestCatDataset(
        aug_records, processed_dir=processed_dir, transform=train_tf, classes=classes
    )
    aug_embs = extract_embeddings(model, aug_ds, batch_size=batch_size, device=device)
    return np.vstack([clean_embs, aug_embs]).astype(np.float32)


def compute_evaluation_metrics(
    y_true_bin: np.ndarray,
    y_class_idx: np.ndarray,
    classes: List[str],
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    이진 판별 점수에 대해 전체 ROC-AUC, 하드 샴고양이 ROC-AUC, 최적 F1, FPR을 계산합니다.

    매개변수:
        y_true_bin: 정답 이진 라벨 (main_siamese_cat=1, 나머지=0)
        y_class_idx: 3클래스 정수 라벨
        classes: 정렬된 클래스명 목록
        scores: 모델 예측 점수 (높을수록 내 고양이)

    반환값:
        성능 지표 딕셔너리
    """
    main_idx = classes.index("main_siamese_cat")
    other_idx = classes.index("other_siamese_cat")

    overall_auc = float(roc_auc_score(y_true_bin, scores))

    hard_mask = np.isin(y_class_idx, [main_idx, other_idx])
    hard_auc = float(roc_auc_score(y_true_bin[hard_mask], scores[hard_mask]))

    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, scores)
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = 0.5

    for idx, thr in enumerate(thresholds):
        p = float(precisions[idx])
        r = float(recalls[idx])
        denom = p + r
        f1 = (2.0 * p * r / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_prec = p
            best_rec = r
            best_thr = float(thr)

    other_mask = y_class_idx == other_idx
    unseen_fpr = (
        float(np.mean(scores[other_mask] >= best_thr))
        if np.any(other_mask)
        else 0.0
    )

    return {
        "overall_roc_auc": round(overall_auc, 4),
        "hard_siamese_roc_auc": round(hard_auc, 4),
        "best_f1": round(best_f1, 4),
        "precision": round(best_prec, 4),
        "recall": round(best_rec, 4),
        "threshold": round(best_thr, 4),
        "unseen_siamese_fpr": round(unseen_fpr, 4),
    }


def evaluate_backbone_cv(
    backbone_key: str,
    records: List[Dict],
    folds: List[Tuple[List[Dict], List[Dict]]],
    processed_dir: Path,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    main_target_count: int = 500,
    neg_aug_passes: int = 1,
    enable_hflip: bool = True,
) -> Dict:
    """
    단일 백본에 대해 5-Fold Group CV로 prototype_cosine과 linear_probe 성능을 평가합니다.
    각 Fold의 학습 세트 내부에서만 main_siamese_cat을 main_target_count(기본 500장)까지
    증강 확장하여 검증 데이터 누수를 방지합니다.

    매개변수:
        backbone_key: BACKBONE_MODELS의 모델 키
        records: 전체 manifest 레코드
        folds: create_group_folds()로 생성된 5-Fold 분할
        processed_dir: 전처리 이미지 폴더 경로
        device: 연산 장치
        batch_size: 배치 크기
        main_target_count: Fold별 학습용 main_siamese_cat 증강 목표 수량
        neg_aug_passes: 음성 클래스(other/etc) 추가 증강 패스 횟수
        enable_hflip: 좌우 반전 증강 사용 여부

    반환값:
        백본 및 방식별 OOF/Fold 평균 성능 딕셔너리
    """
    model_id = BACKBONE_MODELS.get(backbone_key, backbone_key)
    print(f"\n[백본 평가 시작] {backbone_key} ({model_id}) | main 증강 목표={main_target_count}장")
    model, img_size = load_backbone(backbone_key, device)

    classes = sorted(list(GDRIVE_CATEGORIES.keys()))
    rel_to_idx = {r["processed_rel_path"]: i for i, r in enumerate(records)}

    y_bin_all = np.array([1 if r["class_name"] == "main_siamese_cat" else 0 for r in records])
    y_cls_all = np.array([classes.index(r["class_name"]) for r in records])

    # 클린 임베딩 및 음성 클래스용 증강 패스 사전 추출
    eval_tf = build_transforms(image_size=img_size, is_train=False)
    train_tf = build_transforms(image_size=img_size, is_train=True, enable_hflip=enable_hflip)

    clean_ds = ManifestCatDataset(records, processed_dir=processed_dir, transform=eval_tf, classes=classes)
    clean_embs = extract_embeddings(model, clean_ds, batch_size=batch_size, device=device)

    neg_aug_list: List[np.ndarray] = []
    if neg_aug_passes > 0:
        aug_ds = ManifestCatDataset(records, processed_dir=processed_dir, transform=train_tf, classes=classes)
        for _ in range(neg_aug_passes):
            neg_aug_list.append(extract_embeddings(model, aug_ds, batch_size=batch_size, device=device))

    oof_proto = np.zeros(len(records), dtype=np.float32)
    oof_linear = np.zeros(len(records), dtype=np.float32)
    fold_hard_proto: List[float] = []
    fold_hard_linear: List[float] = []

    main_idx = classes.index("main_siamese_cat")
    other_idx = classes.index("other_siamese_cat")

    for fold_idx, (tr_recs, val_recs) in enumerate(folds):
        val_indices = np.array([rel_to_idx[r["processed_rel_path"]] for r in val_recs], dtype=np.int64)

        tr_main_recs = [r for r in tr_recs if r["class_name"] == "main_siamese_cat"]
        tr_neg_indices = np.array(
            [rel_to_idx[r["processed_rel_path"]] for r in tr_recs if r["class_name"] != "main_siamese_cat"],
            dtype=np.int64,
        )

        # 학습 세트의 main_siamese_cat만 온라인 증강으로 main_target_count(500장)까지 확장
        tr_main_embs = extract_augmented_main_embeddings(
            model=model,
            main_records=tr_main_recs,
            processed_dir=processed_dir,
            target_count=main_target_count,
            img_size=img_size,
            batch_size=batch_size,
            device=device,
            enable_hflip=enable_hflip,
            seed=RANDOM_SEED + fold_idx,
        )

        # 방식 A: Prototype Cosine Similarity (증강된 500장 평균 임베딩)
        proto_vec = tr_main_embs.mean(axis=0)
        proto_norm = np.linalg.norm(proto_vec)
        if proto_norm > 0:
            proto_vec = proto_vec / proto_norm

        val_embs = clean_embs[val_indices]
        proto_scores = val_embs @ proto_vec
        oof_proto[val_indices] = proto_scores

        # 방식 B: Linear Probe (main 500장 + 음성 클래스 클린/증강 + balanced class weight)
        tr_neg_embs = np.vstack([clean_embs[tr_neg_indices]] + [na[tr_neg_indices] for na in neg_aug_list])
        tr_embs_stacked = np.vstack([tr_main_embs, tr_neg_embs])
        tr_labels_stacked = np.concatenate([
            np.ones(len(tr_main_embs), dtype=np.int64),
            np.zeros(len(tr_neg_embs), dtype=np.int64),
        ])

        clf = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED + fold_idx,
        )
        clf.fit(tr_embs_stacked, tr_labels_stacked)
        linear_probs = clf.predict_proba(val_embs)[:, 1]
        oof_linear[val_indices] = linear_probs

        # Fold별 Hard Siamese ROC-AUC 기록
        val_cls = y_cls_all[val_indices]
        val_bin = y_bin_all[val_indices]
        hard_mask = np.isin(val_cls, [main_idx, other_idx])
        fold_hard_proto.append(float(roc_auc_score(val_bin[hard_mask], proto_scores[hard_mask])))
        fold_hard_linear.append(float(roc_auc_score(val_bin[hard_mask], linear_probs[hard_mask])))

    proto_metrics = compute_evaluation_metrics(y_bin_all, y_cls_all, classes, oof_proto)
    proto_metrics["fold_hard_auc_mean"] = round(float(np.mean(fold_hard_proto)), 4)
    proto_metrics["fold_hard_auc_std"] = round(float(np.std(fold_hard_proto)), 4)

    linear_metrics = compute_evaluation_metrics(y_bin_all, y_cls_all, classes, oof_linear)
    linear_metrics["fold_hard_auc_mean"] = round(float(np.mean(fold_hard_linear)), 4)
    linear_metrics["fold_hard_auc_std"] = round(float(np.std(fold_hard_linear)), 4)

    return {
        "backbone_key": backbone_key,
        "backbone_id": model_id,
        "image_size": img_size,
        "methods": {
            "prototype_cosine": proto_metrics,
            "linear_probe": linear_metrics,
        },
        "_cache": (model, clean_embs, neg_aug_list, y_bin_all, img_size),
    }


def train_and_save_best_model(
    best_entry: Dict,
    best_method: str,
    records: List[Dict],
    processed_dir: Path,
    device: torch.device,
    main_target_count: int = 500,
    enable_hflip: bool = True,
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_baseline.pt",
) -> None:
    """
    가장 우수한 백본/방식을 전체 데이터(main_siamese_cat 500장 증강 포함)로
    재학습해 체크포인트(.pt)로 저장합니다.
    """
    model, clean_embs, neg_aug_list, y_bin_all, img_size = best_entry["_cache"]
    main_records = [r for r in records if r["class_name"] == "main_siamese_cat"]
    neg_mask = y_bin_all == 0

    all_main_embs = extract_augmented_main_embeddings(
        model=model,
        main_records=main_records,
        processed_dir=processed_dir,
        target_count=main_target_count,
        img_size=img_size,
        batch_size=BATCH_SIZE,
        device=device,
        enable_hflip=enable_hflip,
        seed=RANDOM_SEED,
    )
    proto_vec = all_main_embs.mean(axis=0)
    proto_norm = np.linalg.norm(proto_vec)
    if proto_norm > 0:
        proto_vec = proto_vec / proto_norm

    all_neg_embs = np.vstack([clean_embs[neg_mask]] + [na[neg_mask] for na in neg_aug_list])
    all_embs_stacked = np.vstack([all_main_embs, all_neg_embs])
    all_labels_stacked = np.concatenate([
        np.ones(len(all_main_embs), dtype=np.int64),
        np.zeros(len(all_neg_embs), dtype=np.int64),
    ])

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    clf.fit(all_embs_stacked, all_labels_stacked)

    best_metrics = best_entry["methods"][best_method]
    payload = {
        "backbone_key": best_entry["backbone_key"],
        "backbone_id": best_entry["backbone_id"],
        "image_size": img_size,
        "preprocessed_size": PREPROCESSED_SIZE,
        "method": best_method,
        "main_target_count": main_target_count,
        "hyperparameters": {
            "c": 1.0,
            "main_target_count": main_target_count,
            "positive_aug_passes": 7,
            "neg_aug_passes": len(neg_aug_list),
            "rotation_degrees": 15.0,
            "hflip_probability": 0.5 if enable_hflip else 0.0,
        },
        "threshold": float(best_metrics["threshold"]),
        "prototype": proto_vec.astype(np.float32),
        "linear_coef": clf.coef_[0].astype(np.float32),
        "linear_intercept": float(clf.intercept_[0]),
        "metrics": best_metrics,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    print(f"\n[체크포인트 저장 완료] {checkpoint_path} (main={main_target_count}장 확장 적용)")


def run_baseline_comparison(
    model_keys: List[str],
    manifest_path: Path = PROCESSED_DATA_DIR / "manifest.json",
    processed_dir: Path = PROCESSED_DATA_DIR,
    main_target_count: int = 500,
    neg_aug_passes: int = 1,
    enable_hflip: bool = True,
) -> Dict:
    """
    지정된 백본 목록에 대해 5-Fold Group CV 비교를 수행하고 결과를 저장·출력합니다.
    """
    set_seed(RANDOM_SEED)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"전처리 manifest.json이 없습니다: {manifest_path}\n"
            "먼저 `python src/preprocess.py`를 실행하세요."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"[학습 환경] Device: {device} ({gpu_name}) | 총 샘플: {len(records)}장 | "
        f"main 증강 목표: {main_target_count}장 | Seed: {RANDOM_SEED}"
    )

    folds = create_group_folds(records, n_splits=5, seed=RANDOM_SEED)
    print("\n=== 5-Fold Group CV 분할 검증 ===")
    for idx, (tr, va) in enumerate(folds):
        tr_main = sum(1 for r in tr if r["class_name"] == "main_siamese_cat")
        va_main = sum(1 for r in va if r["class_name"] == "main_siamese_cat")
        va_other_ids = sorted({r["individual_id"] for r in va if r["class_name"] == "other_siamese_cat"})
        va_etc = sum(1 for r in va if r["class_name"] == "etc_cats")
        print(
            f"  Fold {idx + 1}: Train={len(tr):4d} (main={tr_main} -> 증강 {main_target_count}장) | "
            f"Val={len(va):3d} (main={va_main}, other_ids={len(va_other_ids)}개, etc={va_etc})"
        )

    evaluated: List[Dict] = []
    for key in model_keys:
        res = evaluate_backbone_cv(
            backbone_key=key,
            records=records,
            folds=folds,
            processed_dir=processed_dir,
            device=device,
            main_target_count=main_target_count,
            neg_aug_passes=neg_aug_passes,
            enable_hflip=enable_hflip,
        )
        evaluated.append(res)

    # 최고 성능 모델 선정 (Hard Siamese ROC-AUC -> Overall ROC-AUC -> Best F1 순)
    best_entry = None
    best_method = ""
    best_rank = (-1.0, -1.0, -1.0)
    for entry in evaluated:
        for method_name, m in entry["methods"].items():
            rank = (m["hard_siamese_roc_auc"], m["overall_roc_auc"], m["best_f1"])
            if rank > best_rank:
                best_rank = rank
                best_entry = entry
                best_method = method_name

    assert best_entry is not None
    train_and_save_best_model(
        best_entry=best_entry,
        best_method=best_method,
        records=records,
        processed_dir=processed_dir,
        device=device,
        main_target_count=main_target_count,
        enable_hflip=enable_hflip,
    )

    # JSON 직렬화를 위해 _cache 제거
    serializable_results = {
        "device": f"{device} ({gpu_name})",
        "total_samples": len(records),
        "main_target_count": main_target_count,
        "neg_aug_passes": neg_aug_passes,
        "best_model": {
            "backbone_key": best_entry["backbone_key"],
            "backbone_id": best_entry["backbone_id"],
            "method": best_method,
            "metrics": best_entry["methods"][best_method],
        },
        "comparisons": [
            {
                "backbone_key": e["backbone_key"],
                "backbone_id": e["backbone_id"],
                "image_size": e["image_size"],
                "methods": e["methods"],
            }
            for e in evaluated
        ],
    }

    out_json = OUTPUTS_DIR / "baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("  5-Fold Group CV 모델 성능 비교표 (기준: 미확인 다른 샴고양이 구분 Hard ROC-AUC)")
    print("=" * 92)
    header = (
        f"{'Backbone':<15} | {'Method':<16} | {'Hard AUC':>8} | "
        f"{'Fold Mean±Std':>15} | {'All AUC':>7} | {'Best F1':>7} | {'Unseen FPR':>10}"
    )
    print(header)
    print("-" * 92)
    for e in evaluated:
        for m_name, m in e["methods"].items():
            mark = " *" if (e["backbone_key"] == best_entry["backbone_key"] and m_name == best_method) else ""
            fold_str = f"{m['fold_hard_auc_mean']:.4f}±{m['fold_hard_auc_std']:.4f}"
            print(
                f"{e['backbone_key']:<15} | {m_name:<16} | {m['hard_siamese_roc_auc']:>8.4f} | "
                f"{fold_str:>15} | {m['overall_roc_auc']:>7.4f} | {m['best_f1']:>7.4f} | "
                f"{m['unseen_siamese_fpr']:>9.2%}{mark}"
            )
    print("=" * 92)
    print(f"[결과 저장] {out_json}")
    return serializable_results


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Fold Group CV 기반 사전학습 백본 베이스라인 비교")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BACKBONE_MODELS.keys()),
        choices=list(BACKBONE_MODELS.keys()),
        help="평가할 백본 키 목록",
    )
    parser.add_argument(
        "--main-target-count",
        type=int,
        default=500,
        help="각 Fold 학습 시 main_siamese_cat 온라인 증강 목표 수량 (기본 500)",
    )
    parser.add_argument(
        "--neg-aug-passes",
        type=int,
        default=1,
        help="음성 클래스(other/etc) 추가 증강 패스 횟수 (기본 1)",
    )
    parser.add_argument("--no-hflip", action="store_true", help="좌우 반전 증강 비활성화")
    args = parser.parse_args()

    run_baseline_comparison(
        model_keys=args.models,
        main_target_count=args.main_target_count,
        neg_aug_passes=args.neg_aug_passes,
        enable_hflip=not args.no_hflip,
    )


if __name__ == "__main__":
    main()
