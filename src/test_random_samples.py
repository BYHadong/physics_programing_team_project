"""
main_siamese_cat, other_siamese_cat, etc_cats 각 클래스에서 N장(기본 10장)씩
랜덤으로 추출하여 모델 판별 성능을 종합 테스트하는 CLI 스크립트.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from ultralytics import YOLO

try:
    from src.config import (
        BATCH_SIZE,
        CHECKPOINTS_DIR,
        GDRIVE_CATEGORIES,
        OUTPUTS_DIR,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
        SOURCE_DATA_DIR,
        YOLO_WEIGHTS_PATH,
    )
    from src.dataset import ManifestCatDataset, build_transforms
    from src.predict import predict_image
    from src.preprocess import collect_source_files
    from src.train_baseline import extract_embeddings, load_backbone, set_seed
except ImportError:
    from config import (
        BATCH_SIZE,
        CHECKPOINTS_DIR,
        GDRIVE_CATEGORIES,
        OUTPUTS_DIR,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
        SOURCE_DATA_DIR,
        YOLO_WEIGHTS_PATH,
    )
    from dataset import ManifestCatDataset, build_transforms
    from predict import predict_image
    from preprocess import collect_source_files
    from train_baseline import extract_embeddings, load_backbone, set_seed


CLASS_ORDER = ["main_siamese_cat", "other_siamese_cat", "etc_cats"]
CLASS_ABBR = {
    "main_siamese_cat": "MAIN",
    "other_siamese_cat": "OTHER",
    "etc_cats": "ETC",
}


def sample_per_class(
    source_dir: Path = SOURCE_DATA_DIR,
    n_per_class: int = 10,
    seed: Optional[int] = RANDOM_SEED,
) -> List[Tuple[Path, str, str]]:
    """
    원본 데이터 디렉토리에서 클래스별로 n_per_class장씩 중복 없이 랜덤 추출합니다.

    매개변수:
        source_dir: 클래스 폴더가 포함된 원본 데이터 루트 경로
        n_per_class: 클래스당 추출할 이미지 장수
        seed: 난수 시드 (None이면 매 실행마다 다른 랜덤 추출)

    반환값:
        (파일 경로, class_name, individual_id) 튜플 리스트
    """
    all_items = collect_source_files(Path(source_dir))
    if not all_items:
        raise FileNotFoundError(f"샘플링할 이미지를 찾을 수 없습니다: {source_dir}")

    grouped: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)
    for item in all_items:
        grouped[item[1]].append(item)

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


def fit_holdout_checkpoint(
    sampled_items: List[Tuple[Path, str, str]],
    base_ckpt: Dict,
    device: torch.device,
    backbone_model: Optional[torch.nn.Module] = None,
    manifest_path: Path = PROCESSED_DATA_DIR / "manifest.json",
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Dict:
    """
    테스트 샘플(및 other_siamese_cat의 해당 개체 전체)을 학습 데이터에서 완전히 제외한 뒤,
    남은 데이터만으로 임베딩 프로토타입과 선형 분류기를 즉석 재학습합니다.

    매개변수:
        sampled_items: 이번 테스트에 뽑힌 샘플 목록
        base_ckpt: 백본 정보와 기본 임계값을 참조할 체크포인트 딕셔너리
        device: 연산 장치
        backbone_model: 사전 로드된 동결 백본 모델
        manifest_path: 전처리 manifest.json 경로
        processed_dir: 전처리 이미지 루트 경로

    반환값:
        Holdout 데이터로 재학습된 임시 체크포인트 딕셔너리
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"--strict-holdout 모드에는 전처리 manifest.json이 필요합니다: {manifest_path}\n"
            "먼저 `python src/preprocess.py`를 실행하세요."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    sampled_paths = {str(Path(p).resolve()) for p, _, _ in sampled_items}
    excluded_other_ids = {
        ind_id for _, cls_name, ind_id in sampled_items if cls_name == "other_siamese_cat"
    }

    train_records = [
        rec for rec in records
        if str(Path(rec["source_path"]).resolve()) not in sampled_paths
        and not (
            rec["class_name"] == "other_siamese_cat"
            and rec.get("individual_id") in excluded_other_ids
        )
    ]

    if not any(r["class_name"] == "main_siamese_cat" for r in train_records):
        raise ValueError("Holdout 제외 후 남은 main_siamese_cat 학습 샘플이 없습니다.")

    print(
        f"[Strict Holdout 재학습] 전체 {len(records)}장 중 테스트 관련 샘플 제외 -> "
        f"순수 학습 {len(train_records)}장 사용 (제외된 other 개체: {len(excluded_other_ids)}개)"
    )

    model = (
        backbone_model
        if backbone_model is not None
        else load_backbone(base_ckpt["backbone_id"], device)[0]
    )
    img_size = int(base_ckpt["image_size"])
    classes = sorted(list(GDRIVE_CATEGORIES.keys()))

    eval_tf = build_transforms(image_size=img_size, is_train=False)
    train_tf = build_transforms(image_size=img_size, is_train=True, enable_hflip=True)

    clean_ds = ManifestCatDataset(
        train_records, processed_dir=processed_dir, transform=eval_tf, classes=classes
    )
    aug_ds = ManifestCatDataset(
        train_records, processed_dir=processed_dir, transform=train_tf, classes=classes
    )

    embs_list = [extract_embeddings(model, clean_ds, BATCH_SIZE, device)]
    for _ in range(3):
        embs_list.append(extract_embeddings(model, aug_ds, BATCH_SIZE, device))

    y_bin = np.array([1 if r["class_name"] == "main_siamese_cat" else 0 for r in train_records])
    main_mask = y_bin == 1

    proto_vec = np.vstack([e[main_mask] for e in embs_list]).mean(axis=0)
    proto_norm = np.linalg.norm(proto_vec)
    if proto_norm > 0:
        proto_vec = proto_vec / proto_norm

    all_embs = np.vstack(embs_list)
    all_labels = np.tile(y_bin, len(embs_list))

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    clf.fit(all_embs, all_labels)

    holdout_ckpt = dict(base_ckpt)
    holdout_ckpt["prototype"] = proto_vec.astype(np.float32)
    holdout_ckpt["linear_coef"] = clf.coef_[0].astype(np.float32)
    holdout_ckpt["linear_intercept"] = float(clf.intercept_[0])
    return holdout_ckpt


def save_result_grid(results: List[Dict], output_path: Path) -> None:
    """
    추론 결과와 크롭된 이미지를 정돈된 그리드 PNG 파일로 저장합니다.
    """
    if not results:
        return

    n_total = len(results)
    # 클래스당 10장씩 3클래스(30장)일 경우 깔끔하게 3행 x 10열로 배치
    class_counts = defaultdict(int)
    for r in results:
        class_counts[r["true_class"]] += 1

    if len(class_counts) == 3 and len(set(class_counts.values())) == 1 and n_total <= 36:
        cols = next(iter(class_counts.values()))
        rows = 3
    else:
        cols = min(10, n_total)
        rows = math.ceil(n_total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for idx, res in enumerate(results):
        ax = axes_flat[idx]
        img = res.get("square_img")
        if img is not None:
            ax.imshow(img)

        true_abbr = CLASS_ABBR.get(res["true_class"], res["true_class"])
        pred_abbr = "MAIN" if res["pred_is_main"] else "OTHER"
        status = "OK" if res["correct"] else "WRONG"
        color = "#1b7f3a" if res["correct"] else "#c92a2a"

        ax.set_title(
            f"T:{true_abbr} | P:{pred_abbr}\np={res['probability']:.2f} ({status})",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            pad=4,
        )
        # 테두리를 다시 켜서 정답/오답 색상 프레임 표시
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_random_sample_test(
    source_dir: Path = SOURCE_DATA_DIR,
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_baseline.pt",
    n_per_class: int = 10,
    seed: Optional[int] = RANDOM_SEED,
    strict_holdout: bool = False,
    save_grid: bool = True,
    output_json: Path = OUTPUTS_DIR / "random_sample_test_results.json",
    output_grid: Path = OUTPUTS_DIR / "random_sample_test_grid.png",
) -> Dict:
    """
    클래스당 n_per_class장을 랜덤 추출하여 모델 추론 및 종합 지표를 계산·출력합니다.
    """
    if seed is not None:
        set_seed(seed)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
            "먼저 `python src/train_baseline.py`를 실행하세요."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sampled_items = sample_per_class(source_dir=source_dir, n_per_class=n_per_class, seed=seed)

    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
    backbone_model, _ = load_backbone(base_ckpt["backbone_id"], device)

    active_ckpt = (
        fit_holdout_checkpoint(
            sampled_items=sampled_items,
            base_ckpt=base_ckpt,
            device=device,
            backbone_model=backbone_model,
        )
        if strict_holdout
        else base_ckpt
    )

    mode_label = (
        "Strict Holdout (테스트 샘플 및 해당 개체 제외 후 재학습)"
        if strict_holdout
        else "기본 체크포인트 (best_baseline.pt 전체 학습 모델)"
    )
    print("\n" + "=" * 96)
    print(f"  랜덤 샘플 테스트 | 모드: {mode_label}")
    print(
        f"  모델: {active_ckpt['backbone_key']} ({active_ckpt['method']}) | "
        f"임계값: {active_ckpt['threshold']:.4f} | 시드: {seed} | 총 {len(sampled_items)}장"
    )
    print("=" * 96)
    if not strict_holdout:
        print(
            "[안내] best_baseline.pt는 전체 데이터로 학습된 모델입니다. "
            "완전 미학습 기준 평가를 원하시면 `--strict-holdout` 옵션을 추가해 실행하세요."
        )
        print("-" * 96)

    results: List[Dict] = []
    class_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    tp = fp = fn = tn = 0

    header = (
        f"{'No':>2} | {'실제 클래스(개체)':<22} | {'파일명':<28} | "
        f"{'YOLO':<5} | {'예측':<8} | {'확률':>7} | {'유사도':>6} | {'결과'}"
    )
    print(header)
    print("-" * 96)

    for idx, (img_path, true_class, individual_id) in enumerate(sampled_items, start=1):
        pred = predict_image(
            image_path=img_path,
            checkpoint_path=checkpoint_path,
            ckpt=active_ckpt,
            yolo_model=yolo_model,
            backbone_model=backbone_model,
            device=device,
            return_crop=save_grid,
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
        yolo_str = f"{pred['yolo_conf']:.2f}" if pred["yolo_detected"] else "미탐지"
        pred_disp = "내 고양이" if pred_is_main else "다른고양"
        ox = "O" if correct else "X"

        print(
            f"{idx:>2} | {true_label_disp:<22} | {fname:<28} | "
            f"{yolo_str:<5} | {pred_disp:<8} | {pred['probability']*100:>6.2f}% | "
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
            "score": round(float(pred["score"]), 4),
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

    serializable_results = [
        {k: v for k, v in r.items() if k != "square_img"} for r in results
    ]
    summary_payload = {
        "mode": "strict_holdout" if strict_holdout else "full_checkpoint",
        "backbone_key": active_ckpt["backbone_key"],
        "method": active_ckpt["method"],
        "threshold": float(active_ckpt["threshold"]),
        "seed": seed,
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
        "metrics": {
            "accuracy": round(overall_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "samples": serializable_results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
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
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="랜덤 추출 시드 (기본 42)")
    parser.add_argument(
        "--random",
        action="store_true",
        help="지정 시 매 실행마다 다른 랜덤 시드(None)로 샘플 추출",
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR, help="원본 데이터 폴더")
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUTS_DIR / "random_sample_test_results.json",
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    effective_seed = None if args.random else args.seed
    run_random_sample_test(
        source_dir=args.source,
        checkpoint_path=args.checkpoint,
        n_per_class=args.n_per_class,
        seed=effective_seed,
        strict_holdout=args.strict_holdout,
        save_grid=not args.no_grid,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
