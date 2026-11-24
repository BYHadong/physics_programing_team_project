"""
단일 이미지에 대한 '내 고양이' vs '다른 고양이' 추론 CLI 모듈
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

try:
    from src.config import CHECKPOINTS_DIR, YOLO_WEIGHTS_PATH
    from src.dataset import build_transforms, letterbox_square
    from src.preprocess import crop_cat_with_yolo, load_exif_corrected_rgb
    from src.train_baseline import load_backbone
except ImportError:
    from config import CHECKPOINTS_DIR, YOLO_WEIGHTS_PATH
    from dataset import build_transforms, letterbox_square
    from preprocess import crop_cat_with_yolo, load_exif_corrected_rgb
    from train_baseline import load_backbone


def predict_image(
    image_path: Path,
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_baseline.pt",
    ckpt: Optional[Dict] = None,
    yolo_model=None,
    backbone_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    return_crop: bool = False,
) -> Dict:
    """
    단일 이미지 파일에 대해 전처리, YOLO 크롭 및 학습된 베이스라인 모델로 판별을 수행합니다.

    매개변수:
        image_path: 판별할 고양이 이미지 파일 경로
        checkpoint_path: 저장된 best_baseline.pt 체크포인트 경로
        ckpt: 사전 로드된 체크포인트 딕셔너리 (배치 추론 재사용용)
        yolo_model: 사전 로드된 YOLO 모델 객체 (배치 추론 재사용용)
        backbone_model: 사전 로드된 백본 모델 (배치 추론 재사용용)
        device: 연산 장치 (미지정 시 자동 선택)
        return_crop: True면 반환 딕셔너리에 크롭·패딩된 PIL 이미지(square_img)를 포함

    반환값:
        판별 결과, 확률, 코사인 유사도, 사용 임계값 등 메타데이터 딕셔너리
    """
    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"모델 체크포인트가 없습니다: {checkpoint_path}\n"
                "먼저 `python src/train_baseline.py`를 실행하여 모델을 학습하세요."
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    backbone_id = ckpt["backbone_id"]
    image_size = int(ckpt["image_size"])
    preprocessed_size = int(ckpt.get("preprocessed_size", 512))
    method = ckpt["method"]
    threshold = float(ckpt["threshold"])

    # 1. EXIF 보정 + YOLO 고양이 크롭 + 정사각형 패딩
    pil_img = load_exif_corrected_rgb(image_path)
    if yolo_model is None:
        yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
    cropped, detected, bbox, det_conf = crop_cat_with_yolo(pil_img, yolo_model)
    square_img = letterbox_square(cropped, preprocessed_size)

    # 2. 백본 임베딩 추출
    eval_transform = build_transforms(image_size=image_size, is_train=False)
    tensor_img = eval_transform(square_img).unsqueeze(0).to(device)

    model = backbone_model if backbone_model is not None else load_backbone(backbone_id, device)[0]
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            feats = model(tensor_img)
        feats = F.normalize(feats.float(), p=2, dim=1)
        emb = feats.cpu().numpy()[0]

    # 3. 코사인 유사도 및 선형 분류기 확률 계산
    prototype = np.asarray(ckpt["prototype"], dtype=np.float32)
    cosine_sim = float(np.dot(emb, prototype))

    coef = np.asarray(ckpt["linear_coef"], dtype=np.float32)
    intercept = float(ckpt["linear_intercept"])
    logit = float(np.dot(emb, coef) + intercept)
    prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))

    score = prob if method == "linear_probe" else cosine_sim
    is_main_cat = score >= threshold
    label_str = "내 고양이 (main_siamese_cat)" if is_main_cat else "다른 고양이 (other / etc)"

    result = {
        "image_path": str(image_path),
        "backbone_key": ckpt["backbone_key"],
        "method": method,
        "prediction": label_str,
        "is_main_cat": is_main_cat,
        "score": score,
        "threshold": threshold,
        "probability": prob,
        "cosine_similarity": cosine_sim,
        "yolo_detected": detected,
        "yolo_conf": det_conf,
        "yolo_bbox": bbox,
    }
    if return_crop:
        result["square_img"] = square_img
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 고양이 이미지 판별 추론 CLI")
    parser.add_argument("image_path", type=Path, help="판별할 이미지 파일 경로")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_baseline.pt",
        help="체크포인트 파일 경로",
    )
    args = parser.parse_args()

    res = predict_image(args.image_path, args.checkpoint)
    print("=" * 60)
    print(f"[입력 파일]     {res['image_path']}")
    print(f"[사용 모델]     {res['backbone_key']} ({res['method']})")
    print(f"[YOLO 탐지]     {'성공' if res['yolo_detected'] else '미탐지(원본 사용)'} "
          f"(conf={res['yolo_conf']})")
    print("-" * 60)
    print(f"[판정 결과]     {res['prediction']}")
    print(f"[판정 점수]     {res['score']:.4f} (기준 임계값: {res['threshold']:.4f})")
    print(f"[내 고양이 확률] {res['probability'] * 100:.2f}%")
    print(f"[코사인 유사도] {res['cosine_similarity']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python src/predict.py <이미지_경로>")
        sys.exit(1)
    main()
