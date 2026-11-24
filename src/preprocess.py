"""
Stage 1 오프라인 이미지 전처리 파이프라인 모듈.

기능:
1. SOURCE_DATA_DIR 내 3개 클래스 폴더(하위 개체 폴더 포함)를 재귀 탐색합니다.
2. PIL + ImageOps.exif_transpose로 EXIF 회전 보정 및 RGB 3채널 변환을 수행합니다.
   (Windows 한글 경로 호환을 위해 OpenCV imread 대신 PIL을 사용합니다.)
3. YOLOv8n으로 고양이(COCO class 15) 영역을 탐지하여 여백 포함 크롭합니다.
4. 비율 유지 정사각형 패딩(letterbox_square) 후 data/processed/에 저장하고 manifest.json을 생성합니다.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

try:
    from src.config import (
        COCO_CAT_CLASS_ID,
        CROP_MARGIN_RATIO,
        GDRIVE_CATEGORIES,
        PREPROCESSED_SIZE,
        PROCESSED_DATA_DIR,
        SOURCE_DATA_DIR,
        YOLO_CONF_THRESHOLD,
        YOLO_WEIGHTS_PATH,
    )
    from src.dataset import VALID_IMAGE_EXTS, letterbox_square
except ImportError:
    from config import (
        COCO_CAT_CLASS_ID,
        CROP_MARGIN_RATIO,
        GDRIVE_CATEGORIES,
        PREPROCESSED_SIZE,
        PROCESSED_DATA_DIR,
        SOURCE_DATA_DIR,
        YOLO_CONF_THRESHOLD,
        YOLO_WEIGHTS_PATH,
    )
    from dataset import VALID_IMAGE_EXTS, letterbox_square


def load_exif_corrected_rgb(image_path: Path) -> Image.Image:
    """
    이미지 파일을 읽어 EXIF 회전 태그를 보정한 RGB PIL 이미지로 반환합니다.

    매개변수:
        image_path: 이미지 파일 경로

    반환값:
        EXIF 보정된 RGB PIL 이미지
    """
    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def crop_cat_with_yolo(
    pil_img: Image.Image,
    yolo_model,
    conf_threshold: float = YOLO_CONF_THRESHOLD,
    margin_ratio: float = CROP_MARGIN_RATIO,
) -> Tuple[Image.Image, bool, Optional[List[int]], Optional[float]]:
    """
    YOLOv8 모델을 사용해 이미지에서 고양이 영역을 탐지하고 여백을 포함해 크롭합니다.

    매개변수:
        pil_img: EXIF 보정된 RGB PIL 이미지
        yolo_model: 초기화된 Ultralytics YOLO 모델 객체
        conf_threshold: 최소 탐지 신뢰도
        margin_ratio: Bounding Box 상하좌우에 추가할 여백 비율

    반환값:
        (크롭된 이미지, 탐지 성공 여부, [x1, y1, x2, y2], 신뢰도)
    """
    w, h = pil_img.size
    results = yolo_model.predict(
        source=pil_img,
        classes=[COCO_CAT_CLASS_ID],
        conf=conf_threshold,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return pil_img, False, None, None

    boxes = results[0].boxes
    best_idx = 0
    best_score = -1.0
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i].item())
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        score = area * conf
        if score > best_score:
            best_score = score
            best_idx = i

    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    best_conf = float(boxes.conf[best_idx].item())

    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin_ratio
    my = bh * margin_ratio

    cx1 = max(0, int(round(x1 - mx)))
    cy1 = max(0, int(round(y1 - my)))
    cx2 = min(w, int(round(x2 + mx)))
    cy2 = min(h, int(round(y2 + my)))

    if cx2 <= cx1 or cy2 <= cy1:
        return pil_img, False, None, None

    return pil_img.crop((cx1, cy1, cx2, cy2)), True, [cx1, cy1, cx2, cy2], round(best_conf, 4)


def collect_source_files(
    source_dir: Path,
    limit_per_class: Optional[int] = None,
) -> List[Tuple[Path, str, str]]:
    """
    원본 데이터 디렉토리에서 클래스별 이미지 파일과 개체 ID 목록을 수집합니다.

    매개변수:
        source_dir: 클래스 폴더가 포함된 원본 데이터 루트 경로
        limit_per_class: 클래스별 최대 수집 개수 (테스트용)

    반환값:
        (파일 경로, class_name, individual_id) 튜플 리스트
    """
    collected: List[Tuple[Path, str, str]] = []
    for cls_name in sorted(GDRIVE_CATEGORIES.keys()):
        cls_folder = source_dir / cls_name
        if not cls_folder.exists() or not cls_folder.is_dir():
            continue

        cls_items: List[Tuple[Path, str, str]] = []
        for file_path in sorted(cls_folder.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            rel = file_path.relative_to(cls_folder)
            if len(rel.parts) > 1:
                individual_id = rel.parts[0]
            else:
                individual_id = cls_name
            cls_items.append((file_path, cls_name, individual_id))

        if limit_per_class is not None and limit_per_class > 0:
            cls_items = cls_items[:limit_per_class]
        collected.extend(cls_items)
    return collected


def run_preprocessing(
    source_dir: Path = SOURCE_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    force: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    전체 원본 데이터를 YOLO 크롭 및 정사각형 패딩으로 전처리하고 manifest.json을 저장합니다.

    매개변수:
        source_dir: 원본 데이터 경로
        output_dir: 전처리 이미지 및 manifest 저장 경로
        force: True면 기존 전처리 이미지가 있어도 덮어씀
        limit: 클래스당 처리 개수 상한 (테스트용)

    반환값:
        생성된 manifest 레코드 리스트
    """
    from ultralytics import YOLO

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    items = collect_source_files(source_dir, limit_per_class=limit)
    if not items:
        raise FileNotFoundError(f"처리할 이미지가 없습니다: {source_dir}")

    print(f"[전처리 시작] 원본 경로: {source_dir}")
    print(f"[전처리 대상] 총 {len(items)}장 | YOLO 가중치: {YOLO_WEIGHTS_PATH}")

    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))

    existing_by_source: Dict[str, Dict] = {}
    if manifest_path.exists() and not force:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                existing_by_source[rec["source_path"]] = rec

    records: List[Dict] = []
    used_rel_paths: set = set()
    stats = defaultdict(lambda: {"total": 0, "detected": 0})

    for src_path, cls_name, individual_id in tqdm(items, desc="YOLO 크롭 및 전처리"):
        src_str = str(src_path)
        if not force and src_str in existing_by_source:
            cached = existing_by_source[src_str]
            if (output_dir / cached["processed_rel_path"]).exists():
                records.append(cached)
                used_rel_paths.add(cached["processed_rel_path"])
                stats[cls_name]["total"] += 1
                if cached.get("yolo_detected"):
                    stats[cls_name]["detected"] += 1
                continue

        pil_img = load_exif_corrected_rgb(src_path)
        orig_w, orig_h = pil_img.size

        cropped, detected, bbox, conf = crop_cat_with_yolo(pil_img, yolo_model)
        square_img = letterbox_square(cropped, PREPROCESSED_SIZE)

        # 파일명 충돌 방지를 위한 고유 출력 경로 결정
        rel_dir = Path(cls_name) / individual_id
        candidate = rel_dir / f"{src_path.stem}.jpg"
        suffix_idx = 1
        while candidate.as_posix() in used_rel_paths:
            candidate = rel_dir / f"{src_path.stem}_{suffix_idx}.jpg"
            suffix_idx += 1

        used_rel_paths.add(candidate.as_posix())
        out_path = output_dir / candidate
        out_path.parent.mkdir(parents=True, exist_ok=True)
        square_img.save(out_path, format="JPEG", quality=95)

        rec = {
            "source_path": src_str,
            "processed_rel_path": candidate.as_posix(),
            "class_name": cls_name,
            "individual_id": individual_id,
            "orig_size": [orig_w, orig_h],
            "processed_size": [PREPROCESSED_SIZE, PREPROCESSED_SIZE],
            "yolo_detected": detected,
            "yolo_bbox": bbox,
            "yolo_conf": conf,
        }
        records.append(rec)
        stats[cls_name]["total"] += 1
        if detected:
            stats[cls_name]["detected"] += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n=== Stage 1 전처리 결과 요약 ===")
    total_all = 0
    detected_all = 0
    for cls_name in sorted(stats.keys()):
        t = stats[cls_name]["total"]
        d = stats[cls_name]["detected"]
        total_all += t
        detected_all += d
        rate = (d / t * 100.0) if t > 0 else 0.0
        print(f"  - {cls_name:20s}: {t:4d}장 처리 | YOLO 탐지 성공 {d:4d}장 ({rate:.1f}%)")
    overall_rate = (detected_all / total_all * 100.0) if total_all > 0 else 0.0
    print(f"  - {'전체 합계':20s}: {total_all:4d}장 처리 | YOLO 탐지 성공 {detected_all:4d}장 ({overall_rate:.1f}%)")
    print(f"[저장 완료] {manifest_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 YOLO 고양이 크롭 및 정사각형 패딩 전처리")
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR, help="원본 데이터 폴더 경로")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR, help="전처리 출력 경로")
    parser.add_argument("--force", action="store_true", help="기존 전처리 결과 무시하고 재실행")
    parser.add_argument("--limit", type=int, default=None, help="클래스당 처리할 최대 이미지 수")
    args = parser.parse_args()

    run_preprocessing(
        source_dir=args.source,
        output_dir=args.output,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
