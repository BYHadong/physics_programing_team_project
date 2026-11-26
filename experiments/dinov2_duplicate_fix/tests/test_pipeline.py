"""
전처리 유틸리티, 증강 변환, GroupKFold 데이터 누수 방지 분할 단위 테스트
"""

from PIL import Image
import torch

from src.dataset import (
    RandomJPEGCompression,
    build_transforms,
    create_group_folds,
    letterbox_square,
)
from src.test_random_samples import sample_per_class


def test_letterbox_square_preserves_size_and_mode():
    """letterbox_square가 비율 왜곡 없이 지정된 정사각형 RGB 이미지를 반환하는지 검증합니다."""
    rect_img = Image.new("RGBA", (400, 200), (255, 0, 0, 255))
    out = letterbox_square(rect_img, target_size=512)
    assert out.size == (512, 512)
    assert out.mode == "RGB"
    # 중앙 행은 원본 빨간색 영역, 상단 여백은 기본 fill_color여야 함
    assert out.getpixel((256, 256)) == (255, 0, 0)
    assert out.getpixel((10, 10)) == (124, 116, 104)


def test_transforms_and_random_jpeg_compression():
    """RandomJPEGCompression 및 Train/Val Transform 출력 텐서 형태를 검증합니다."""
    sample = Image.new("RGB", (320, 320), (120, 150, 180))
    compressor = RandomJPEGCompression(min_quality=50, max_quality=95, p=1.0)
    compressed = compressor(sample)
    assert compressed.size == (320, 320)
    assert compressed.mode == "RGB"

    train_tf = build_transforms(image_size=224, is_train=True, enable_hflip=True)
    val_tf = build_transforms(image_size=224, is_train=False)

    train_tensor = train_tf(sample)
    val_tensor = val_tf(sample)
    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert val_tensor.shape == (3, 224, 224)


def test_create_group_folds_no_leakage():
    """5-Fold 분할에서 파일 중복 및 other_siamese_cat 개체 ID 누수가 없는지 검증합니다."""
    records = []
    for i in range(15):
        records.append({
            "processed_rel_path": f"main_siamese_cat/main/m_{i}.jpg",
            "class_name": "main_siamese_cat",
            "individual_id": "main_siamese_cat",
        })
    for cat_num in range(1, 11):
        for j in range(3):
            records.append({
                "processed_rel_path": f"other_siamese_cat/{cat_num}번 고양이/o_{cat_num}_{j}.jpg",
                "class_name": "other_siamese_cat",
                "individual_id": f"{cat_num}번 고양이",
            })
    for i in range(25):
        records.append({
            "processed_rel_path": f"etc_cats/etc/e_{i}.jpg",
            "class_name": "etc_cats",
            "individual_id": "etc_cats",
        })

    folds = create_group_folds(records, n_splits=5, seed=42)
    assert len(folds) == 5

    seen_val_paths = set()
    for tr_recs, va_recs in folds:
        tr_paths = {r["processed_rel_path"] for r in tr_recs}
        va_paths = {r["processed_rel_path"] for r in va_recs}
        assert len(tr_paths & va_paths) == 0
        assert len(tr_paths) + len(va_paths) == len(records)
        assert len(seen_val_paths & va_paths) == 0
        seen_val_paths.update(va_paths)

        tr_other_ids = {
            r["individual_id"] for r in tr_recs if r["class_name"] == "other_siamese_cat"
        }
        va_other_ids = {
            r["individual_id"] for r in va_recs if r["class_name"] == "other_siamese_cat"
        }
        assert len(tr_other_ids) > 0
        assert len(va_other_ids) > 0
        assert len(tr_other_ids & va_other_ids) == 0

    assert len(seen_val_paths) == len(records)


def test_sample_per_class_counts_and_reproducibility(tmp_path):
    """sample_per_class가 클래스별 N장을 중복 없이 추출하고 시드 재현성을 보장하는지 검증합니다."""
    main_dir = tmp_path / "main_siamese_cat"
    other_dir = tmp_path / "other_siamese_cat"
    etc_dir = tmp_path / "etc_cats"
    main_dir.mkdir(parents=True)
    etc_dir.mkdir(parents=True)

    for i in range(15):
        (main_dir / f"main_{i:02d}.jpg").write_bytes(b"fake")
        (etc_dir / f"etc_{i:02d}.jpg").write_bytes(b"fake")

    for cat_num in range(1, 6):
        sub = other_dir / f"{cat_num}번 고양이"
        sub.mkdir(parents=True)
        for j in range(3):
            (sub / f"o_{cat_num}_{j}.png").write_bytes(b"fake")

    run1 = sample_per_class(source_dir=tmp_path, n_per_class=10, seed=42)
    run2 = sample_per_class(source_dir=tmp_path, n_per_class=10, seed=42)
    run3 = sample_per_class(source_dir=tmp_path, n_per_class=10, seed=99)

    assert len(run1) == 30
    assert run1 == run2
    assert run1 != run3

    paths = [p for p, _, _ in run1]
    assert len(set(paths)) == 30

    counts = {}
    for _, cls_name, _ in run1:
        counts[cls_name] = counts.get(cls_name, 0) + 1
    assert counts == {
        "main_siamese_cat": 10,
        "other_siamese_cat": 10,
        "etc_cats": 10,
    }
