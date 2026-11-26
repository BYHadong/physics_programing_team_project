"""추가 원본 전처리와 백업 보존을 소규모 합성 데이터로 검증합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from src import preprocess
from src.experiment_splits import MAIN, OTHER, load_manifest, source_key
from src.prepare_dataset import prepare_dataset


def test_preprocess_new_image_and_keep_existing_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """경로가 바뀐 기존 크롭과 동명 신규 원본을 구분하고 완성본을 재사용합니다."""
    source = tmp_path / "raw"
    output = tmp_path / "processed_1121"
    base_archive = tmp_path / "drive" / "processed.zip"
    target_archive = base_archive.with_name("processed_1121.zip")
    base_archive.parent.mkdir()
    entries = [
        (f"{MAIN}/same.jpg", MAIN, MAIN),
        (f"{OTHER}/cat-1/other.jpg", OTHER, "cat-1"),
        ("etc_cats/etc.jpg", "etc_cats", "etc_cats"),
    ]
    records = []
    for name, cls, individual in entries:
        original = source / name
        original.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12)).save(original)
        records.append(
            {
                "source_path": f"D:/previous_mount/{name}",
                "processed_rel_path": f"{cls}/{individual}/{original.stem}.jpg",
                "class_name": cls,
                "individual_id": individual,
            }
        )
    with ZipFile(base_archive, "w") as archive:
        for row in records:
            archive.writestr(f"processed/{row['processed_rel_path']}", b"existing crop bytes")
        archive.writestr("processed/manifest.json", json.dumps(records))
    original_backup = base_archive.read_bytes()

    added = source / MAIN / "same.bmp"
    Image.new("RGB", (20, 24), color=(30, 120, 210)).save(added)

    crops = []

    def crop(image: Image.Image, model: object) -> tuple[Image.Image, bool, None, None]:
        """크롭 호출과 신규 이미지 크기를 기록합니다."""
        crops.append(image.size)
        return image, False, None, None

    monkeypatch.setattr(preprocess, "crop_cat_with_yolo", crop)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _p: object()))
    base_counts = {MAIN: 1, OTHER: 1, "etc_cats": 1}
    target_counts = {**base_counts, MAIN: 2}
    kwargs = {"base_counts": base_counts, "target_counts": target_counts, "expected_other_count": 1}

    _, saved_archive = prepare_dataset(source, output, target_archive, [base_archive], **kwargs)
    assert crops == [(20, 24)]
    assert saved_archive == target_archive
    assert base_archive.read_bytes() == original_backup
    for row in records:
        assert (output / row["processed_rel_path"]).read_bytes() == b"existing crop bytes"

    prepared = load_manifest(output / "manifest.json")
    by_source = {source_key(row["source_path"], row["class_name"]): row for row in prepared}
    new_row = by_source[f"{MAIN}/same.bmp"]
    assert new_row["processed_rel_path"] == f"{MAIN}/{MAIN}/same_1.jpg"
    assert len(prepared) == 4
    assert added.is_file()

    backup_bytes = target_archive.read_bytes()
    prepare_dataset(source, output, target_archive, [base_archive], **kwargs)
    restored = tmp_path / "fresh_runtime"
    prepare_dataset(source, restored, target_archive, [base_archive], **kwargs)
    assert crops == [(20, 24)]
    assert (restored / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()
    assert target_archive.read_bytes() == backup_bytes
