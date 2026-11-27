"""최종 실행의 실제 보관 형식과 캐시·전처리 계약을 검증합니다."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image

from src import study_data
from src.embedding_cache import EmbeddingCache, fingerprint
from src.experiment_splits import MAIN, OTHER
from src.linear_probe import Hyperparameters, thresholds_from_validation
from src.study_history import restore_legacy
from src.train_e15 import E15
from src.tune_hyperparameters import write_csv

ETC = "etc_cats"


def test_saved_legacy_archive_rejects_missing_or_conflicting_conditions(tmp_path: Path) -> None:
    """실제 JSON·CSV 계약에서 누락 번호나 불일치 설정을 찾아 재평가를 차단합니다."""
    values = [
        Hyperparameters()
        if i == 0
        else E15
        if i == 15
        else replace(E15, c=i * 0.02)
        if i < 20
        else Hyperparameters()
        for i in range(24)
    ]
    rows = [{"experiment": f"E{i:02}", "parameters": asdict(p)} for i, p in enumerate(values)]
    history = tmp_path / "hyperparameter_results.json"
    history.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="E23"):
        restore_legacy(tmp_path)
    history.write_text(json.dumps(rows))
    restored, _ = restore_legacy(tmp_path)
    assert list(restored.values()) == values and len(set(restored.values())) == 20
    other = [{"experiment": r["experiment"], **r["parameters"]} for r in rows]
    other[0]["c"] = 0.09
    write_csv(tmp_path / "hyperparameter_results.csv", other)
    with pytest.raises(ValueError, match="E00 설정"):
        restore_legacy(tmp_path)


def test_fpr_targets_can_require_zero_recall() -> None:
    """1% FPR에서 유효한 양성 검출이 불가능한 경우를 따로 기록합니다."""
    classes = [OTHER] * 25 + [MAIN]
    outcomes = thresholds_from_validation(classes, np.linspace(1, 0, 26), (0.01, 0.10))
    result = outcomes["fpr_le_0.01"]
    assert not result["target_supports_positive_recall"] and result["fpr_resolution"] == 0.04


def test_1_121_archive_and_partial_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """정상 1,121장 복원과 손상 한 장만 골라 처리하는 흐름을 검증합니다."""
    raw, archive = tmp_path / "raw", tmp_path / "processed_1121.zip"
    counts = {MAIN: 87, OTHER: 572, ETC: 462}
    stream = BytesIO()
    Image.new("RGB", (512, 512), (20, 40, 60)).save(stream, "JPEG")
    crop = stream.getvalue()
    records = []
    for cls, size in counts.items():
        for i in range(size):
            rel = Path(cls) / ("id_01" if cls == OTHER else cls) / f"{i}.jpg"
            path = raw / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(crop)
            records.append(
                {
                    "class_name": cls,
                    "individual_id": rel.parts[1],
                    "processed_rel_path": rel.as_posix(),
                    "source_path": str(path),
                }
            )
    manifest = json.dumps(records).encode()

    def archive_of(path: Path, broken: str | None = None) -> None:
        """같은 목록에서 지정한 크롭 파일만 제외한 ZIP을 만듭니다."""
        with ZipFile(path, "w") as zip_file:
            zip_file.writestr("processed/manifest.json", manifest)
            for record in records:
                relative = record["processed_rel_path"]
                if relative != broken:
                    zip_file.writestr("processed/" + relative, crop)

    def repair(source_dir: Path, output_dir: Path) -> list[dict]:
        """전처리 재호출 시 누락된 한 장만 넘겨받았는지 검증합니다."""
        assert source_dir == raw
        current = json.loads((output_dir / "manifest.json").read_bytes())
        assert len(current) == 1120
        missing = records[10]
        target = output_dir / missing["processed_rel_path"]
        target.write_bytes(crop)
        return [*current, missing]

    archive_of(archive)
    original = archive.read_bytes()
    result = study_data.prepare(raw, archive, tmp_path / "restored", verify_only=True)
    assert result["reused"] == 1121 and result["reprocessed"] == 0
    assert archive.read_bytes() == original
    assert (tmp_path / "restored/manifest.json").read_bytes() == manifest
    damaged = tmp_path / "damaged.zip"
    archive_of(damaged, records[10]["processed_rel_path"])
    monkeypatch.setattr(study_data, "run_preprocessing", repair)
    result = study_data.prepare(raw, damaged, tmp_path / "repaired")
    assert result["reused"] == 1120 and result["reprocessed"] == 1
    assert (
        study_data.prepare(raw, damaged, tmp_path / "new_runtime", verify_only=True)["reused"]
        == 1121
    )


def test_cache_legacy_shards_and_partial_data_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """기존 인덱스 캐시와 변경되지 않은 사진을 새 코호트에서도 재사용합니다."""
    root = tmp_path / "embeddings"
    model = torch.nn.Identity()
    records = [{} for _ in range(3)]
    hashes = [("raw0", "crop0"), ("raw1", "crop1"), ("raw2", "crop2")]

    def create(digest: str, photo_hashes: list[tuple[str, str]]) -> EmbeddingCache:
        """지정한 서명으로 CPU 캐시를 초기화합니다."""
        return EmbeddingCache(
            root,
            records,
            tmp_path,
            digest,
            "mock",
            model,
            224,
            8,
            torch.device("cpu"),
            photo_hashes=photo_hashes,
        )

    initial = create("one", hashes)
    variant = fingerprint({"aug_pass": 0, "rotation_degrees": 0.0, "flipped": False})
    directory = initial.legacy_root / variant
    directory.mkdir(parents=True)
    np.savez(
        directory / "existing.npz",
        identity=initial.legacy_key,
        variant=variant,
        indices=np.array([0], dtype=np.int64),
        features=np.array([[0, 1]], dtype=np.float32),
    )
    pending = []

    def fake_extract(_: EmbeddingCache, indices: list[int], *options: object) -> np.ndarray:
        """실제로 계산되었어야 하는 특징 인덱스만 적습니다."""
        pending.extend(indices)
        return np.array([[i, 1] for i in indices], dtype=np.float32)

    monkeypatch.setattr(EmbeddingCache, "_extract", fake_extract)
    initial.get([0, 1, 2])
    assert pending == [1, 2] and initial.hits == 1
    updated = create("two", [hashes[1], ("new", "new"), hashes[2]])
    pending.clear()
    updated.get([0, 1, 2])
    assert pending == [1] and updated.hits == 2


def test_single_image_rejects_another_file_with_the_same_relative_name(tmp_path: Path) -> None:
    """상대 경로가 같아도 외부 이미지가 기존 크롭으로 판별되는 문제를 막습니다."""
    from hashlib import sha256

    from src.predict import verified_crop_record

    relative = Path(MAIN) / "sample.jpg"
    source = tmp_path / "source" / relative
    external = tmp_path / "external" / relative
    processed = tmp_path / "processed"
    target = processed / relative
    for path, data in ((source, b"original"), (external, b"changed"), (target, b"crop")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    records = [
        {
            "source_path": str(source),
            "processed_rel_path": relative.as_posix(),
            "class_name": MAIN,
            "individual_id": MAIN,
        }
    ]
    (processed / "manifest.json").write_text(json.dumps(records))
    assert verified_crop_record(source, processed) == records[0]
    with pytest.raises(ValueError, match="경로"):
        verified_crop_record(external, processed)
    (processed / study_data.INTEGRITY).write_text(
        json.dumps(
            {
                "sources": {
                    relative.as_posix(): {
                        "raw_sha256": sha256(source.read_bytes()).hexdigest(),
                        "crop_sha256": sha256(target.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="SHA-256"):
        verified_crop_record(external, processed)


def test_external_evaluation_rejects_reencoded_duplicate(tmp_path: Path) -> None:
    """외부 데이터 명칭이 달라져도 동일 RGB 픽셀은 평가에 사용하지 않습니다."""
    from argparse import Namespace

    from src.test_study_model import run_test

    source, external = tmp_path / "raw", tmp_path / "external"
    for root, name in ((source, "original.png"), (external, "export.bmp")):
        path = root / MAIN / name
        path.parent.mkdir(parents=True)
        Image.new("RGB", (16, 24), (20, 30, 40)).save(path)
    args = Namespace(
        run_id="evaluation",
        mode="external",
        source=source,
        external_source=external,
        output=tmp_path,
        processed=tmp_path / "processed",
        checkpoint=None,
        n_per_class=10,
    )
    with pytest.raises(ValueError, match="같은 원본"):
        run_test(args)
