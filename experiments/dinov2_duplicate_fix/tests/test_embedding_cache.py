"""합성 특징으로 증강 재사용, 캐시 분리 및 안전한 ZIP 복원을 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from src.drive_sync import restore_processed
from src.embedding_cache import EmbeddingCache
from src.experiment_splits import MAIN, OTHER
from src.linear_probe import Hyperparameters, fit_probe


def test_cache_variants_and_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """캐시 재개와 증강 수량·반전 재사용 및 입력 키 변경을 확인합니다."""
    calls = []

    def fake_extract(
        self: EmbeddingCache,
        indices: list[int],
        aug_pass: int,
        rotation_degrees: float,
        flip: bool,
    ) -> np.ndarray:
        """추출 요청을 기록하고 고정 길이 합성 특징을 반환합니다."""
        calls.append((indices, aug_pass, rotation_degrees, flip))
        return np.array([[i, aug_pass, rotation_degrees, flip] for i in indices], dtype=np.float32)

    monkeypatch.setattr(EmbeddingCache, "_extract", fake_extract)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)

    def cache(data: str = "manifest_and_images", model_id: str = "dinov2") -> EmbeddingCache:
        """같은 가중치에서 설정으로 구분한 캐시를 생성합니다."""
        return EmbeddingCache(
            tmp_path,
            [{} for _ in range(8)],
            tmp_path,
            data,
            model_id,
            model,
            224,
            2,
            torch.device("cpu"),
        )

    first = cache()
    first.get([4, 1], 1, 15, 0)
    first.get([4, 6], 1, 15, 0)
    assert calls[1][0] == [6]
    resumed = cache()
    before = len(calls)
    assert resumed.get([6, 1, 4], 1, 15, 0)[:, 0].tolist() == [6, 1, 4]
    assert len(calls) == before
    changed_flip = resumed.get([6, 1, 4], 1, 15, 1)
    assert changed_flip[:, 3].tolist() == [1, 1, 1]
    resumed.get([4, 1, 6], 1, 15, 0.5)
    assert len(calls) == before + 1
    resumed.get([4], 2, 15, 0)
    resumed.get([4], 1, 30, 0)
    assert len(calls) == before + 3
    resumed.get([4])
    resumed.get([4], 0, 30, 1)
    assert len(calls) == before + 4
    assert cache("changed_manifest").key != first.key
    assert cache(model_id="changed_model").key != first.key
    with torch.no_grad():
        model.weight.fill_(2)
    assert cache().key != first.key


def test_restore_processed_preserves_backups(tmp_path: Path) -> None:
    """ZIP을 한 번 복원하며 기존 로컬 파일과 ZIP을 덮어쓰지 않는지 확인합니다."""
    archive = tmp_path / "processed.zip"
    output = tmp_path / "data/processed"
    relative = "main_siamese_cat/sample.jpg"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("processed/manifest.json", json.dumps([{"processed_rel_path": relative}]))
        zipped.writestr(f"processed/{relative}", b"example")
    expected_archive = archive.read_bytes()
    assert restore_processed(archive, output)
    assert not restore_processed(archive, output)
    assert archive.read_bytes() == expected_archive
    conflict = tmp_path / "conflict"
    (conflict / relative).parent.mkdir(parents=True)
    (conflict / relative).write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        restore_processed(archive, conflict)
    assert (conflict / relative).read_bytes() == b"existing"

    unsafe = tmp_path / "unsafe.zip"
    with ZipFile(unsafe, "w") as zipped:
        zipped.writestr("../after.jpg", b"path escape")
    with pytest.raises(ValueError, match="안전하지 않은"):
        restore_processed(unsafe, tmp_path / "new_output")


def test_c_change_reuses_cached_dino_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C를 바꾸면 분류기만 재학습하고 증강 특징을 다시 추출하지 않습니다."""
    calls = []

    def extract(
        self: EmbeddingCache,
        indices: list[int],
        aug_pass: int,
        rotation_degrees: float,
        flip: bool,
    ) -> np.ndarray:
        """C와 독립된 모의 특징을 만들고 추출 횟수를 기록합니다."""
        calls.append(indices)
        return np.asarray([[float(i), float(aug_pass)] for i in indices], dtype=np.float32)

    monkeypatch.setattr(EmbeddingCache, "_extract", extract)
    cache = EmbeddingCache(
        tmp_path,
        [{"class_name": MAIN}, {"class_name": OTHER}],
        tmp_path,
        "synthetic",
        "dinov2",
        torch.nn.Identity(),
        224,
        32,
        torch.device("cpu"),
    )
    first, _, _ = fit_probe(cache, [0, 1], Hyperparameters(c=1.0, neg_aug_passes=1))
    extracted = len(calls)
    stronger, _, _ = fit_probe(cache, [0, 1], Hyperparameters(c=0.01, neg_aug_passes=1))
    assert len(calls) == extracted
    assert (first.C, stronger.C) == (1.0, 0.01)
    assert np.linalg.norm(stronger.coef_) < np.linalg.norm(first.coef_)
