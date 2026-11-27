"""Manifest의 원본·개체·촬영 정보를 기준으로 실험 분할을 고정합니다."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from src.config import GDRIVE_CATEGORIES, GROUP_CV_FOLDS, RANDOM_SEED
from src.dataset import create_group_folds
from src.preprocess import collect_source_files

MAIN = "main_siamese_cat"
OTHER = "other_siamese_cat"
CLASSES = sorted(GDRIVE_CATEGORIES)
FILENAME_DATE = re.compile(r"(?<!\d)(20\d{2})[-_.]?([01]\d)[-_.]?([0-3]\d)(?:\d{6})?(?!\d)")


def source_key(path: str | Path, class_name: str) -> str:
    """OS 및 원본 루트와 독립적인 클래스 상대 경로를 반환합니다."""
    normalized = unicodedata.normalize("NFC", str(path).replace("\\", "/")).casefold()
    parts = PurePosixPath(normalized).parts
    if class_name not in parts or parts[-1] == class_name:
        raise ValueError(f"Manifest 원본 경로에 클래스 상대 경로가 없습니다: {class_name}")
    class_index = parts.index(class_name)
    if ".." in parts[class_index:]:
        raise ValueError("Manifest 원본 경로에 상위 경로 이동이 있습니다.")
    return "/".join(parts[class_index:])


def processed_path(root: Path, relative: str) -> Path:
    """전처리 루트 내부에 한정된 Manifest 이미지 경로를 반환합니다."""
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("전처리 이미지 경로가 데이터 폴더를 벗어납니다.")
    return candidate


def load_manifest(manifest_path: Path, *, require_other_ids: bool = True) -> list[dict]:
    """Manifest 경로를 검증하고 요청 시 누락된 다른 샴고양이 ID를 거부합니다."""
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("비어 있거나 지원하지 않는 전처리 Manifest입니다.")
    paths = set()
    for record in records:
        cls = record["class_name"]
        rel = record["processed_rel_path"]
        if cls not in CLASSES or rel in paths:
            raise ValueError("알 수 없는 클래스 또는 중복 처리 경로가 있습니다.")
        source_key(record["source_path"], cls)
        if (
            require_other_ids
            and cls == OTHER
            and record.get("individual_id") in (None, "", OTHER, "unknown")
        ):
            raise ValueError("다른 샴고양이는 개체 ID가 있는 Manifest가 필요합니다.")
        paths.add(rel)
    return records


def source_index(source_dir: Path) -> dict[str, Path]:
    """원본 위치가 바뀌어도 Manifest와 대조할 수 있는 경로 색인을 만듭니다."""
    index = {}
    for path, cls, _ in collect_source_files(source_dir):
        key = source_key(path.relative_to(source_dir), cls)
        if key in index:
            raise ValueError("대소문자 또는 유니코드 정규화 후 중복 원본 경로가 있습니다.")
        index[key] = path
    return index


def verified_dates(
    record: dict, original: Path | None, *, prefer_capture_metadata: bool = False
) -> set[tuple[str, str]]:
    """Manifest, 실제 파일명, 원본 EXIF에서 확인 가능한 촬영 단위를 반환합니다."""
    evidence = set()
    for key in ("scene_id", "burst_id"):
        if record.get(key):
            evidence.add((key, str(record[key])))
    date_values = [record.get(key) for key in ("capture_date", "date_taken", "datetime_original")]
    filename = record["source_path"].replace("\\", "/").rsplit("/", 1)[-1]
    exif_date = None
    if original is not None:
        try:
            with Image.open(original) as image:
                exif = image.getexif()
                exif_date = exif.get(36867) or exif.get_ifd(34665).get(36867)
        except (OSError, ValueError, SyntaxError):
            pass

    def dates(values: list) -> set[tuple[str, str]]:
        """실제로 존재하는 달력 날짜만 증거로 읽습니다."""
        found = set()
        for value in values:
            for match in FILENAME_DATE.finditer(str(value).replace(":", "-")):
                try:
                    day = date(*map(int, match.groups()))
                except ValueError:
                    continue
                found.add(("date", day.isoformat()))
        return found

    if prefer_capture_metadata:
        candidates = dates([exif_date]) or dates(date_values) or dates([filename])
        if len(candidates) == 1:
            evidence.update(candidates)
    else:
        evidence.update(dates([*date_values, filename, exif_date]))
    return evidence


def audit_dataset(
    records: list[dict],
    processed_dir: Path,
    originals: dict[str, Path],
    manifest_sha256: str,
    *,
    find_near: bool = True,
) -> tuple[str, dict[str, str], dict]:
    """파일 해시와 촬영 증거를 확인하고 분할 그룹 및 유사 사진 후보를 반환합니다."""
    digest = sha256(manifest_sha256.encode())
    parents = list(range(len(records)))
    seen: dict[tuple[str, str], int] = {}
    hashes: list[int] = []
    main_with_evidence = 0

    def root(index: int) -> int:
        """그룹 구성 요소의 대표 인덱스를 반환합니다."""
        while parents[index] != index:
            index = parents[index]
        return index

    def connect(key: tuple[str, str], index: int) -> None:
        """같은 증거를 가진 원본을 동일 그룹으로 연결합니다."""
        other = seen.setdefault(key, index)
        if records[other]["class_name"] != records[index]["class_name"]:
            raise ValueError("서로 다른 클래스에 동일한 전처리 이미지가 있습니다.")
        first, second = sorted((root(index), root(other)))
        parents[second] = first

    for index, record in enumerate(records):
        rel = record["processed_rel_path"]
        cls = record["class_name"]
        source = source_key(record["source_path"], cls)
        connect(("source", source), index)
        if cls == OTHER:
            individual = unicodedata.normalize("NFC", str(record["individual_id"])).casefold()
            connect(("other_identity", individual), index)
        if cls == MAIN:
            evidence = verified_dates(record, originals.get(source))
            main_with_evidence += bool(evidence)
            for key in sorted(evidence):
                connect(key, index)

        raw = processed_path(processed_dir, rel).read_bytes()
        digest.update(sha256(raw).digest())
        with Image.open(BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            connect(
                ("exact_pixels", sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()), index
            )
            if find_near:
                gray = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = gray.tobytes()
                dhash = 0
                for row in range(8):
                    for column in range(8):
                        offset = row * 9 + column
                        dhash = (dhash << 1) | (pixels[offset] > pixels[offset + 1])
                hashes.append(dhash)

    near_pairs = [
        [left, right, distance]
        for left, value in enumerate(hashes)
        for right in range(left + 1, len(hashes))
        if (distance := (value ^ hashes[right]).bit_count()) <= 4 and root(left) != root(right)
    ]
    groups = {
        record["processed_rel_path"]: str(root(index)) for index, record in enumerate(records)
    }
    group_counts = {
        cls: len({groups[r["processed_rel_path"]] for r in records if r["class_name"] == cls})
        for cls in CLASSES
    }
    audit = {
        "class_counts": dict(Counter(record["class_name"] for record in records)),
        "group_counts": group_counts,
        "main_with_capture_evidence": main_with_evidence,
        "main_without_capture_evidence": sum(r["class_name"] == MAIN for r in records)
        - main_with_evidence,
        "near_duplicate_review": "64-bit difference hash, Hamming distance <= 4",
        "near_duplicate_candidates": near_pairs,
    }
    return digest.hexdigest(), groups, audit


def indexed_folds(
    records: list[dict], indices: list[int], groups: dict[str, str], n_splits: int
) -> list[dict[str, list[int]]]:
    """선택한 데이터에 동일 그룹을 유지한 Fold별 Manifest 인덱스를 반환합니다."""
    position = {id(records[index]): index for index in indices}
    return [
        {"train": [position[id(r)] for r in train], "validation": [position[id(r)] for r in val]}
        for train, val in create_group_folds(
            [records[index] for index in indices], n_splits, RANDOM_SEED, groups
        )
    ]


def split_plan(records: list[dict], groups: dict[str, str], audit: dict) -> dict:
    """튜닝 전 그룹 Holdout을 확정하고 개발 데이터의 5개 Fold를 저장합니다."""
    all_indices = list(range(len(records)))
    development, holdout = all_indices, []
    min_groups = min(audit["group_counts"].values())
    if min_groups >= 3:
        outer = indexed_folds(records, all_indices, groups, GROUP_CV_FOLDS + 1)[0]
        development, holdout = outer["train"], outer["validation"]
    if min(audit["group_counts"][MAIN], audit["group_counts"][OTHER]) < 2:
        raise ValueError("Main/Other를 누수 없이 검증할 독립 그룹이 각각 2개 이상 필요합니다.")
    folds = indexed_folds(records, development, groups, GROUP_CV_FOLDS)
    for fold in folds:
        classes = {records[i]["class_name"] for i in fold["train"]}
        if not {MAIN, OTHER}.issubset(classes):
            raise ValueError("5-Fold 학습에 필요한 양성과 다른 샴고양이 그룹이 부족합니다.")
    near_pairs = audit["near_duplicate_candidates"]
    holdout_indices = set(holdout)
    fold_partitions = [set(fold["validation"]) for fold in folds]
    return {
        "development": development,
        "holdout": holdout,
        "folds": folds,
        "groups": groups,
        "near_duplicate_cross_holdout": [
            [a, b, d] for a, b, d in near_pairs if (a in holdout_indices) != (b in holdout_indices)
        ],
        "near_duplicate_cross_fold_counts": [
            sum(
                (a in val) != (b in val)
                for a, b, _ in near_pairs
                if a not in holdout_indices and b not in holdout_indices
            )
            for val in fold_partitions
        ],
    }


def strict_training_indices(
    records: list[dict],
    sampled_items: list[tuple[Path, str, str]],
    groups: dict[str, str],
) -> list[int]:
    """샘플 원본과 확인된 촬영 그룹 및 다른 샴고양이 전체 개체를 제외합니다."""
    sampled = {source_key(path, cls) for path, cls, _ in sampled_items}
    present = {source_key(r["source_path"], r["class_name"]) for r in records}
    if sampled - present:
        raise ValueError(
            "Strict Holdout 원본을 Manifest와 대조할 수 없습니다. 원본 루트를 확인하세요."
        )
    excluded = {
        groups[record["processed_rel_path"]]
        for record in records
        if source_key(record["source_path"], record["class_name"]) in sampled
    }
    return [
        i
        for i, record in enumerate(records)
        if groups[record["processed_rel_path"]] not in excluded
    ]


def matching_record(path: Path, records: list[dict], class_name: str | None = None) -> dict:
    """대조한 원본 경로에서 전처리 레코드를 찾고 모호한 경우 실패합니다."""
    candidates = [class_name] if class_name else [cls for cls in CLASSES if cls in path.parts]
    if len(candidates) != 1:
        raise ValueError("원본 이미지가 속한 단일 클래스 폴더가 필요합니다.")
    cls = candidates[0]
    key = source_key(path, cls)
    matches = [
        record
        for record in records
        if record["class_name"] == cls and source_key(record["source_path"], cls) == key
    ]
    if len(matches) != 1:
        raise ValueError("원본 이미지에 대응하는 단일 Manifest 레코드를 찾지 못했습니다.")
    return matches[0]
