"""기존 크롭을 보존하며 신규 사진을 추가한 실험 데이터를 준비합니다."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import BadZipFile, ZipFile
from zlib import crc32

from src.config import COLAB_BACKUP_DIR, OUTPUTS_DIR, SOURCE_DATA_DIR
from src.drive_sync import restore_processed
from src.experiment_splits import (
    MAIN,
    OTHER,
    load_manifest,
    processed_path,
    source_index,
    source_key,
)
from src.preprocess import run_preprocessing

BASE_COUNTS = {MAIN: 77, OTHER: 572, "etc_cats": 462}
TARGET_COUNTS = {**BASE_COUNTS, MAIN: 87}


def record_keys(records: list[dict]) -> list[str]:
    """Manifest 행 순서에 대응하는 클래스 기준 원본 키를 반환합니다."""
    keys = [source_key(row["source_path"], row["class_name"]) for row in records]
    if len(set(keys)) != len(records):
        raise ValueError("Manifest에 동일 원본이 여러 번 기록되어 있습니다.")
    return keys


def read_snapshot(path: Path) -> tuple[list[dict], list[Path]]:
    """폴더 또는 ZIP의 Manifest와 누락 없이 존재하는 크롭 경로를 반환합니다."""
    if path.is_dir():
        manifest = path / "manifest.json"
        records = load_manifest(manifest)
        references = [Path(row["processed_rel_path"]) for row in records]
        if not all(processed_path(path, str(ref)).is_file() for ref in references):
            raise FileNotFoundError("Manifest에 있는 전처리 이미지가 빠졌습니다.")
        return records, references

    with ZipFile(path) as archive:
        members = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
        manifests = [entry for entry in members if PurePosixPath(entry).name == "manifest.json"]
        if len(manifests) != 1:
            raise ValueError("ZIP의 manifest.json은 한 개여야 합니다.")
        records = json.loads(archive.read(manifests[0]))
        references = [PurePosixPath(row["processed_rel_path"]) for row in records]
        if len(set(references)) != len(records) or any(
            ref.is_absolute() or ".." in ref.parts for ref in references
        ):
            raise ValueError("ZIP Manifest에 중복 또는 안전하지 않은 전처리 경로가 있습니다.")
        parent = PurePosixPath(manifests[0]).parent
        if any((parent / ref.as_posix()).as_posix() not in members for ref in references):
            raise FileNotFoundError("ZIP에 필요한 전처리 이미지가 빠졌습니다.")
    return records, [Path(str(ref)) for ref in references]


def snapshot_counts(
    records: list[dict], originals: dict[str, Path], expected_other_count: int
) -> dict[str, int]:
    """Manifest가 현재 원본과 대응하는지 검증하고 클래스별 수량을 반환합니다."""
    if set(record_keys(records)).difference(originals):
        raise FileNotFoundError("전처리 Manifest에 대응하는 일부 원본을 찾지 못했습니다.")
    if (
        len({r["individual_id"] for r in records if r["class_name"] == OTHER})
        != expected_other_count
    ):
        raise ValueError("Manifest의 다른 샴고양이 개체 수가 예상과 다릅니다.")
    return dict(Counter(row["class_name"] for row in records))


def select_snapshot(
    candidates: list[Path],
    originals: dict[str, Path],
    base_counts: dict[str, int],
    target_counts: dict[str, int],
    expected_other_count: int,
) -> tuple[Path, bool]:
    """완성된 확장본을 우선하고, 없으면 기존 크롭을 추가 작업의 기초로 선택합니다."""
    base = None
    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        try:
            records, _ = read_snapshot(path)
            counts = snapshot_counts(records, originals, expected_other_count)
        except (OSError, BadZipFile, ValueError, KeyError, TypeError) as error:
            print(f"[전처리 데이터 확인] {path} · {error}", flush=True)
            continue
        if counts == target_counts:
            return path, False
        if counts == base_counts and base is None:
            base = path
    if base is None:
        raise FileNotFoundError(
            "현재 원본과 일치하는 기존 1,111장 또는 완성된 1,121장 전처리를 찾지 못했습니다."
        )
    return base, True


def copy_snapshot(source: Path, destination: Path) -> None:
    """선택한 기존 크롭과 Manifest를 로컬 작업 폴더로 복원하거나 복사합니다."""
    if source.is_file():
        restore_processed(source, destination)
        return
    _, references = read_snapshot(source)
    for relative in [*references, Path("manifest.json")]:
        original = processed_path(source, str(relative))
        output = processed_path(destination, str(relative))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, output)


def backup_snapshot(processed: Path, archive_path: Path) -> Path:
    """완성본을 새 ZIP으로 저장하고 동일한 백업이 있으면 그 경로를 반환합니다."""
    _, references = read_snapshot(processed)
    files = [Path("manifest.json"), *references]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    # 동일 Manifest와 크롭의 ZIP을 재사용하며, 충돌하는 기존 ZIP은 보존합니다.
    for existing in [
        archive_path,
        *sorted(archive_path.parent.glob(f"{archive_path.stem}-*.zip")),
    ]:
        if not existing.is_file():
            continue
        try:
            with ZipFile(existing) as archive:
                if all(
                    archive.getinfo(f"processed/{ref.as_posix()}").CRC
                    == crc32((processed / ref).read_bytes())
                    for ref in files
                ):
                    return existing
        except (OSError, BadZipFile, KeyError):
            continue
    if archive_path.exists():
        archive_path = archive_path.with_stem(f"{archive_path.stem}-{uuid4().hex[:8]}")
    with TemporaryDirectory(dir=processed.parent) as directory:
        staged = Path(directory) / "dataset.zip"
        # 기존 크롭은 JPEG이므로 ZIP에서 다시 압축하지 않습니다.
        with ZipFile(staged, "w") as archive:
            for ref in files:
                archive.write(processed / ref, f"processed/{ref.as_posix()}")
        with staged.open("rb") as source:
            # 이 이름이 이미 있으면 실패하므로 기존 백업을 덮어쓰지 않습니다.
            with archive_path.open("xb") as target:
                shutil.copyfileobj(source, target)
    return archive_path


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    archive_path: Path,
    candidates: list[Path],
    *,
    base_counts: dict[str, int] | None = None,
    target_counts: dict[str, int] | None = None,
    expected_other_count: int = 21,
) -> tuple[Path, Path]:
    """원본과 일치하는 확장본을 준비하고 예시 원본 및 Drive 백업 경로를 반환합니다."""
    base_counts = BASE_COUNTS if base_counts is None else base_counts
    target_counts = TARGET_COUNTS if target_counts is None else target_counts
    originals = source_index(source_dir)
    counts = dict(Counter(PurePosixPath(key).parts[0] for key in originals))
    if counts != target_counts:
        raise ValueError(f"원본 수량이 다릅니다. 현재 {counts}, 예상 {target_counts}")
    print(f"[원본 확인] {sum(counts.values())}장 · {counts}", flush=True)

    if not output_dir.exists():
        previous_archives = sorted(archive_path.parent.glob(f"{archive_path.stem}-*.zip"))
        source, extend = select_snapshot(
            [archive_path, *previous_archives, *candidates],
            originals,
            base_counts,
            target_counts,
            expected_other_count,
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output_dir.parent) as directory:
            staging = Path(directory) / "processed"
            copy_snapshot(source, staging)
            manifest = staging / "manifest.json"
            records = load_manifest(manifest)
            if extend:
                # 이전 Drive 루트나 Windows 경로의 차이를 해소해 기존 크롭을 재사용합니다.
                for row, key in zip(records, record_keys(records)):
                    row["source_path"] = str(originals[key])
                manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
                print(
                    f"[전처리] 기존 {len(records)}장 재사용 · 신규 {len(originals) - len(records)}장",
                    flush=True,
                )
                run_preprocessing(source_dir=source_dir, output_dir=staging)
            else:
                print(f"[전처리] 완성된 {len(records)}장 사용 · {source}", flush=True)
            records, _ = read_snapshot(staging)
            if snapshot_counts(records, originals, expected_other_count) != target_counts:
                raise ValueError("확장 전처리 결과의 클래스별 수량이 다릅니다.")
            staging.rename(output_dir)

    records, _ = read_snapshot(output_dir)
    if snapshot_counts(records, originals, expected_other_count) != target_counts:
        raise ValueError(f"기존 출력 폴더가 이 실험과 일치하지 않습니다. {output_dir}")
    saved = backup_snapshot(output_dir, archive_path)
    keys = record_keys(records)
    example = next(originals[key] for key, row in zip(keys, records) if row["class_name"] == MAIN)
    return example, saved


def main() -> None:
    """1,121장 전처리 확장 및 새 백업 저장을 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "processed_1121")
    parser.add_argument("--archive", type=Path, default=COLAB_BACKUP_DIR / "processed_1121.zip")
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    args = parser.parse_args()
    example, backup = prepare_dataset(args.source, args.output, args.archive, args.candidate)
    print(f"[백업 저장] {backup}", flush=True)
    print(json.dumps({"image": str(example), "backup": str(backup)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
