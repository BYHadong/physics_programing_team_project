"""전처리 ZIP을 읽기 전용으로 검증하고 누락·손상된 크롭만 복구합니다."""

from __future__ import annotations

import argparse
import json
import stat
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from PIL import Image

from src.config import (
    CROP_INTEGRITY_FILENAME,
    CROP_MARGIN_RATIO,
    PREPROCESSED_SIZE,
    YOLO_CONF_THRESHOLD,
    YOLO_WEIGHTS_PATH,
)
from src.experiment_splits import load_manifest, processed_path, source_index
from src.prepare_dataset import TARGET_COUNTS, record_keys
from src.preprocess import run_preprocessing
from src.study_audit import file_sha256
from src.tune_hyperparameters import write_artifact, write_json

INTEGRITY = CROP_INTEGRITY_FILENAME
PREPROCESS = {
    "size": PREPROCESSED_SIZE,
    "margin": CROP_MARGIN_RATIO,
    "confidence": YOLO_CONF_THRESHOLD,
    "detector": YOLO_WEIGHTS_PATH.name,
    "jpeg_quality": 95,
}


def verify_crop(data: bytes) -> bool:
    """JPEG 디코딩과 출력 모드 및 512픽셀 크기를 확인합니다."""
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return (
                image.format == "JPEG"
                and image.mode == "RGB"
                and image.size == (PREPROCESSED_SIZE, PREPROCESSED_SIZE)
            )
    except (OSError, ValueError):
        return False


def recover(
    path: Path, staging: Path, originals: dict[str, Path], raw_hashes: dict[str, str]
) -> tuple[list[dict], bytes, dict, list[str], dict]:
    """CRC와 매니페스트, 원본 해시를 확인하고 정상적인 행만 복사합니다."""
    with ZipFile(path) as archive:
        names = {}
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
                or name in names
            ):
                raise ValueError("ZIP에 안전하지 않거나 중복된 경로가 있습니다.")
            names[name] = entry
        manifests = [name for name in names if name.name == "manifest.json"]
        if len(manifests) != 1:
            raise ValueError("ZIP 내 manifest.json이 하나여야 합니다.")
        manifest = manifests[0]
        original_manifest = archive.read(names[manifest])
        tmp_manifest = staging / "manifest.json"
        tmp_manifest.write_bytes(original_manifest)
        records = load_manifest(tmp_manifest, require_other_ids=False)
        keys = record_keys(records)
        integrity_name = manifest.with_name(INTEGRITY)
        integrity = (
            json.loads(archive.read(names[integrity_name])) if integrity_name in names else {}
        )
        origin = {"sha256": sha256(original_manifest).hexdigest(), "original_keys": keys}
        sidecar = path.with_suffix(".integrity.json")
        if not integrity and sidecar.is_file():
            saved = json.loads(sidecar.read_text("utf-8"))
            if saved.get("archived_manifest", {}).get("sha256") == origin["sha256"]:
                integrity = saved
        if integrity and integrity.get("preprocessing") != PREPROCESS:
            raise ValueError("보관된 전처리 설정이 현재 파이프라인과 다릅니다.")
        previous = integrity.get("sources", {})
        valid = []
        failures = []
        for row, key in zip(records, keys):
            if key not in originals:
                failures.append(f"{key}: original no longer present")
                continue
            try:
                relative = row["processed_rel_path"]
                target = processed_path(staging, relative)
                data = archive.read(names[manifest.parent / relative])
                if (
                    row.get("processed_size", [PREPROCESSED_SIZE] * 2) != [PREPROCESSED_SIZE] * 2
                    or not verify_crop(data)
                    or (
                        key in previous
                        and (
                            raw_hashes[key] != previous[key]["raw_sha256"]
                            or sha256(data).hexdigest() != previous[key]["crop_sha256"]
                        )
                    )
                ):
                    raise ValueError("이미지 디코딩, 크기 또는 해시 불일치")
            except (BadZipFile, KeyError, OSError, ValueError) as error:
                failures.append(f"{key}: {error}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            valid.append(row)
    return valid, original_manifest, integrity, failures, integrity.get("archived_manifest", origin)


def prepare(
    source: Path,
    archive: Path,
    output: Path,
    *,
    expected: dict = TARGET_COUNTS,
    verify_only: bool = False,
) -> dict:
    """ZIP을 먼저 사용해 복구하고 Drive 원본과 기존 ZIP을 읽기 전용으로 유지합니다."""
    originals = source_index(source)
    counts = dict(Counter(key.split("/", 1)[0] for key in originals))
    if counts != expected:
        raise ValueError(f"원본 클래스 수량이 다릅니다. 현재 {counts}, 예상 {expected}")
    raw_hashes = {key: file_sha256(path) for key, path in originals.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        records = load_manifest(output / "manifest.json", require_other_ids=False)
        integrity = json.loads((output / INTEGRITY).read_text("utf-8"))
        entries = integrity["sources"]
        if (
            set(record_keys(records)) != set(originals)
            or any(
                entries[key]["raw_sha256"] != raw_hashes[key]
                or entries[key]["crop_sha256"]
                != file_sha256(processed_path(output, row["processed_rel_path"]))
                for row, key in zip(records, record_keys(records))
            )
            or integrity["preprocessing"] != PREPROCESS
        ):
            raise ValueError(
                f"로컬 복원본이 변경되었습니다. 새 --output 폴더를 사용하세요. {output}"
            )
        summary = {**integrity["summary"], "reused": len(records), "reprocessed": 0}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary

    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder) / "empty"
        staging.mkdir()
        reused, original_manifest, verified, failures, origin = [], b"", {}, [], {}
        candidate = None
        repaired = sorted(archive.parent.glob(f"{archive.stem}-repair-*.zip"), reverse=True)
        for index, option in enumerate([archive, *repaired, archive.with_name("processed.zip")]):
            if not option.is_file():
                continue
            target = Path(folder) / f"candidate-{index}"
            target.mkdir()
            try:
                found = recover(option, target, originals, raw_hashes)
            except (BadZipFile, OSError, KeyError, ValueError, TypeError) as error:
                print(f"[ZIP 확인] {option.name} · {error}", flush=True)
                continue
            if candidate is None or len(found[0]) > len(reused):
                reused, original_manifest, verified, failures, origin = found
                staging, candidate = target, option
            if len(reused) == len(originals):
                break
        missing = len(originals) - len(reused)
        if missing and verify_only:
            raise ValueError(f"복구가 필요한 전처리 이미지 {missing}장. {failures[:5]}")
        manifest = staging / "manifest.json"
        if not missing and not failures:
            # 기존 바이트 및 행 순서가 같으면 기존 224 임베딩의 데이터 키도 유지됩니다.
            manifest.write_bytes(original_manifest)
            records = reused
        else:
            fixed = [
                {**row, "source_path": str(originals[key])}
                for row, key in zip(reused, record_keys(reused))
            ]
            manifest.write_text(json.dumps(fixed, ensure_ascii=False), "utf-8")
            records = run_preprocessing(source_dir=source, output_dir=staging)
            ordered = {key: index for index, key in enumerate(origin.get("original_keys", []))}
            records.sort(key=lambda row: ordered.get(record_keys([row])[0], len(ordered)))
            manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
        keys = record_keys(records)
        if set(keys) != set(originals):
            raise ValueError("전처리 결과와 전체 원본의 일대일 대응에 실패했습니다.")
        entries = {}
        for row, key in zip(records, keys):
            data = processed_path(staging, row["processed_rel_path"]).read_bytes()
            if not verify_crop(data):
                raise ValueError(f"검증을 통과하지 못한 크롭 {row['processed_rel_path']}")
            entries[key] = {"raw_sha256": raw_hashes[key], "crop_sha256": sha256(data).hexdigest()}
        summary = {
            "archive": str(candidate) if candidate else None,
            "counts": counts,
            "reused": len(reused),
            "reprocessed": missing,
            "archived_source_hashes_verified": len(verified.get("sources", {})),
            "legacy_provenance": "historical source hashes absent" if not verified else None,
            "rejected": failures,
        }
        contents = {
            "preprocessing": PREPROCESS,
            "sources": entries,
            "summary": summary,
            "archived_manifest": origin,
        }
        write_json(staging / INTEGRITY, contents)
        if missing:
            # 수리본은 원본 ZIP과 다른 이름으로만 보관합니다.
            digest = sha256((staging / INTEGRITY).read_bytes()).hexdigest()[:16]
            saved = archive.with_stem(f"{archive.stem}-repair-{digest}")
            with ZipFile(Path(folder) / "repaired.zip", "w") as bundle:
                for relative in [
                    "manifest.json",
                    INTEGRITY,
                    *(r["processed_rel_path"] for r in records),
                ]:
                    # ZIP의 시간 값이 결과 바이트를 바꾸지 않게 합니다.
                    bundle.writestr(f"processed/{relative}", (staging / relative).read_bytes())
            write_artifact(saved, (Path(folder) / "repaired.zip").read_bytes())
            summary["repair_archive"] = str(saved)
        elif candidate == archive and not archive.with_suffix(".integrity.json").exists():
            # 과거 원본까지 확인할 수 없으므로 현재로부터의 변경을 검출하는 기준을 남깁니다.
            write_json(archive.with_suffix(".integrity.json"), contents)
        staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    """실험용 전처리 상태를 검증하거나 부분 복구합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.archive, args.output, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
