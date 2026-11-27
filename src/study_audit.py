"""원본·촬영 정보와 이미지 유사도를 감사해 분할에 쓸 그룹을 확정합니다."""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.experiment_splits import CLASSES, MAIN, OTHER, processed_path, source_key, verified_dates

DHASH_THRESHOLDS = (4, 8)
COSINE_THRESHOLDS = (0.95, 0.98, 0.995)


def metadata_csv(path: Path | None, columns: set[str]) -> list[dict]:
    """선택적인 수동 확인 CSV의 필수 열과 행을 읽습니다."""
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not columns <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: CSV 필수 열 {', '.join(sorted(columns))}이 필요합니다.")
        rows = list(reader)
    if any(not (row.get(key) or "").strip() for row in rows for key in columns):
        raise ValueError(f"{path}: CSV의 필수 값이 비어 있습니다.")
    return [{key: row[key].strip() for key in columns} for row in rows]


def file_sha256(path: Path) -> str:
    """원본을 변경하지 않고 바이트 SHA-256을 계산합니다."""
    digest = sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    """축소된 명암의 좌우 차이를 64비트 dHash로 변환합니다."""
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            index = row * 9 + column
            value = (value << 1) | (pixels[index] > pixels[index + 1])
    return value


def original_pixel_sha256(path: Path) -> str:
    """EXIF 방향을 정리하고 파일 인코딩과 독립적인 RGB 픽셀 해시를 반환합니다."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()


def other_id(record: dict) -> str | None:
    """신뢰 가능한 Manifest 개체명을 그룹 비교용 문자열로 반환합니다."""
    cls = record["class_name"]
    identity = record.get("individual_id")
    if cls == MAIN or not identity:
        return None
    normalized = unicodedata.normalize("NFC", str(identity)).strip().casefold()
    return normalized if normalized not in ("", cls, "unknown") else None


def processed_digest(records: list[dict], directory: Path, manifest_hash: str) -> str:
    """캐시 형식과 동일한 Manifest·전처리 이미지 서명을 다시 계산합니다."""
    digest = sha256(manifest_hash.encode())
    for row in records:
        digest.update(
            bytes.fromhex(file_sha256(processed_path(directory, row["processed_rel_path"])))
        )
    return digest.hexdigest()


def quarantine_other_copy_conflicts(
    records: list[dict], groups: dict[str, str], file_hashes: list[dict], conflicts: list[dict]
) -> tuple[dict[int, str], list[dict], list[dict]]:
    """개체 라벨이 충돌하는 다른 샴고양이의 동일 원본을 격리합니다."""
    identity = [other_id(row) if row["class_name"] == OTHER else None for row in records]
    by_hash = defaultdict(list)
    for row in file_hashes:
        by_hash[row["original_sha256"]].append(row["index"])

    copies = {}
    for conflict in conflicts:
        if conflict["evidence"] != "original_sha256":
            continue
        left, right = conflict["left"], conflict["right"]
        if not identity[left] or not identity[right] or identity[left] == identity[right]:
            continue
        digest = file_hashes[left]["original_sha256"]
        indices = by_hash[digest]
        if all(identity[index] for index in indices):
            copies[digest] = indices

    # 라벨은 보존하면서, 복제 관계로 연결된 개체를 분할에서만 함께 둡니다.
    paths = [row["processed_rel_path"] for row in records]
    neighbors = defaultdict(set)
    for indices in copies.values():
        involved = {groups[paths[index]] for index in indices}
        for group in involved:
            neighbors[group].update(involved)
    representative = {}
    for group in sorted(neighbors):
        if group in representative:
            continue
        pending = [group]
        while pending:
            current = pending.pop()
            if current not in representative:
                representative[current] = group
                pending.extend(neighbors[current].difference(representative))
    for path, group in groups.items():
        groups[path] = representative.get(group, group)

    copy_hash = {index: digest for digest, indices in copies.items() for index in indices}
    retained_groups = {}
    for index, digest in copy_hash.items():
        path = paths[index]
        retained_groups[index] = groups[path]
        groups[path] = f"quarantined-copy:{digest}"

    resolved, unresolved = [], []
    for conflict in conflicts:
        left_hash = copy_hash.get(conflict["left"])
        if (
            left_hash
            and left_hash == copy_hash.get(conflict["right"])
            and conflict["evidence"] in {"original_sha256", "crop_sha256", "pixel_sha256"}
        ):
            resolved.append(conflict)
        else:
            unresolved.append(conflict)
    return retained_groups, resolved, unresolved


def scan_dataset(
    records: list[dict],
    processed_dir: Path,
    originals: dict[str, Path],
    manifest_hash: str,
    sessions: Path | None = None,
    review: Path | None = None,
    *,
    quarantine_other_copies: bool = False,
) -> tuple[str, dict[str, str], list[int], list[int], dict]:
    """확인된 증거만 그룹으로 결합하고, 증거 없는 양성은 평가 대상에서 제외합니다."""
    paths = {r["processed_rel_path"]: i for i, r in enumerate(records)}
    session_ids: dict[int, str] = {}
    for row in metadata_csv(sessions, {"processed_rel_path", "session_id"}):
        index = paths.get(row["processed_rel_path"])
        if index is None or records[index]["class_name"] != MAIN or index in session_ids:
            raise ValueError("촬영 CSV의 양성 경로가 Manifest에 없거나 중복되었습니다.")
        session_ids[index] = row["session_id"]
    decisions = {}
    for row in metadata_csv(review, {"left", "right", "decision"}):
        left = paths.get(row["left"])
        right = paths.get(row["right"])
        if left is None or right is None or left == right:
            raise ValueError("유사 사진 검토 CSV에 Manifest와 대조할 수 없는 쌍이 있습니다.")
        pair = tuple(sorted((left, right)))
        if pair in decisions or row["decision"] not in {"duplicate", "different"}:
            raise ValueError("검토 CSV는 쌍마다 duplicate 또는 different 한 개를 지정해야 합니다.")
        decisions[pair] = row["decision"]

    digest = sha256(manifest_hash.encode())
    parents = list(range(len(records)))
    identities = [{identity} if (identity := other_id(r)) else set() for r in records]
    seen = {}
    confirmed = set()
    exact = []
    conflicts = []
    hashes = []
    file_hashes = []
    group_evidence = []

    def root(index: int) -> int:
        """그룹 연결의 현재 대표를 반환합니다."""
        while parents[index] != index:
            index = parents[index]
        return index

    def join(left: int, right: int, evidence: str) -> None:
        """동일 클래스와 모순 없는 개체 정보에서만 그룹을 결합합니다."""
        first, second = sorted((root(left), root(right)))
        if records[left]["class_name"] != records[right]["class_name"] or (
            (identities[first] | identities[second])
            and len(identities[first] | identities[second]) > 1
        ):
            conflicts.append({"left": left, "right": right, "evidence": evidence})
            return
        parents[second] = first
        identities[first].update(identities[second])

    def identify(key: tuple, index: int, duplicate: bool = False) -> None:
        """이미 확인된 증거와 같은 행을 찾습니다."""
        previous = seen.setdefault(key, index)
        if previous != index:
            if duplicate:
                exact.append({"left": previous, "right": index, "evidence": key[0]})
            if key[0] == "source":
                conflicts.append(
                    {"left": previous, "right": index, "evidence": "duplicate_manifest_source"}
                )
            join(previous, index, key[0])

    missing = [
        r["processed_rel_path"]
        for r in records
        if source_key(r["source_path"], r["class_name"]) not in originals
    ]
    if missing:
        raise FileNotFoundError(f"Manifest 원본 {len(missing)}장의 SHA-256을 읽을 수 없습니다.")
    for index, record in enumerate(records):
        relative, cls = record["processed_rel_path"], record["class_name"]
        source = source_key(record["source_path"], cls)
        original = originals[source]
        raw_hash = file_sha256(original)
        crop = processed_path(processed_dir, relative).read_bytes()
        crop_hash = sha256(crop)
        digest.update(crop_hash.digest())  # 이전 캐시와 동일한 데이터·Manifest 키
        identify(("source", source), index, duplicate=True)
        identify(("original_sha256", raw_hash), index, duplicate=True)
        identify(("crop_sha256", crop_hash.hexdigest()), index, duplicate=True)
        with Image.open(BytesIO(crop)) as image:
            rgb = image.convert("RGB")
            pixel_hash = sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
            identify(("pixel_sha256", pixel_hash), index, duplicate=True)
            hashes.append(difference_hash(rgb))
        file_hashes.append(
            {
                "index": index,
                "processed_rel_path": relative,
                "class_name": cls,
                "original_sha256": raw_hash,
                "crop_sha256": crop_hash.hexdigest(),
                "pixel_sha256": pixel_hash,
                "dhash": f"{hashes[-1]:016x}",
            }
        )
        if (identity := other_id(record)) is not None:
            source_parts = source.split("/")
            if cls == OTHER and len(source_parts) > 2 and source_parts[1] != identity:
                conflicts.append(
                    {
                        "left": index,
                        "right": index,
                        "evidence": "manifest_identity_vs_source_folder",
                    }
                )
            identify(("identity", cls, identity), index)
        if cls == MAIN:
            evidence = (
                {("verified_session", session_ids[index])}
                if index in session_ids
                else verified_dates(record, original, prefer_capture_metadata=True)
            )
            if evidence:
                confirmed.add(index)
                for kind, name in sorted(evidence):
                    identify((kind, cls, name), index)
                    group_evidence.append({"index": index, "type": kind, "value": name})

    for (left, right), decision in decisions.items():
        if decision == "duplicate":
            join(left, right, "reviewed_duplicate")
    for (left, right), decision in decisions.items():
        if decision == "different" and root(left) == root(right):
            # 다른 이미지라도 동일 촬영일·개체를 격리하는 그룹에는 함께 속할 수 있습니다.
            if any(pair["left"] == left and pair["right"] == right for pair in exact):
                conflicts.append({"left": left, "right": right, "evidence": "exact_vs_review"})
    confirmed_roots = {root(i) for i in confirmed}
    unknown_main = [
        i
        for i, row in enumerate(records)
        if row["class_name"] == MAIN and root(i) not in confirmed_roots
    ]
    unknown_other = [
        i for i, row in enumerate(records) if row["class_name"] == OTHER and not identities[root(i)]
    ]
    excluded = sorted(unknown_main + unknown_other)
    groups = {row["processed_rel_path"]: str(root(i)) for i, row in enumerate(records)}
    quarantined, resolved = {}, []
    if quarantine_other_copies:
        quarantined, resolved, conflicts = quarantine_other_copy_conflicts(
            records, groups, file_hashes, conflicts
        )
        excluded = sorted(set(excluded).union(quarantined))
    audit = {
        "class_counts": dict(Counter(r["class_name"] for r in records)),
        "group_counts": {
            cls: len({root(i) for i, r in enumerate(records) if r["class_name"] == cls})
            for cls in CLASSES
        },
        "verified_main_group_count": len(confirmed_roots),
        "unverified_main_indices": unknown_main,
        "unverified_other_indices": unknown_other,
        "group_evidence": group_evidence,
        "file_hashes": file_hashes,
        "exact_duplicates": exact,
        "exact_or_review_conflicts": conflicts,
        "other_copy_quarantine": {
            "indices": sorted(quarantined),
            "retained_split_groups_by_index": quarantined,
            "resolved_conflicts": resolved,
        },
        "reviewed_pairs": [
            {"left": a, "right": b, "decision": value} for (a, b), value in decisions.items()
        ],
    }
    return digest.hexdigest(), groups, excluded, hashes, audit


def similarity_reviews(
    records: list[dict],
    hashes: list[int],
    embeddings: np.ndarray,
    plan: dict,
    reviewed: list[dict] | None = None,
) -> dict[str, list[dict]]:
    """dHash 및 동결 DINOv2 코사인 기준별 검토 목록을 반환합니다."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if (
        len(records) != len(hashes)
        or len(records) != len(embeddings)
        or not np.isfinite(norms).all()
        or not np.all(norms > 0)
    ):
        raise ValueError("유사 이미지 감사에 유효한 행별 임베딩이 필요합니다.")
    unit = embeddings / norms
    cosine = unit @ unit.T
    decisions = {(row["left"], row["right"]): row["decision"] for row in reviewed or []}
    held, excluded = set(plan["holdout"]), set(plan["excluded"])
    fold_membership = {
        i: fold for fold, row in enumerate(plan["folds"], 1) for i in row["validation"]
    }
    reports: dict[str, list[dict]] = {f"dhash_le_{v}": [] for v in DHASH_THRESHOLDS}
    reports.update({f"cosine_ge_{v}": [] for v in COSINE_THRESHOLDS})
    for left, first in enumerate(records):
        for right in range(left + 1, len(records)):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            similarity = max(-1.0, min(1.0, float(cosine[left, right])))
            if distance > max(DHASH_THRESHOLDS) and similarity < min(COSINE_THRESHOLDS):
                continue
            second = records[right]
            first_path, second_path = first["processed_rel_path"], second["processed_rel_path"]
            row = {
                "left": first_path,
                "right": second_path,
                "left_class": first["class_name"],
                "right_class": second["class_name"],
                "dhash_distance": distance,
                "cosine": similarity,
                "review_decision": decisions.get((left, right), ""),
                "same_group": plan["groups"][first_path] == plan["groups"][second_path],
                "cross_holdout": (left in held) != (right in held)
                and left not in excluded
                and right not in excluded,
                "cross_fold": left in fold_membership
                and right in fold_membership
                and fold_membership[left] != fold_membership[right],
            }
            for limit in DHASH_THRESHOLDS:
                if distance <= limit:
                    reports[f"dhash_le_{limit}"].append(row)
            for limit in COSINE_THRESHOLDS:
                if similarity >= limit:
                    reports[f"cosine_ge_{limit}"].append(row)
    return reports
