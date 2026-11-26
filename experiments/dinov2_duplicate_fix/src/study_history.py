"""보관된 E00–E23의 설정을 검증하고 과거 지표와 새 평가를 구별합니다."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

from src.linear_probe import Hyperparameters
from src.train_e15 import E15
from src.tune_hyperparameters import read_json

LEGACY_IDS = tuple(f"E{index:02d}" for index in range(24))


def csv_parameters(row: dict) -> dict:
    """기존 CSV의 설정 열을 정확한 자료형으로 읽습니다."""
    values = {}
    for field in fields(Hyperparameters):
        value = row[field.name].strip()
        if field.name == "main_target_count" and value in ("", "None", "null"):
            values[field.name] = None
        elif field.name.endswith("_passes") or field.name == "main_target_count":
            values[field.name] = int(value)
        else:
            values[field.name] = float(value)
    return values


def restore_legacy(directory: Path) -> tuple[dict[str, Hyperparameters], dict]:
    """실제 결과 파일에서 24개 원래 설정을 복원하고 중복 조건의 수를 확인합니다."""
    mapping: dict[str, Hyperparameters] = {}
    provenance = {}
    for suffix in ("json", "csv"):
        path = directory / f"hyperparameter_results.{suffix}"
        if not path.is_file():
            continue
        provenance[path.name] = sha256(path.read_bytes()).hexdigest()
        if suffix == "json":
            rows = read_json(path)
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        seen = set()
        for row in rows:
            name = row["experiment"]
            if name not in LEGACY_IDS or name in seen:
                raise ValueError(f"{path.name}: 예상과 다른 실험 ID 또는 중복 행 {name}")
            seen.add(name)
            try:
                values = row["parameters"] if suffix == "json" else csv_parameters(row)
                if set(values) != {field.name for field in fields(Hyperparameters)}:
                    raise ValueError("필수 설정 열이 빠졌거나 추가 열이 있습니다.")
                params = Hyperparameters(**values)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name} {name}의 설정을 복원할 수 없습니다. {error}"
                ) from error
            if name in mapping and mapping[name] != params:
                raise ValueError(f"보관된 CSV와 JSON의 {name} 설정이 다릅니다.")
            mapping[name] = params
    missing = [name for name in LEGACY_IDS if name not in mapping]
    if missing:
        raise ValueError(f"실제 실험 설정 누락 ({', '.join(missing)}) · {directory}")
    if mapping["E00"] != Hyperparameters() or mapping["E15"] != E15:
        raise ValueError("보관된 E00 또는 E15의 설정이 기록된 기준과 일치하지 않습니다.")
    if len(set(mapping.values())) != 20:
        raise ValueError("보관된 24개 행에서 20개의 고유 조건을 확인할 수 없습니다.")
    return {name: mapping[name] for name in LEGACY_IDS}, provenance


def history_audit(directory: Path) -> dict:
    """기존 수치의 원본 파일에서 확인한 값과 아직 확인할 수 없는 항목을 기록합니다."""
    expected = {
        "E00_holdout_accuracy": 0.8088,
        "E15_holdout_accuracy": 0.9363,
        "E00_holdout_f1": 0.6061,
        "E15_holdout_f1": 0.8116,
        "E15_random_accuracy": 0.9333,
        "E15_strict_accuracy": 0.8667,
        "E15_random_hard_siamese_accuracy": 0.90,
        "E15_strict_hard_siamese_accuracy": 0.80,
    }
    values: dict[str, dict] = {}
    saved_checkpoint = directory.parent.parent / "checkpoints" / directory.name / "final_model.pt"
    checkpoint_sha = (
        sha256(saved_checkpoint.read_bytes()).hexdigest() if saved_checkpoint.is_file() else None
    )
    selected = None
    path = directory / "final_model_metrics.json"
    comparison = directory / "baseline_comparison.csv"
    candidates = []
    if path.is_file():
        saved_result = read_json(path)
        selected = saved_result.get("selected_experiment")
        for label, row in (saved_result.get("reserved_group_holdout") or {}).items():
            if label in ("baseline", "final") and row.get("metrics"):
                candidates.append((path, row["experiment"], row["metrics"]))
    if comparison.is_file():
        with comparison.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("split") == "reserved_group_holdout":
                    candidates.append((comparison, row["experiment"], row))
    for source, experiment, metrics in candidates:
        for metric in ("accuracy", "f1"):
            key = f"{experiment}_holdout_{metric}"
            if key not in expected or metrics.get(metric) is None:
                continue
            seen = float(metrics[metric])
            entry = values.setdefault(key, {"values": []})
            entry["values"].append({"source": str(source), "value": seen})
    for folder in (
        directory / "sample_tests",
        directory.parent.parent / "graphs" / directory.name,
    ):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.json")):
            try:
                content = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(content, dict) or not content.get("samples"):
                continue
            if content.get("sample_source", "project_dataset") != "project_dataset":
                continue
            if content.get("mode") == "strict_holdout":
                mode = "strict"
            elif content.get("mode") == "full_checkpoint":
                mode = "random"
            else:
                continue
            rows = content["samples"]
            if len(rows) != 30 or not all("correct" in row for row in rows):
                continue
            hard = [
                row
                for row in rows
                if row.get("true_class")
                in (
                    "main_siamese_cat",
                    "other_siamese_cat",
                )
            ]
            if len(hard) != 20:
                continue
            identity_proven = content.get("experiment") == "E15" or bool(
                selected == "E15"
                and checkpoint_sha
                and content.get("checkpoint_sha256") == checkpoint_sha
            )
            for suffix, selected in (("accuracy", rows), ("hard_siamese_accuracy", hard)):
                key = f"E15_{mode}_{suffix}"
                values.setdefault(key, {"values": []})["values"].append(
                    {
                        "source": str(path),
                        "value": sum(row["correct"] for row in selected) / len(selected),
                        "checkpoint_identity_proven": identity_proven,
                    }
                )
    return {
        key: {
            "reference": reference,
            "observations": (observations := values.get(key, {}).get("values", [])),
            "matches_rounded_reference": (
                abs(observations[0]["value"] - reference) <= 0.000051
                if observations
                and all(
                    abs(row["value"] - observations[0]["value"]) <= 0.000051
                    and row.get("checkpoint_identity_proven", True)
                    for row in observations
                )
                else None
            ),
        }
        for key, reference in expected.items()
    }


def legacy_rows(mapping: dict[str, Hyperparameters]) -> list[dict]:
    """복원한 ID와 조건, 첫 동일 조건의 ID를 저장합니다."""
    first = {}
    rows = []
    for name, params in mapping.items():
        reference = first.setdefault(params, name)
        rows.append(
            {
                "experiment": name,
                **asdict(params),
                "duplicate_of": (reference if reference != name else None),
            }
        )
    return rows
