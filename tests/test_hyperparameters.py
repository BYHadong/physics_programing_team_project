"""합성 이미지로 실험·Strict Holdout·추론 인터페이스를 검증합니다."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src import linear_probe, tune_hyperparameters
from src.analyze_results import generate_graphs
from src.config import BACKBONE_INPUT_SIZE, DINO_BACKBONE_ID
from src.dataset import build_transforms
from src.experiment_splits import (
    MAIN,
    OTHER,
    audit_dataset,
    load_manifest,
    source_index,
    split_plan,
    strict_training_indices,
)
from src.linear_probe import Hyperparameters, evaluate, thresholds_from_validation, training_recipe
from src.predict import predict_image
from src.test_random_samples import fit_holdout_checkpoint, run_random_sample_test, save_result_grid
from src.train_baseline import compute_evaluation_metrics
from src.tune_hyperparameters import read_json, write_json

ETC = "etc_cats"


class TinyBackbone(torch.nn.Module):
    """다운로드 없이 정규화 가능한 간단한 테스트 특징을 생성합니다."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB 평균과 중앙 왼쪽 평균 및 상수 특징을 반환합니다."""
        return torch.cat(
            (
                x.mean((2, 3)),
                x[:, :, :, : x.shape[-1] // 2].mean((2, 3)),
                torch.ones_like(x[:, :1, 0, 0]),
            ),
            dim=1,
        )


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """환경 이동을 시뮬레이션하는 이전 Windows 경로의 합성 Manifest를 만듭니다."""
    source, processed = tmp_path / "raw", tmp_path / "processed"
    records = []
    rng = np.random.default_rng(42)
    for cls_index, cls in enumerate((MAIN, OTHER, ETC)):
        for group in range(7):
            for index in range(2):
                filename = f"IMG_202611{group + 1:02d}_{index}.png"
                individual = f"cat_{group}" if cls == OTHER else cls
                relative = Path(cls) / individual / filename
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = rng.integers(0, 96, (24, 24, 3), dtype=np.uint8)
                image[:, :, cls_index] += 120
                Image.fromarray(image).save(path)
                crop = processed / relative
                crop.parent.mkdir(parents=True, exist_ok=True)
                crop.write_bytes(path.read_bytes())
                records.append(
                    {
                        "source_path": "D:\\backup\\" + str(relative).replace("/", "\\"),
                        "processed_rel_path": relative.as_posix(),
                        "class_name": cls,
                        "individual_id": individual,
                        "yolo_detected": True,
                    }
                )
    manifest = processed / "manifest.json"
    write_json(manifest, records)
    return source, processed, manifest


def test_transform_options() -> None:
    """회전 각도와 확률이 학습용 변환에 반영되고 검증 변환은 그대로인지 확인합니다."""
    for degrees, probability in [(5, 0), (15, 0.25), (30, 1)]:
        result = build_transforms(16, rotation_degrees=degrees, hflip_probability=probability)
        rotation = next(op for op in result.transforms if isinstance(op, transforms.RandomRotation))
        flip = next(
            op for op in result.transforms if isinstance(op, transforms.RandomHorizontalFlip)
        )
        assert rotation.degrees == [-degrees, degrees]
        assert flip.p == probability
    evaluation = build_transforms(16, False, rotation_degrees=30, hflip_probability=1)
    assert not any(
        isinstance(op, (transforms.RandomHorizontalFlip, transforms.RandomRotation))
        for op in evaluation.transforms
    )


def test_augmentation_recipe() -> None:
    """대칭 7회 및 목표 장수 방식에서 원본 번호와 추가 증강 횟수를 확인합니다."""
    records = [{"class_name": MAIN}, {"class_name": MAIN}, {"class_name": OTHER}]
    train = [0, 2]
    symmetric = training_recipe(records, train, Hyperparameters())
    assert len(symmetric) == 16
    assert all(indices == ([0] if label else [2]) for indices, _, label in symmetric)
    params = Hyperparameters(main_target_count=3, neg_aug_passes=1)
    asymmetric = training_recipe(records, train, params)
    assert asymmetric == [([0], 0, 1), ([0], 1, 1), ([0], 2, 1), ([2], 0, 0), ([2], 1, 0)]
    richer = training_recipe(records, train, Hyperparameters(main_target_count=4))
    assert richer[:3] == asymmetric[:3]


def test_thresholds_match_readme_definition() -> None:
    """풀링 AUC, F1 임계값, OTHER 분모 및 FPR 제약을 검증합니다."""
    classes = [MAIN, OTHER, ETC, MAIN, OTHER, ETC]
    scores = np.array([0.91, 0.85, 0.82, 0.8, 0.6, 0.4])
    policies = thresholds_from_validation(classes, scores)
    legacy_classes = sorted(set(classes))
    legacy = compute_evaluation_metrics(
        np.asarray(classes) == MAIN,
        np.array([legacy_classes.index(cls) for cls in classes]),
        legacy_classes,
        scores,
    )
    chosen = policies["max_f1"]
    assert chosen["threshold"] == legacy["threshold"]
    assert round(chosen["hard_siamese_roc_auc"], 4) == legacy["hard_siamese_roc_auc"]
    assert round(chosen["overall_roc_auc"], 4) == legacy["overall_roc_auc"]
    assert policies["target_siamese_fpr"]["unseen_siamese_fpr"] == 0
    assert policies["target_siamese_fpr"]["recall"] == 0.5
    assert evaluate(classes, scores, 0.8)["confusion_matrix"] == [[2, 2], [0, 2]]


def test_capture_groups_and_source_paths(synthetic_data: tuple[Path, Path, Path]) -> None:
    """촬영 날짜·동일 이미지·개체를 고정해 분할 및 Strict Holdout 누수를 막습니다."""
    source, processed, manifest = synthetic_data
    records = load_manifest(manifest)
    copy_path = processed / records[2]["processed_rel_path"]
    from_first_group = processed / records[0]["processed_rel_path"]
    copy_path.write_bytes(from_first_group.read_bytes())
    _, groups, audit = audit_dataset(records, processed, source_index(source), "synthetic")
    plan = split_plan(records, groups, audit)
    assert audit["main_with_capture_evidence"] == 14
    assert groups[records[0]["processed_rel_path"]] == groups[records[3]["processed_rel_path"]]
    assert len(plan["folds"]) == 5 and plan["holdout"]
    assert not (set(plan["development"]) & set(plan["holdout"]))
    assert sorted(i for fold in plan["folds"] for i in fold["validation"]) == sorted(
        plan["development"]
    )
    for fold in plan["folds"]:
        left = {groups[records[i]["processed_rel_path"]] for i in fold["train"]}
        right = {groups[records[i]["processed_rel_path"]] for i in fold["validation"]}
        assert left.isdisjoint(right)
        assert (set(fold["train"]) | set(fold["validation"])) == set(plan["development"])
    main_path = source / records[0]["processed_rel_path"]
    other_path = source / records[14]["processed_rel_path"]
    keep = strict_training_indices(
        records, [(main_path, MAIN, MAIN), (other_path, OTHER, "cat_0")], groups
    )
    assert set(keep).isdisjoint({0, 1, 2, 3, 14, 15})
    with pytest.raises(ValueError, match="대조"):
        strict_training_indices(records, [(source / MAIN / "missing.jpg", MAIN, MAIN)], groups)


def test_pipeline_resume_and_inference(
    synthetic_data: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """합성 데이터로 기준·탐색·재개·출력·Strict Holdout 및 체크포인트 추론을 실행합니다."""
    source, processed, manifest = synthetic_data
    root = tmp_path / "results"
    candidate_config = tmp_path / "candidates.json"
    write_json(
        candidate_config,
        {
            "c": [0.1],
            "main_target_count": [24],
            "neg_aug_passes": [1],
            "rotation_degrees": [5],
            "hflip_probability": [1.0],
        },
    )
    model = TinyBackbone().eval()
    calls = []

    def fake_load(model_id: str, device: torch.device) -> tuple[torch.nn.Module, int]:
        """파이프라인이 고정된 백본 ID 하나만 요청하는지 확인합니다."""
        calls.append(model_id)
        return model, BACKBONE_INPUT_SIZE

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tune_hyperparameters, "load_backbone", fake_load)
    monkeypatch.setattr("src.predict.load_backbone", fake_load)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = argparse.Namespace(
            search_space=candidate_config,
            run_id="synthetic",
            manifest=manifest,
            processed=processed,
            source=source,
            output=root,
            threshold_policy="max_f1",
            through="baseline",
        )
        tune_hyperparameters.run(args)
        baseline_path = root / "experiments/synthetic/E00/result.json"
        baseline_before = baseline_path.read_bytes()
        args.through = "final"
        tune_hyperparameters.run(args)
        tune_hyperparameters.run(args)
        assert baseline_path.read_bytes() == baseline_before
        assert set(calls) == {DINO_BACKBONE_ID}
        results = root / "results/synthetic"
        assert {p.name for p in results.iterdir()} >= {
            "baseline_comparison.csv",
            "best_hyperparameters.json",
            "experiment_config.json",
            "final_model_metrics.json",
            "fold_results.csv",
            "hyperparameter_results.csv",
            "hyperparameter_results.json",
            "environment_versions.txt",
            "fold_indices.json",
        }
        trials = read_json(results / "hyperparameter_results.json")
        assert [row["field"] for row in trials[:6]] == [
            None,
            "c",
            "neg_aug_passes",
            "main_target_count",
            "rotation_degrees",
            "hflip_probability",
        ]
        final = read_json(results / "final_model_metrics.json")
        assert (
            final["reserved_group_holdout"]["final"]["metrics"]["hard_siamese_roc_auc"] is not None
        )
        graphs = root / "graphs/synthetic"
        generate_graphs(results, graphs)
        generate_graphs(results, graphs)
        assert len(list(graphs.glob("*.png"))) == 10

        checkpoint = root / "checkpoints/synthetic/final_model.pt"
        saved = torch.load(checkpoint, weights_only=True)
        assert (saved["method"], saved["backbone_id"], saved["backbone_frozen"]) == (
            "linear_probe",
            DINO_BACKBONE_ID,
            True,
        )
        manifest_records = load_manifest(manifest)
        sample_index = saved["holdout_indices"][0]
        record = manifest_records[sample_index]
        image_path = source / record["processed_rel_path"]
        predicted = predict_image(
            image_path, checkpoint, processed_record=record, processed_dir=processed
        )
        holdout = final["reserved_group_holdout"]["final"]["predictions"]
        assert predicted["score"] == pytest.approx(holdout["scores"][0], abs=1e-6)
        assert predicted["threshold"] == saved["threshold"]

        chosen_params = Hyperparameters(
            c=0.05,
            main_target_count=18,
            neg_aug_passes=1,
            rotation_degrees=5,
            hflip_probability=0.25,
        )
        fitted = []
        real_fit = linear_probe.fit_probe

        def checked_fit(cache: object, indices: list[int], params: Hyperparameters) -> tuple:
            """Fit마다 하이퍼파라미터와 학습 목록을 관찰합니다."""
            clf, prototype, counts = real_fit(cache, indices, params)
            fitted.append((indices, params, clf.C))
            return clf, prototype, counts

        monkeypatch.setattr(linear_probe, "fit_probe", checked_fit)
        monkeypatch.setattr("src.test_random_samples.fit_probe", checked_fit)
        active = fit_holdout_checkpoint(
            [(image_path, record["class_name"], record["individual_id"])],
            {**saved, "hyperparameters": asdict(chosen_params), "threshold": -10.0},
            torch.device("cpu"),
            model,
            manifest,
            processed,
            source_dir=source,
            cache_dir=root / "embeddings",
        )
        assert active["threshold"] != -10.0
        assert len(fitted) == 6
        assert all(
            sample_index not in indices and params == chosen_params and c == 0.05
            for indices, params, c in fitted
        )
        assert active["strict_training_counts"]["positive_training"] == 18
        for strict in (False, True):
            target = tmp_path / f"samples_{strict}.json"
            grid = tmp_path / f"samples_{strict}.png"
            report = run_random_sample_test(
                source,
                checkpoint,
                1,
                42,
                strict,
                True,
                target,
                grid,
                processed_dir=processed,
                cache_dir=root / "embeddings",
                reuse_processed=True,
                reserved_cohort=True,
            )
            assert target.exists() and grid.read_bytes().startswith(b"\x89PNG")
            assert report["total_samples"] == 3
            if strict:
                assert report["strict_holdout"]["checkpoint_tuning_overlap"] is False

        extra = source / "main_siamese_cat" / "new_photo.jpg"
        extra.write_bytes(image_path.read_bytes())
        report = run_random_sample_test(
            source_dir=source,
            checkpoint_path=checkpoint,
            n_per_class=len(manifest_records),
            seed=42,
            save_grid=False,
            output_json=tmp_path / "manifest_only_samples.json",
            processed_dir=processed,
            reuse_processed=True,
        )
        assert report["total_samples"] == len(manifest_records)
        assert extra.is_file()
    finally:
        torch.set_num_threads(old_threads)


def test_sample_grid_keeps_class_rows_with_unequal_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holdout의 클래스별 사진 수가 달라도 같은 줄에 섞이지 않는지 확인합니다."""
    from matplotlib.axes import Axes

    labels = []
    real_ylabel = Axes.set_ylabel

    def record_label(axis: Axes, label: str, **kwargs: object) -> object:
        """그림의 클래스 줄 라벨을 기록하고 실제 라벨을 설정합니다."""
        labels.append(label)
        return real_ylabel(axis, label, **kwargs)

    monkeypatch.setattr(Axes, "set_ylabel", record_label)
    results = [
        {"true_class": cls, "pred_is_main": cls == MAIN, "correct": True, "probability": 0.8}
        for cls in (OTHER, MAIN, "etc_cats", OTHER, MAIN, MAIN)
    ]
    output = tmp_path / "sample_grid.png"
    save_result_grid(results, output)
    assert labels == ["Main 사진", "Other 사진", "ETC 사진"]
    assert output.read_bytes().startswith(b"\x89PNG")


def test_train_e15_only_reuses_validation_and_checkpoint(
    synthetic_data: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E15의 고정 설정으로만 학습하고 완료 후 같은 실행을 재사용합니다."""
    from src import train_e15

    source, processed, _ = synthetic_data
    output = tmp_path / "e15"

    class SyntheticCache:
        """각 클래스의 구분 가능한 고정 특징을 모의로 반환합니다."""

        def __init__(self, root: Path, records: list[dict], *args: object) -> None:
            self.records = records
            self.key = "synthetic-cache"
            self.identity = {
                "weights_sha256": "synthetic-backbone",
                "execution_environment": {"device": "cpu"},
            }

        def get(self, indices: list[int], *augmentation: object) -> np.ndarray:
            """검증과 학습에 요청된 순서대로 모의 특징을 반환합니다."""
            labels = [int(self.records[i]["class_name"] == MAIN) for i in indices]
            return np.eye(2, dtype=np.float32)[labels]

    original_fit = linear_probe.fit_probe
    fits = []

    def count_fit(cache: SyntheticCache, indices: list[int], params: Hyperparameters) -> tuple:
        """학습에 들어간 인덱스와 하이퍼파라미터를 기록합니다."""
        fits.append((indices, params))
        return original_fit(cache, indices, params)

    def fake_backbone(model_id: str, device: torch.device) -> tuple[torch.nn.Module, int]:
        """다운로드 없이 E15에 요청된 백본 이름을 확인합니다."""
        assert model_id == DINO_BACKBONE_ID
        return TinyBackbone().eval(), BACKBONE_INPUT_SIZE

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_e15, "load_backbone", fake_backbone)
    monkeypatch.setattr(train_e15, "EmbeddingCache", SyntheticCache)
    monkeypatch.setattr(train_e15, "fit_probe", count_fit)
    monkeypatch.setattr(linear_probe, "fit_probe", count_fit)
    args = argparse.Namespace(
        source=source,
        processed=processed,
        output=output,
        run_id="fixed-e15",
        threshold_policy="max_f1",
    )

    checkpoint = train_e15.run(args)
    report_dir = output / "results" / "fixed-e15"
    report = read_json(report_dir / "final_model_metrics.json")
    validation = read_json(report_dir / "e15_validation.json")
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["experiment"] == report["selected_experiment"] == "E15"
    assert saved["hyperparameters"] == asdict(train_e15.E15)
    assert saved["threshold"] == validation["metrics"]["threshold"]
    assert saved["method"] == "linear_probe"
    assert saved["backbone_id"] == DINO_BACKBONE_ID
    assert saved["backbone_frozen"] is True
    assert report["holdout_index_count"] > 0
    assert len(fits) == 6
    assert all(params == train_e15.E15 for _, params in fits)
    assert all(not set(indices) & set(saved["holdout_indices"]) for indices, _ in fits)
    assert len(validation["fold_results"]) == 5

    assert train_e15.run(args) == checkpoint
    assert len(fits) == 6


def test_adaptive_group_cv_has_both_classes_and_isolated_holdout() -> None:
    """불균형 그룹에서 양성 빈 Fold나 학습·검증·테스트 그룹 중복을 방지합니다."""
    from src.group_validation import check_partition, complete_fold, make_plan

    sizes = {MAIN: (47, 10, 30), OTHER: (16, 8, 4, 2, 2, 2), ETC: (3, 3, 3)}
    records, groups = [], {}
    for cls, counts in sizes.items():
        for number, count in enumerate(counts):
            for index in range(count):
                path = f"{cls}/{number}/{index}.png"
                records.append({"class_name": cls, "processed_rel_path": path})
                groups[path] = f"{cls}/{number}"
    old_missing = list(range(57, 87))
    plan = make_plan(records, groups, old_missing)
    assert len(plan["folds"]) == 2 and not plan["holdout"]
    assert {row["method"] for row in plan["splitter_comparison"]} >= {
        "GroupKFold",
        "StratifiedGroupKFold",
    }
    assert sorted(sum(i < 57 for i in f["validation"]) for f in plan["folds"]) == [10, 47]
    assert not make_plan(records, groups, list(range(47, 87)))["folds"]
    new_plan = make_plan(records, groups, [], prior=[0, 87])
    assert new_plan["holdout"] and "격리" in new_plan["prior_holdout_status"]
    assert {0, 87} <= set(new_plan["holdout"])
    assert not set(new_plan["development"]).intersection(range(47))
    for fold in new_plan["folds"]:
        assert all(complete_fold(records, part) for part in fold.values())
        check_partition(records, groups, fold["train"], fold["validation"], new_plan["holdout"])
    with pytest.raises(ValueError, match="동일 그룹"):
        check_partition(records, groups, [0], [47, 87], [1])


def test_similarity_review_keeps_individuals_separate(
    synthetic_data: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """확인되지 않은 촬영 그룹과 다른 개체를 유사도만으로 합치지 않습니다."""
    from src.group_validation import make_plan
    from src.study_audit import scan_dataset, similarity_reviews

    source, processed, manifest = synthetic_data
    records = load_manifest(manifest)
    old = source / records[0]["processed_rel_path"]
    original = old.with_name("no_date.png")
    old.rename(original)
    records[0]["source_path"] = str(original)
    records[18]["individual_id"] = "unknown"
    target, unrelated = records[14]["processed_rel_path"], records[16]["processed_rel_path"]
    review = tmp_path / "review.csv"
    review.write_text(f"left,right,decision\n{target},{unrelated},different\n")
    _, groups, excluded, hashes, audit = scan_dataset(
        records, processed, source_index(source), "hash", review=review
    )
    assert excluded == [0, 18]
    assert audit["verified_main_group_count"] == 7
    assert groups[target] != groups[unrelated]
    plan = make_plan(records, groups, excluded)
    embeddings = np.ones((len(records), 2))
    lists = similarity_reviews(records, hashes, embeddings, plan, audit["reviewed_pairs"])
    pair = next(
        r for r in lists["cosine_ge_0.995"] if (r["left"], r["right"]) == (target, unrelated)
    )
    assert pair["review_decision"] == "different" and not pair["same_group"]
    # 동일 픽셀이 다른 개체로 등록된 경우 자동으로 개체를 합치지 않습니다.
    (processed / unrelated).write_bytes((processed / target).read_bytes())
    _, groups, _, _, audit = scan_dataset(
        records, processed, source_index(source), "hash", review=review
    )
    assert groups[target] != groups[unrelated]
    assert audit["exact_or_review_conflicts"]


def test_generalization_pipeline_and_test_modes(
    synthetic_data: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """새 튜닝·해상도·재개·테스트 경로를 소규모 합성 데이터로 실행합니다."""
    from dataclasses import replace

    from src import generalization_study as study
    from src import test_random_samples, test_study_model
    from src.embedding_cache import EmbeddingCache
    from src.experiment_splits import source_key
    from src.group_validation import check_partition, group_ids
    from src.plot_generalization import generate

    source, processed, manifest = synthetic_data
    root = tmp_path / "generalization"
    model = TinyBackbone().eval()
    fits = []
    augmented = set()
    real_fit = study.fit_probe

    def fake_load(model_id: str, _: torch.device) -> tuple[torch.nn.Module, int]:
        """모델 ID와 임베딩 형식을 확인할 수 있게 가벼운 모델을 반환합니다."""
        assert model_id == DINO_BACKBONE_ID
        return model, 224

    def fake_extract(
        cache: EmbeddingCache, indices: list[int], aug_pass: int, *_: object
    ) -> np.ndarray:
        """원본 인덱스별 특징으로 실제 캐시 저장을 빠르게 실행합니다."""
        if aug_pass:
            augmented.update(indices)
        features = np.random.default_rng(42).standard_normal((len(cache.records), 7))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        return features[indices].astype(np.float32)

    def counting_fit(*args: object) -> tuple:
        """이미 계산된 후보가 재학습되지 않는지 학습 요청 수를 센 결과를 반환합니다."""
        fits.append(1)
        return real_fit(*args)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(study, "load_backbone", fake_load)
    monkeypatch.setattr("src.predict.load_backbone", fake_load)
    monkeypatch.setattr(EmbeddingCache, "_extract", fake_extract)
    monkeypatch.setattr(study, "SEARCH", (("c", (0.005, 0.01)),))
    from src import study_history

    old_e15 = replace(study_history.E15, main_target_count=26)
    monkeypatch.setattr(study_history, "E15", old_e15)
    legacy = tmp_path / "legacy"
    setups = [
        Hyperparameters()
        if i == 0
        else old_e15
        if i == 15
        else replace(old_e15, c=0.003 * i)
        if i < 20
        else Hyperparameters()
        for i in range(24)
    ]
    write_json(
        legacy / "hyperparameter_results.json",
        [
            {"experiment": f"E{i:02}", "parameters": asdict(params)}
            for i, params in enumerate(setups)
        ],
    )
    monkeypatch.setattr(study, "fit_probe", counting_fit)
    args = argparse.Namespace(
        run_id="synth-study",
        source=source,
        output=root,
        processed=processed,
        threshold_policy="max_f1",
        prior_split=None,
        sessions=None,
        review=None,
        audit_only=False,
        legacy_results=legacy,
        input_sizes=(224, 336, 448),
        additional_budget=2,
        fpr_targets=(0.01, 0.05, 0.10),
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        study.run(args)
        first_fit = len(fits)
        assert first_fit > 0
        study.run(args)
        assert len(fits) == first_fit
        result = root / "results" / args.run_id
        checkpoint = root / "checkpoints" / args.run_id / "final_model.pt"
        saved = torch.load(checkpoint, weights_only=True)
        folds = read_json(result / "fold_indices.json")
        report = read_json(result / "final_model_metrics.json")
        assert len(saved["holdout_indices"]) == report["holdout_index_count"] > 0
        assert augmented.isdisjoint(saved["holdout_indices"])
        assert set(report["comparisons"]["e15"]["oof_policies"]) >= {
            *study.POLICIES,
            "fpr_le_0.01",
            "fpr_le_0.05",
            "fpr_le_0.1",
            "fixed_0_5",
        }
        assert len((result / "legacy_reevaluation.csv").read_text().splitlines()) == 25
        timing = read_json(result / "timing.json")
        assert timing["cumulative_unique_cv"] >= 22
        assert read_json(result / "data_ablation.json")["reduced_folds"]
        assert (result / "baseline_comparison.csv").is_file()
        assert (result / "resolution_comparison.csv").is_file()
        records = load_manifest(manifest)
        for fold in folds["folds"]:
            check_partition(
                records, saved["study_groups"], *fold.values(), saved["holdout_indices"]
            )
        graphs = root / "graphs" / args.run_id
        generate(result, graphs)
        generate(result, graphs)
        assert (graphs / "overfitting_gap.png").read_bytes()[:4] == b"\x89PNG"

        sample = []
        for cls in (MAIN, OTHER):
            row = next(
                records[i] for i in saved["training_indices"] if records[i]["class_name"] == cls
            )
            sample.append((source / row["processed_rel_path"], cls, row["individual_id"]))
        strict = fit_holdout_checkpoint(
            sample,
            saved,
            torch.device("cpu"),
            model,
            manifest,
            processed,
            source_dir=source,
            cache_dir=root / "embeddings",
        )
        sample_keys = {source_key(path, cls) for path, cls, _ in sample}
        sampled_indices = [
            i
            for i, r in enumerate(records)
            if source_key(r["source_path"], r["class_name"]) in sample_keys
        ]
        for fold in strict["strict_fold_indices"]:
            check_partition(records, saved["study_groups"], *fold.values(), sampled_indices)
        strict_train = strict["strict_training_indices"]
        assert group_ids(records, saved["study_groups"], strict_train).isdisjoint(
            group_ids(records, saved["study_groups"], saved["holdout_indices"])
        )
        assert strict["hyperparameters"] == saved["hyperparameters"]
        cached = fit_holdout_checkpoint(
            sample,
            saved,
            torch.device("cpu"),
            model,
            manifest,
            processed,
            source_dir=source,
            cache_dir=root / "embeddings",
        )
        assert cached["strict_cache_reused"] and cached["threshold"] == strict["threshold"]

        def unexpected_refit(**_: object) -> None:
            """저장 모델 평가에서 재학습을 요청하면 실패합니다."""
            pytest.fail("고정 모델 추론에서 재학습이 호출되었습니다.")

        monkeypatch.setattr(test_random_samples, "fit_holdout_checkpoint", unexpected_refit)
        monkeypatch.setattr(test_random_samples, "save_result_grid", lambda *_: None)
        args.checkpoint, args.test_seed, args.n_per_class = checkpoint, 123, 1
        args.strict_development = False
        for mode in ("random", "strict", "reserved-all", "reproducibility"):
            args.mode = mode
            tested = read_json(test_study_model.run_test(args) / "results.json")
            if mode == "reproducibility":
                assert tested["same_sample_order"]
            else:
                assert not tested["refitted"]
                assert tested["metrics"]["hard_siamese_accuracy"] is not None
                if mode == "reserved-all":
                    assert tested["total_samples"] == len(saved["holdout_indices"])
    finally:
        torch.set_num_threads(previous_threads)


def test_other_copy_quarantine_keeps_linked_individuals_out_of_development(
    synthetic_data: tuple[Path, Path, Path],
) -> None:
    """서로 다른 개체 폴더의 바이트 동일 복제본을 제외하고 예전 Holdout 연결을 유지합니다."""
    import json
    from hashlib import sha256

    from src.group_validation import check_partition, make_plan
    from src.study_audit import scan_dataset
    from src.test_random_samples import study_sample_records

    source, processed, manifest = synthetic_data
    records = load_manifest(manifest)
    first_copy_index = len(records)
    ids = [("cat_0", "cat_1")] * 4 + [("cat_1", "cat_2")]
    for n, pair in enumerate(ids):
        for individual in pair:
            relative = Path(OTHER) / individual / f"copy_{n}.png"
            raw = source / relative
            Image.new("RGB", (24, 24), (0, 180, n)).save(raw)
            (processed / relative).write_bytes(raw.read_bytes())
            records.append(
                {
                    "source_path": str(raw),
                    "processed_rel_path": relative.as_posix(),
                    "class_name": OTHER,
                    "individual_id": individual,
                }
            )

    originals = source_index(source)
    digest, _, before, _, baseline = scan_dataset(records, processed, originals, "original")
    after_digest, groups, excluded, _, audit = scan_dataset(
        records, processed, originals, "original", quarantine_other_copies=True
    )
    quarantined = list(range(first_copy_index, len(records)))
    assert not before and len(baseline["exact_or_review_conflicts"]) == 15
    assert after_digest == digest
    assert excluded == quarantined
    assert not audit["exact_or_review_conflicts"]
    info = audit["other_copy_quarantine"]
    assert info["indices"] == quarantined and len(info["resolved_conflicts"]) == 15
    retained = list(range(14, 20))  # 처음 세 개체의 복제본을 제외한 사진
    assert len({groups[records[i]["processed_rel_path"]] for i in retained}) == 1

    prior = [0, quarantined[0]]
    plan = make_plan(
        records,
        groups,
        excluded,
        prior=prior,
        prior_related_groups={info["retained_split_groups_by_index"][quarantined[0]]},
    )
    assert set(retained).issubset(plan["holdout"])
    assert set(plan["development"]).isdisjoint(retained)
    assert set(plan["holdout"] + plan["development"]).isdisjoint(quarantined)
    assert plan["folds"] and plan["prior_holdout_overlap"] == 1
    for fold in plan["folds"]:
        check_partition(records, groups, fold["train"], fold["validation"], plan["holdout"])
    manifest.write_text(json.dumps(records), encoding="utf-8")
    checkpoint = {
        "study_quarantined": quarantined,
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
    }
    assert study_sample_records(records, manifest, checkpoint) == records[:first_copy_index]
    with pytest.raises(ValueError, match="Manifest"):
        study_sample_records(records, manifest, {**checkpoint, "manifest_sha256": "other"})


def test_other_copy_policy_keeps_cross_class_or_crop_only_conflicts(
    synthetic_data: tuple[Path, Path, Path],
) -> None:
    """클래스 간 복제나 크롭만 같은 경우 기존 충돌 중단 조건을 유지합니다."""
    from src.study_audit import scan_dataset

    source, processed, manifest = synthetic_data
    records = load_manifest(manifest)
    main, other, second, third = (records[i]["processed_rel_path"] for i in (0, 14, 16, 18))
    for folder in (source, processed):
        (folder / other).write_bytes((folder / main).read_bytes())
    (processed / third).write_bytes((processed / second).read_bytes())
    _, _, excluded, _, audit = scan_dataset(
        records, processed, source_index(source), "hash", quarantine_other_copies=True
    )
    assert not excluded and not audit["other_copy_quarantine"]["indices"]
    conflicts = audit["exact_or_review_conflicts"]
    assert {tuple((c["left"], c["right"])) for c in conflicts} == {(0, 14), (16, 18)}
    assert len(conflicts) == 5
