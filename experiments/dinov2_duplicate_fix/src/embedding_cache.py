"""동결 백본의 클린 및 결정적 증강 임베딩을 재사용합니다."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
import torch

from src.config import (
    COCO_CAT_CLASS_ID,
    CROP_MARGIN_RATIO,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSED_SIZE,
    RANDOM_SEED,
    YOLO_CONF_THRESHOLD,
    YOLO_WEIGHTS_PATH,
)
from src.dataset import ManifestCatDataset, build_transforms
from src.train_baseline import extract_embeddings

CACHE_SCHEMA = 1
# 캐시 I/O와 통계만 바뀌기 전의 출력 호환 구현입니다.
LEGACY_CACHE_SOURCE = "8dd0ca0df63154fdf38f03f7d6ed1a239ce51cadfd13e6352938ad39b06587e6"
PACKAGES = (
    "numpy",
    "Pillow",
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "scipy",
    "ultralytics",
    "matplotlib",
)


def fingerprint(value: object) -> str:
    """JSON 직렬화 가능한 값에서 순서 독립적인 SHA-256 키를 반환합니다."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def environment_versions(device: torch.device) -> dict:
    """실험 재현에 필요한 Python, 패키지 및 연산 장치 버전을 반환합니다."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        versions["gpu"] = torch.cuda.get_device_name(device)
        versions["cuda_capability"] = list(torch.cuda.get_device_capability(device))
        versions["cudnn"] = torch.backends.cudnn.version()
    return versions


def backbone_digest(model: torch.nn.Module) -> str:
    """로드된 백본 가중치와 버퍼의 실제 내용으로 키를 계산합니다."""
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def deterministic_seed(index: int, aug_pass: int, purpose: str) -> int:
    """시드 42에서 이미지·추가 패스·난수 용도별 시드를 파생합니다."""
    key = f"{RANDOM_SEED}:{index}:{aug_pass}:{purpose}"
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


@contextmanager
def preserve_rng() -> Iterator[None]:
    """특징 추출 전후 Python 및 CPU Torch 난수 상태를 보존합니다."""
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


class SeededDataset(ManifestCatDataset):
    """각 원본의 증강 난수를 Fold나 배치 순서에 독립적으로 적용합니다."""

    def __init__(
        self, records: list[dict], indices: list[int], aug_pass: int, **kwargs: object
    ) -> None:
        """원본 인덱스와 증강 패스를 보존하여 데이터셋을 초기화합니다."""
        super().__init__(records, **kwargs)
        self.indices = indices
        self.aug_pass = aug_pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, str, str]:
        """재현 가능한 초기 상태에서 변환한 이미지와 라벨을 반환합니다."""
        seed = deterministic_seed(self.indices[idx], self.aug_pass, "transforms")
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        return super().__getitem__(idx)


class EmbeddingCache:
    """전처리·가중치·환경별 임베딩을 패스와 실제 변환으로 구분해 저장합니다."""

    def __init__(
        self,
        root: Path,
        records: list[dict],
        processed_dir: Path,
        dataset_digest: str,
        backbone_id: str,
        model: torch.nn.Module,
        image_size: int,
        batch_size: int,
        device: torch.device,
        *,
        photo_hashes: list[tuple[str, str]] | None = None,
    ) -> None:
        """데이터와 백본 설정에 따른 캐시 이름 공간과 빈 메모리 색인을 만듭니다."""
        versions = environment_versions(device)
        self.identity = {
            "schema": CACHE_SCHEMA,
            "dataset_and_manifest": dataset_digest,
            "backbone_id": backbone_id,
            "weights_sha256": backbone_digest(model),
            "image_size": image_size,
            "batch_size": batch_size,
            "random_seed": RANDOM_SEED,
            "preprocessing": {
                "method": "EXIF-RGB-YOLOv8n-crop-letterbox",
                "size": PREPROCESSED_SIZE,
                "margin": CROP_MARGIN_RATIO,
                "confidence": YOLO_CONF_THRESHOLD,
                "detector_weights": YOLO_WEIGHTS_PATH.name,
                "detector_class": COCO_CAT_CLASS_ID,
                "normalization": [IMAGENET_MEAN, IMAGENET_STD],
            },
            "execution_environment": versions,
            "source_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "dataset.py",
                    "train_baseline.py",
                    "embedding_cache.py",
                    "preprocess.py",
                )
            },
        }
        self.key = fingerprint(self.identity)
        self.root = root / self.key
        legacy = {
            **self.identity,
            "source_sha256": {
                **self.identity["source_sha256"],
                "embedding_cache.py": LEGACY_CACHE_SOURCE,
            },
        }
        self.legacy_key = fingerprint(legacy)
        self.legacy_root = root / self.legacy_key
        if photo_hashes is not None and len(photo_hashes) != len(records):
            raise ValueError("캐시 사진 해시 목록과 Manifest의 크기가 다릅니다.")
        self.photo_hashes = photo_hashes
        self.photo_indexes: dict[bool, list[str]] = {}
        reusable = {k: v for k, v in self.identity.items() if k != "dataset_and_manifest"}
        self.shared_key = fingerprint({**reusable, "schema": "photo-hash-v1"})
        self.shared_root = root / self.shared_key
        self.records, self.processed_dir, self.model = records, processed_dir, model
        self.image_size, self.batch_size, self.device = image_size, batch_size, device
        self.banks: dict[str, dict[int, np.ndarray]] = {}
        self.hits = self.extracted = 0
        self.extraction_seconds = 0.0
        self.oom_retries = 0

    def photo_keys(self, augmented: bool) -> list[str]:
        """증강에 한해 난수에 사용되는 Manifest 행 번호를 사진 키에 포함합니다."""
        if augmented not in self.photo_indexes:
            self.photo_indexes[augmented] = [
                fingerprint([*hashes, i if augmented else None])
                for i, hashes in enumerate(self.photo_hashes or [])
            ]
        return self.photo_indexes[augmented]

    def _bank(self, key: str, photos: list[str]) -> dict[int, np.ndarray]:
        """이미 완료된 캐시 조각의 유효한 인덱스와 특징 벡터를 읽습니다."""
        if key not in self.banks:
            rows = {}
            by_photo: dict[str, list[int]] = {}
            for i, photo in enumerate(photos):
                by_photo.setdefault(photo, []).append(i)
            roots = [(self.root, self.key), (self.legacy_root, self.legacy_key)]
            if photos:
                roots.append((self.shared_root, self.shared_key))
            for path, identity in (
                (chunk, identity)
                for folder, identity in roots
                for chunk in sorted((folder / key).glob("*.npz"))
            ):
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        indices, features = saved["indices"], saved["features"]
                        if (
                            saved["identity"].item() != identity
                            or saved["variant"].item() != key
                            or indices.ndim != 1
                            or indices.dtype != np.int64
                            or features.ndim != 2
                            or len(indices) != len(features)
                            or features.shape[1] == 0
                            or features.shape[1]
                            != getattr(self.model, "num_features", features.shape[1])
                            or features.dtype != np.float32
                            or not np.isfinite(features).all()
                        ):
                            raise ValueError("캐시 조각 검증 실패")
                        if identity == self.shared_key:
                            keys = saved["photo_keys"]
                            if keys.shape != indices.shape or keys.dtype.kind != "U":
                                raise ValueError("사진 캐시 키 검증 실패")
                            for photo, feature in zip(keys, features):
                                for i in by_photo.get(photo, []):
                                    rows.setdefault(i, feature)
                        else:
                            if not np.all((0 <= indices) & (indices < len(self.records))):
                                raise ValueError("캐시 행 번호 범위 오류")
                            for index, feature in zip(indices.tolist(), features):
                                rows.setdefault(index, feature)
                except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                    print(f"[임베딩 캐시] 읽을 수 없는 조각을 건너뜁니다: {path.name}", flush=True)
            self.banks[key] = rows
        return self.banks[key]

    def _extract(
        self,
        indices: list[int],
        aug_pass: int,
        rotation_degrees: float,
        flip: bool,
    ) -> np.ndarray:
        """요청한 클린 또는 증강 이미지의 특징 벡터를 배치로 추출합니다."""
        transform = build_transforms(
            image_size=self.image_size,
            is_train=aug_pass > 0,
            rotation_degrees=rotation_degrees,
            fixed_hflip=flip,
        )
        dataset = SeededDataset(
            [self.records[i] for i in indices],
            indices,
            aug_pass,
            processed_dir=self.processed_dir,
            transform=transform,
        )
        batch = self.batch_size
        while True:
            try:
                with preserve_rng():
                    return extract_embeddings(self.model, dataset, batch, self.device)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or batch == 1:
                    raise
                batch = max(1, batch // 2)
                self.oom_retries += 1
            torch.cuda.empty_cache()
            print(f"[CUDA] {self.image_size}px 특징 배치를 {batch}로 줄여 재시도", flush=True)

    def _materialize(
        self,
        indices: list[int],
        aug_pass: int,
        rotation_degrees: float,
        flip: bool,
    ) -> dict[int, np.ndarray]:
        """캐시에 없는 인덱스만 추출하고 기존 조각과 별도로 공유 저장소에 기록합니다."""
        variant = fingerprint(
            {
                "aug_pass": aug_pass,
                "rotation_degrees": float(rotation_degrees),
                "flipped": flip,
            }
        )
        photos = self.photo_keys(aug_pass != 0)
        bank = self._bank(variant, photos)
        missing = sorted(set(indices).difference(bank))
        self.hits += len(indices) - len(missing)
        for start in range(0, len(missing), self.batch_size * 8):
            batch = missing[start : start + self.batch_size * 8]
            started = perf_counter()
            features = self._extract(batch, aug_pass, rotation_degrees, flip)
            self.extraction_seconds += perf_counter() - started
            if features.ndim != 2 or len(features) != len(batch) or not np.isfinite(features).all():
                raise ValueError("동결 백본이 유효하지 않은 특징 벡터를 생성했습니다.")
            identity = self.shared_key if photos else self.key
            folder = (self.shared_root if photos else self.root) / variant
            folder.mkdir(parents=True, exist_ok=True)
            unique_name = uuid.uuid4().hex
            temporary = folder / f".{unique_name}.tmp"
            # 조각 파일은 개별 이름을 사용하므로 동시에 만들어져도 기존 조각이 보존됩니다.
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    identity=identity,
                    variant=variant,
                    indices=np.asarray(batch, dtype=np.int64),
                    features=features,
                    **({"photo_keys": np.asarray([photos[i] for i in batch])} if photos else {}),
                )
            temporary.replace(folder / f"{unique_name}.npz")
            bank.update(zip(batch, features))
            self.extracted += len(batch)
        return bank

    def get(
        self,
        indices: list[int],
        aug_pass: int = 0,
        rotation_degrees: float = 0.0,
        hflip_probability: float = 0.0,
    ) -> np.ndarray:
        """원본 순서의 임베딩을 반환하며 반전 결과가 같은 특징을 재사용합니다."""
        if not indices:
            raise ValueError("임베딩을 요청할 인덱스가 없습니다.")
        if aug_pass == 0:
            clean = self._materialize(indices, 0, 0.0, False)
            return np.stack([clean[i] for i in indices])
        flipped = {
            i
            for i in indices
            if random.Random(deterministic_seed(i, aug_pass, "flip")).random() < hflip_probability
        }
        rows = {}
        for flip in (False, True):
            bucket = [i for i in indices if (i in flipped) == flip]
            if bucket:
                bank = self._materialize(bucket, aug_pass, rotation_degrees, flip)
                rows.update({i: bank[i] for i in bucket})
        return np.stack([rows[i] for i in indices])
