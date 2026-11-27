"""
고양이 이미지 데이터셋 로더, 전처리 변환 및 5-Fold Group CV 분할 모듈
"""

import io
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from sklearn.model_selection import GroupKFold, KFold
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from src.config import (
        DEFAULT_HFLIP_PROBABILITY,
        DEFAULT_ROTATION_DEGREES,
        GDRIVE_CATEGORIES,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
        SOURCE_DATA_DIR,
    )
except ImportError:
    from config import (
        DEFAULT_HFLIP_PROBABILITY,
        DEFAULT_ROTATION_DEGREES,
        GDRIVE_CATEGORIES,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PROCESSED_DATA_DIR,
        RANDOM_SEED,
        SOURCE_DATA_DIR,
    )

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def letterbox_square(
    image: Image.Image,
    target_size: int,
    fill_color: Tuple[int, int, int] = (124, 116, 104),
) -> Image.Image:
    """
    가로세로 비율을 유지하면서 정사각형 캔버스에 이미지를 배치하고 목표 크기로 리사이즈합니다.

    매개변수:
        image: 입력 PIL 이미지 객체
        target_size: 출력 정사각형 한 변의 픽셀 크기
        fill_color: 여백(Letterbox)을 채울 RGB 색상

    반환값:
        (target_size, target_size) 크기의 RGB PIL 이미지
    """
    rgb_img = image.convert("RGB")
    w, h = rgb_img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), fill_color)

    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


class RandomJPEGCompression:
    """
    메모리 내에서 확률적으로 JPEG 재압축을 적용하는 커스텀 Transform.
    고화질 4K JPG(내 고양이)와 저화질 스크린샷 PNG(다른 샴고양이) 간의
    화질·선명도 도메인 격차(Shortcut Learning)를 방지합니다.
    """

    def __init__(self, min_quality: int = 50, max_quality: int = 95, p: float = 0.4) -> None:
        """
        매개변수:
            min_quality: 최소 JPEG 품질 (1~100)
            max_quality: 최대 JPEG 품질 (1~100)
            p: 압축 증강을 적용할 확률
        """
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """확률 p로 이미지를 JPEG 압축 후 다시 디코딩하여 반환합니다."""
        if random.random() > self.p:
            return img
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


def build_transforms(
    image_size: int = 224,
    is_train: bool = True,
    enable_hflip: bool = True,
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES,
    hflip_probability: float = DEFAULT_HFLIP_PROBABILITY,
    *,
    fixed_hflip: Optional[bool] = None,
) -> transforms.Compose:
    """
    학습용(온라인 증강 포함) 및 검증/추론용 이미지 전처리 파이프라인을 생성합니다.

    매개변수:
        image_size: 모델 입력 해상도 (정사각형)
        is_train: True면 온라인 데이터 증강 적용, False면 평가용 변환만 적용
        enable_hflip: 학습 시 좌우 반전 증강 사용 여부
        rotation_degrees: 양방향 최대 회전 각도
        hflip_probability: 학습 시 좌우 반전 확률
        fixed_hflip: 캐시에서 확률 추출을 마친 경우 적용할 반전 결과

    반환값:
        torchvision.transforms.Compose 객체
    """
    if not is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    ops: List[Callable] = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
        ),
    ]
    if fixed_hflip is True:
        ops.append(transforms.Lambda(transforms.functional.hflip))
    elif fixed_hflip is None and enable_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=hflip_probability))

    ops.extend([
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        RandomJPEGCompression(min_quality=50, max_quality=95, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


class CatDataset:
    """
    로컬 데이터 폴더에 저장된 원본 이미지 데이터셋을 재귀적으로 로드하는 클래스.
    EXIF 회전 보정, RGB 3채널 통일 및 비율 유지 정사각형 패딩을 적용합니다.
    """

    def __init__(
        self,
        data_dir: Path = SOURCE_DATA_DIR,
        target_size: Tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        """
        매개변수:
            data_dir: 클래스별 하위 폴더가 위치한 루트 경로
            target_size: 반환 이미지의 (가로, 세로) 크기
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.classes = sorted(list(GDRIVE_CATEGORIES.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """클래스 폴더 내부를 재귀 탐색하여 이미지 파일 경로 목록을 구성합니다."""
        self.samples.clear()
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = self.data_dir / cls_name
            if cls_folder.exists() and cls_folder.is_dir():
                for file_path in sorted(cls_folder.rglob("*")):
                    if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTS:
                        self.samples.append((file_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """EXIF 보정 및 정사각형 패딩이 적용된 (PIL Image, class_idx)를 반환합니다."""
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            fixed = ImageOps.exif_transpose(img).convert("RGB")
            image = letterbox_square(fixed, self.target_size[0])
        return image, label

    def get_summary(self) -> Dict[str, int]:
        """클래스별 이미지 수량을 딕셔너리로 반환합니다."""
        counts = {cls_name: 0 for cls_name in self.classes}
        for _, label in self.samples:
            cls_name = self.classes[label]
            counts[cls_name] += 1
        return counts


class ManifestCatDataset(Dataset):
    """
    manifest.json 레코드 목록을 기반으로 전처리된 이미지를 로드하는 PyTorch Dataset.
    """

    def __init__(
        self,
        records: List[Dict],
        processed_dir: Path = PROCESSED_DATA_DIR,
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        """
        매개변수:
            records: manifest.json에서 읽은 샘플 메타데이터 딕셔너리 리스트
            processed_dir: 전처리된 이미지가 저장된 루트 디렉토리
            transform: PyTorch 텐서 변환 및 증강 파이프라인
            classes: 클래스 이름 정렬 목록
        """
        self.records = records
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.classes = classes or sorted(list(GDRIVE_CATEGORIES.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str, str]:
        """
        반환값:
            (image_tensor, binary_label, class_idx, individual_id, rel_path)
            - binary_label: main_siamese_cat이면 1, 나머지는 0
        """
        rec = self.records[idx]
        rel_path = rec["processed_rel_path"]
        img_path = self.processed_dir / rel_path
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        cls_name = rec["class_name"]
        binary_label = 1 if cls_name == "main_siamese_cat" else 0
        class_idx = self.class_to_idx[cls_name]
        individual_id = rec.get("individual_id", cls_name)
        return tensor_img, binary_label, class_idx, individual_id, rel_path


def create_group_folds(
    records: List[Dict],
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    record_groups: Optional[Dict[str, str]] = None,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    데이터 누수를 방지하기 위해 클래스 특성에 맞춰 5-Fold 교차 검증 분할을 생성합니다.
    - main_siamese_cat: 확인된 촬영 그룹이 있으면 GroupKFold, 없으면 KFold
    - other_siamese_cat: individual_id 기준 GroupKFold (검증 시 미확인 개체만 평가)
    - etc_cats: 확인된 중복 그룹이 있으면 GroupKFold, 없으면 KFold

    매개변수:
        records: manifest.json의 전체 레코드 리스트
        n_splits: Fold 개수 (기본 5)
        seed: 난수 시드
        record_groups: 확인된 촬영 단위와 중복을 반영한 전처리 경로별 그룹

    반환값:
        [(train_records, val_records), ...] 길이 n_splits의 리스트
    """
    main_recs = [r for r in records if r["class_name"] == "main_siamese_cat"]
    other_recs = [r for r in records if r["class_name"] == "other_siamese_cat"]
    etc_recs = [r for r in records if r["class_name"] == "etc_cats"]
    extra_recs = [
        r for r in records
        if r["class_name"] not in {"main_siamese_cat", "other_siamese_cat", "etc_cats"}
    ]

    def _split_kfold(bucket: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        if not bucket:
            return [([], []) for _ in range(n_splits)]
        if len(bucket) < n_splits:
            splits: List[Tuple[List[Dict], List[Dict]]] = []
            for fold_idx in range(n_splits):
                val = [r for i, r in enumerate(bucket) if i % n_splits == fold_idx]
                train = [r for i, r in enumerate(bucket) if i % n_splits != fold_idx]
                splits.append((train, val))
            return splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [([bucket[i] for i in tr], [bucket[i] for i in va]) for tr, va in kf.split(bucket)]

    def _split_group(bucket: List[Dict], groups: List[str]) -> List[Tuple[List[Dict], List[Dict]]]:
        if not bucket:
            return [([], []) for _ in range(n_splits)]
        unique_groups = sorted(set(groups))
        if len(unique_groups) < n_splits:
            group_to_fold = {g: i % n_splits for i, g in enumerate(unique_groups)}
            splits: List[Tuple[List[Dict], List[Dict]]] = []
            for fold_idx in range(n_splits):
                val = [r for r, g in zip(bucket, groups) if group_to_fold[g] == fold_idx]
                train = [r for r, g in zip(bucket, groups) if group_to_fold[g] != fold_idx]
                splits.append((train, val))
            return splits
        gkf = GroupKFold(n_splits=n_splits)
        return [
            ([bucket[i] for i in tr], [bucket[i] for i in va])
            for tr, va in gkf.split(bucket, groups=groups)
        ]

    def _split_verified(
        bucket: List[Dict], other: bool = False,
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """확인된 그룹이 있으면 그룹을 유지하고 독립 행은 기존 방식으로 분할합니다."""
        if record_groups is None:
            if other:
                return _split_group(
                    bucket, [record.get("individual_id", "unknown") for record in bucket]
                )
            return _split_kfold(bucket)
        groups = [record_groups[record["processed_rel_path"]] for record in bucket]
        if other or len(set(groups)) < len(groups):
            return _split_group(bucket, groups)
        return _split_kfold(bucket)

    main_splits = _split_verified(main_recs)
    other_splits = _split_verified(other_recs, other=True)
    etc_splits = _split_verified(etc_recs)
    extra_splits = _split_kfold(extra_recs)

    folds: List[Tuple[List[Dict], List[Dict]]] = []
    for k in range(n_splits):
        train_k = (
            main_splits[k][0]
            + other_splits[k][0]
            + etc_splits[k][0]
            + extra_splits[k][0]
        )
        val_k = (
            main_splits[k][1]
            + other_splits[k][1]
            + etc_splits[k][1]
            + extra_splits[k][1]
        )
        folds.append((train_k, val_k))
    return folds
