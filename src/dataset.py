"""
고양이 이미지 데이터셋 로더 모듈
"""

from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image

try:
    from src.config import RAW_DATA_DIR, GDRIVE_CATEGORIES, IMAGE_SIZE
except ImportError:
    from config import RAW_DATA_DIR, GDRIVE_CATEGORIES, IMAGE_SIZE


class CatDataset:
    """
    로컬 RAW_DATA_DIR에 저장된 이미지 데이터셋을 로드하는 클래스
    구글 드라이브의 etc_cats, main_siamese_cat, other_siamese_cat 구조를 지원합니다.
    """
    def __init__(self, data_dir: Path = RAW_DATA_DIR, target_size: Tuple[int, int] = IMAGE_SIZE):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.classes = sorted(list(GDRIVE_CATEGORIES.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        self.samples.clear()
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = self.data_dir / cls_name
            if cls_folder.exists() and cls_folder.is_dir():
                for file_path in cls_folder.iterdir():
                    if file_path.suffix.lower() in valid_exts:
                        self.samples.append((file_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB").resize(self.target_size)
        return image, label

    def get_summary(self) -> Dict[str, int]:
        counts = {cls_name: 0 for cls_name in self.classes}
        for _, label in self.samples:
            cls_name = self.classes[label]
            counts[cls_name] += 1
        return counts
