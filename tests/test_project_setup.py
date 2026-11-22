import os
from pathlib import Path
from src.config import BASE_DIR, DATA_DIR, RAW_DATA_DIR, GDRIVE_CATEGORIES
from src.dataset import CatDataset


def test_project_directories_exist():
    """핵심 프로젝트 디렉토리가 올바르게 생성되어 있는지 검증합니다."""
    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()


def test_gdrive_categories_configured():
    """구글 드라이브 카테고리 ID가 정상적으로 정의되어 있는지 검증합니다."""
    assert "main_siamese_cat" in GDRIVE_CATEGORIES
    assert "other_siamese_cat" in GDRIVE_CATEGORIES
    assert "etc_cats" in GDRIVE_CATEGORIES


def test_dataset_initialization():
    """CatDataset 초기화가 오류 없이 동작하는지 검증합니다."""
    dataset = CatDataset()
    summary = dataset.get_summary()
    assert isinstance(summary, dict)
    assert len(summary) == 3
