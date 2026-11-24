"""
프로젝트 전역 경로, Google Drive ID, 전처리 및 모델 하이퍼파라미터 설정 모듈
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일이 존재하는 경우 환경 변수 로드
load_dotenv()

# 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 및 출력 디렉토리
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"

# 필수 디렉토리 자동 생성
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# 원본 데이터 소스 디렉토리 (환경변수 -> 루트 '데이터' 폴더 -> data/raw 순으로 탐색)
_env_source_dir = os.getenv("SOURCE_DATA_DIR", "").strip()
if _env_source_dir:
    SOURCE_DATA_DIR = Path(_env_source_dir)
elif (BASE_DIR / "데이터").exists():
    SOURCE_DATA_DIR = BASE_DIR / "데이터"
else:
    SOURCE_DATA_DIR = RAW_DATA_DIR

# Google Drive Folder IDs (물리와 인공지능 - 6조 팀플)
GDRIVE_TEAM_FOLDER_ID = os.getenv("GDRIVE_TEAM_FOLDER_ID", "1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl")
GDRIVE_DATA_FOLDER_ID = os.getenv("GDRIVE_DATA_FOLDER_ID", "1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl")

# 드라이브 내 카테고리별 폴더 ID
GDRIVE_CATEGORIES = {
    "main_siamese_cat": "1Q-8uPv-239ygJE3PmwnSl7RdB7ARK3SC",
    "other_siamese_cat": "1JcFD6ELcFM3LXj9jeJxANFGfZDBGZKv-",
    "etc_cats": "1wvugYpIN8BpsqCzt0dsaR3LWEl4hiKJX",
}

# 전처리 및 YOLO 탐지 설정
PREPROCESSED_SIZE = 512
CROP_MARGIN_RATIO = 0.12
COCO_CAT_CLASS_ID = 15
YOLO_CONF_THRESHOLD = 0.20
YOLO_WEIGHTS_PATH = CHECKPOINTS_DIR / "yolov8n.pt"

# 이미지 정규화 상수 (ImageNet 기준)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 비교 대상 사전학습 백본 모델 목록
BACKBONE_MODELS = {
    "megadescriptor": "hf-hub:BVRA/MegaDescriptor-S-224",
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
}

# 이미지 크기 및 모델 학습 하이퍼파라미터
IMAGE_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
RANDOM_SEED = 42
