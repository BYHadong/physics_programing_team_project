import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Ensure essential directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Google Drive Folder IDs (물리와 인공지능 - 6조 팀플)
GDRIVE_TEAM_FOLDER_ID = os.getenv("GDRIVE_TEAM_FOLDER_ID", "1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl")
GDRIVE_DATA_FOLDER_ID = os.getenv("GDRIVE_DATA_FOLDER_ID", "1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl")

# Subfolder IDs in Drive
GDRIVE_CATEGORIES = {
    "main_siamese_cat": "1Q-8uPv-239ygJE3PmwnSl7RdB7ARK3SC",
    "other_siamese_cat": "1JcFD6ELcFM3LXj9jeJxANFGfZDBGZKv-",
    "etc_cats": "1wvugYpIN8BpsqCzt0dsaR3LWEl4hiKJX",
}

# Image Processing & Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
RANDOM_SEED = 42
