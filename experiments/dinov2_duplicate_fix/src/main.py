import sys
from pathlib import Path

# Add src to sys.path if executed directly
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    GDRIVE_TEAM_FOLDER_ID,
    GDRIVE_DATA_FOLDER_ID,
    GDRIVE_CATEGORIES,
)
from src.drive_sync import check_gdrive_cli
from src.dataset import CatDataset


def main() -> None:
    """로컬 프로젝트와 Drive CLI 상태를 표시합니다."""
    print("=" * 60)
    print("  물리와 프로그래밍 6조 팀 프로젝트 - 환경 점검 및 시작")
    print("=" * 60)
    print(f"[경로] 프로젝트 루트 : {BASE_DIR}")
    print(f"[경로] 원본 데이터   : {RAW_DATA_DIR}")
    print(f"[경로] 전처리 데이터 : {PROCESSED_DATA_DIR}")
    print(f"[드라이브] 팀 폴더 ID : {GDRIVE_TEAM_FOLDER_ID}")
    print(f"[드라이브] 데이터 ID : {GDRIVE_DATA_FOLDER_ID}")
    print(f"[드라이브] 카테고리  : {list(GDRIVE_CATEGORIES.keys())}")
    print("-" * 60)

    # 1. Google Drive CLI 연동 확인
    has_gdrive = check_gdrive_cli()
    print(f"[연동] Google Drive CLI 감지 여부: {'[OK] 연결됨' if has_gdrive else '[NO] 미감지'}")

    # 2. 로컬 데이터셋 상태 확인
    dataset = CatDataset()
    summary = dataset.get_summary()
    print(f"[데이터] 로컬 이미지 데이터 현황: 총 {len(dataset)}장")
    for cat, count in summary.items():
        print(f"  - {cat}: {count}장")

    if len(dataset) == 0:
        print("\n[안내] 로컬에 데이터가 없습니다.")
        print("  Google Drive 연동을 통해 데이터를 동기화하려면 다음 명령어를 실행하세요:")
        print("  python src/drive_sync.py")

    print("=" * 60)


if __name__ == "__main__":
    main()
