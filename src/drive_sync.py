"""
Google Drive 연동 유틸리티 모듈

Antigravity gdrive CLI 및 로컬 폴더를 활용하여
구글 드라이브(6조 팀플 데이터 폴더)와의 파일 동기화, 목록 조회, 다운로드/업로드를 수행합니다.
"""

import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from src.config import (
        GDRIVE_DATA_FOLDER_ID,
        GDRIVE_TEAM_FOLDER_ID,
        GDRIVE_CATEGORIES,
        RAW_DATA_DIR,
    )
except ImportError:
    from config import (
        GDRIVE_DATA_FOLDER_ID,
        GDRIVE_TEAM_FOLDER_ID,
        GDRIVE_CATEGORIES,
        RAW_DATA_DIR,
    )


def check_gdrive_cli() -> bool:
    """gdrive CLI가 시스템 PATH에 설치되어 있는지 확인합니다."""
    return shutil.which("gdrive") is not None


def run_gdrive_command(cmd_args: List[str]) -> subprocess.CompletedProcess:
    """gdrive CLI 명령어를 실행하고 결과를 반환합니다."""
    full_cmd = ["gdrive"] + cmd_args
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace"
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gdrive 명령어 실행 실패: {' '.join(full_cmd)}", file=sys.stderr)
        print(f"[STDOUT] {e.stdout}", file=sys.stderr)
        print(f"[STDERR] {e.stderr}", file=sys.stderr)
        raise


def list_drive_folder(folder_id: str) -> str:
    """지정된 드라이브 폴더의 파일 목록을 반환합니다."""
    result = run_gdrive_command(["readonly", "ls", folder_id])
    return result.stdout


def list_team_datasets() -> Dict[str, str]:
    """6조 팀플 데이터 폴더 내 카테고리별 파일 목록을 조회합니다."""
    outputs = {}
    print(f"=== 구글 드라이브 팀플 데이터 폴더 목록 조회 (ID: {GDRIVE_DATA_FOLDER_ID}) ===")
    root_listing = list_drive_folder(GDRIVE_DATA_FOLDER_ID)
    outputs["root"] = root_listing
    print(root_listing)

    for category, cat_id in GDRIVE_CATEGORIES.items():
        print(f"\n--- 카테고리: {category} (ID: {cat_id}) ---")
        cat_listing = list_drive_folder(cat_id)
        outputs[category] = cat_listing
        print(cat_listing)

    return outputs


def download_file(file_id: str, dest_path: Path) -> None:
    """구글 드라이브에서 특정 파일을 다운로드합니다."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    run_gdrive_command(["readonly", "download", file_id, "--destination", str(dest_path)])
    print(f"[DONE] 다운로드 완료: {dest_path}")


def upload_result(file_path: Path, parent_id: Optional[str] = None) -> None:
    """로컬 결과물(모델 체크포인트, 리포트 등)을 구글 드라이브 폴더로 업로드합니다."""
    target_id = parent_id or GDRIVE_TEAM_FOLDER_ID
    if not file_path.exists():
        raise FileNotFoundError(f"업로드할 파일이 존재하지 않습니다: {file_path}")

    args = ["mutate", "upload", str(file_path), "--parent", target_id]
    result = run_gdrive_command(args)
    print(f"[DONE] 구글 드라이브 업로드 완료 ({file_path.name} -> {target_id})")
    print(result.stdout)


if __name__ == "__main__":
    if not check_gdrive_cli():
        print("[WARNING] gdrive CLI가 감지되지 않았습니다. Antigravity gdrive 플러그인 환경에서 실행하세요.")
    else:
        list_team_datasets()
