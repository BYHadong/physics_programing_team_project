"""
Google Drive 연동 유틸리티 모듈

Antigravity gdrive CLI 및 로컬 폴더를 활용하여
구글 드라이브(6조 팀플 데이터 폴더)와의 파일 동기화, 목록 조회, 다운로드/업로드를 수행합니다.
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

try:
    from src.config import (
        COLAB_BACKUP_DIR,
        GDRIVE_CATEGORIES,
        GDRIVE_DATA_FOLDER_ID,
        GDRIVE_TEAM_FOLDER_ID,
        PROCESSED_DATA_DIR,
    )
except ImportError:
    from config import (
        COLAB_BACKUP_DIR,
        GDRIVE_CATEGORIES,
        GDRIVE_DATA_FOLDER_ID,
        GDRIVE_TEAM_FOLDER_ID,
        PROCESSED_DATA_DIR,
    )


def check_gdrive_cli() -> bool:
    """gdrive CLI가 시스템 PATH에 설치되어 있는지 확인합니다."""
    return shutil.which("gdrive") is not None


def run_gdrive_command(cmd_args: List[str]) -> subprocess.CompletedProcess:
    """gdrive CLI 명령어를 실행하고 결과를 반환합니다."""
    full_cmd = ["gdrive"] + cmd_args
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace"
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


def restore_processed(archive_path: Path, output_dir: Path = PROCESSED_DATA_DIR) -> bool:
    """
    읽기 전용 ZIP을 검증하여 필요한 전처리 파일을 로컬 경로에 복원합니다.

    archive_path는 Drive 백업, output_dir는 로컬 전처리 폴더입니다.
    복원한 파일이 있으면 True, 기존 Manifest와 파일이 모두 있으면 False를 반환합니다.
    """

    def image_paths(manifest: Path) -> set[Path]:
        """Manifest가 참조하는 안전한 상대 이미지 경로 집합을 반환합니다."""
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("복원할 Manifest가 비어 있거나 지원하지 않는 형식입니다.")
        paths = {Path(row["processed_rel_path"]) for row in rows}
        if len(paths) != len(rows) or any(
            path.is_absolute() or path.drive or ".." in path.parts for path in paths
        ):
            raise ValueError("Manifest에 중복 또는 안전하지 않은 전처리 경로가 있습니다.")
        return paths

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and all(
        (output_dir / path).is_file() and (output_dir / path).resolve().is_relative_to(output_dir)
        for path in image_paths(manifest_path)
    ):
        print(f"[전처리] 기존 데이터 사용: {output_dir}")
        return False

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary,
        ZipFile(archive_path) as zf,
    ):
        staging = Path(temporary)
        seen = set()
        for member in zf.infolist():
            path = Path(member.filename)
            if (
                path.is_absolute()
                or path.drive
                or ".." in path.parts
                or "\\" in member.filename
                or ":" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
                or path in seen
            ):
                raise ValueError("ZIP에 중복 또는 안전하지 않은 경로가 있습니다.")
            seen.add(path)
            if not member.is_dir():
                zf.extract(member, staging)
        options = list(staging.rglob("manifest.json"))
        if len(options) != 1:
            raise ValueError("ZIP 내에 전처리 manifest.json이 하나 있어야 합니다.")
        manifest = options[0]
        references = sorted(image_paths(manifest)) + [Path("manifest.json")]
        for path in references:
            original = manifest.parent / path
            target = output_dir / path
            if not original.is_file() or not target.resolve().is_relative_to(output_dir):
                raise ValueError("ZIP 내 요청한 전처리 이미지가 없거나 경로가 올바르지 않습니다.")
            if target.exists() and (
                not target.is_file()
                or hashlib.sha256(target.read_bytes()).digest()
                != hashlib.sha256(original.read_bytes()).digest()
            ):
                raise FileExistsError(
                    f"기존 파일과 ZIP이 다릅니다. 원본 파일을 유지합니다: {target}"
                )
        for path in references:
            target = output_dir / path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                # 로컬 staging과 출력은 같은 볼륨에 있으므로 기존 파일을 덮어쓰지 않습니다.
                os.link(manifest.parent / path, target)
    print(f"[전처리 복원] {output_dir}")
    return True


def main() -> None:
    """기존 팀 데이터 조회 또는 로컬 전처리 복원을 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-processed", type=Path, nargs="?", const=COLAB_BACKUP_DIR / "processed.zip"
    )
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    if args.restore_processed is not None:
        restore_processed(args.restore_processed, args.output)
    elif not check_gdrive_cli():
        print("[WARNING] gdrive CLI가 감지되지 않았습니다.")
    else:
        list_team_datasets()


if __name__ == "__main__":
    main()
