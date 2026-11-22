# Workspace Agent Guidelines

## Agent Execution & Collaboration Guidelines
- 본 프로젝트는 **물리와 프로그래밍 6조 팀 프로젝트**입니다.
- 개발 및 분석 작업 시 가상환경(`.venv`)의 파이썬 인터프리터를 기준으로 작업합니다.
- 데이터셋 조작이나 모델 훈련 코드 작성 시 `src/config.py`의 경로와 설정값을 참조합니다.
- 구글 드라이브 관련 작업은 `src/drive_sync.py` 또는 `gdrive` CLI를 활용합니다.
- 변경 사항 발생 시 단위 테스트(`pytest`)를 수행하여 회귀 버그가 없는지 확인합니다.
