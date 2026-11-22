# Antigravity & Gemini Project Rules

## 1. 프로젝트 개요
- **프로젝트명**: 물리와 프로그래밍 6조 팀 프로젝트 (`physics_programing_team_project`)
- **목표**: 물리 및 인공지능 기법을 접목한 데이터 분석, 시뮬레이션 및 이미지 분류 모델 구현
- **주요 환경**: Python 3.13+, PyTorch/NumPy/SciPy/Pillow, Google Drive 연동

---

## 2. 권한 및 실행 허용 (Agent Permissions & Rules)
- **명령어 실행 허용**: 가상환경(`.venv`) 내 패키지 설치, `pytest` 테스트 실행, `ruff` 린트 및 코드 포맷팅, `python src/main.py` 등 개발 명령어 실행을 허용합니다.
- **파일 변경 허용**: `src/`, `tests/`, `notebooks/`, 설정 파일에 대한 신규 생성 및 수정을 적극 허용합니다.
- **Google Drive 연동 허용**:
  - `gdrive readonly` 명령어를 통한 데이터셋 목록 확인 및 다운로드를 허용합니다.
  - 팀 프로젝트 폴더 ID: `1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl` (6조 팀플)
  - 데이터 폴더 ID: `1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl` (데이터)
  - 단, 영구 삭제(`mutate empty-trash`) 및 사용자 동의 없는 원격 파일 삭제는 금지합니다.

---

## 3. 코딩 표준 및 물리/AI 규칙
1. **코드 스타일**:
   - PEP 8 준수, Python 3.10+ 최신 문법(Type Hinting 적극 활용)
   - 함수 및 클래스에는 역할, 매개변수, 반환값을 명시하는 한국어 Docstring 작성
2. **물리 계산 및 시뮬레이션 규칙**:
   - 모든 물리 상수는 SI 단위계를 기본으로 하며, 변수명 또는 주석에 단위 명시 (예: `time_s`, `velocity_m_per_s`)
   - 수치 계산 시 불필요한 Python for loop 지양, NumPy/SciPy 벡터화 연산 우선
3. **인공지능 및 머신러닝 규칙**:
   - 실험 재현성을 위해 시드 고정 (`torch.manual_seed(42)`, `np.random.seed(42)`)
   - 대용량 데이터셋(`.jpg`, `.png` 등) 및 모델 가중치(`.pt`, `.pth`)는 Git에 커밋하지 않고 구글 드라이브와 연동 (`.gitignore` 등록)
4. **테스트 및 검증**:
   - 주요 모듈 개발 시 `tests/` 디렉토리에 테스트 코드를 작성하고 `pytest`로 검증

---

## 4. 디렉토리 구조 규칙
```text
physics_programing_team_project/
├── .agents/            # Antigravity 에이전트 룰 및 설정
├── .venv/              # 파이썬 가상환경 (Git 제외)
├── data/
│   ├── raw/            # 원본 데이터 (구글 드라이브 동기화)
│   └── processed/      # 전처리 완료된 데이터
├── src/                # 핵심 소스 코드
│   ├── config.py       # 경로 및 하이퍼파라미터 설정
│   ├── dataset.py      # 데이터 로더 모듈
│   ├── drive_sync.py   # 구글 드라이브 연동 유틸리티
│   └── main.py         # 프로젝트 실행 진입점
├── tests/              # 단위 테스트 (pytest)
├── notebooks/          # 분석 및 실험용 주피터 노트북
├── requirements.txt    # 의존성 패키지 목록
├── pyproject.toml      # 프로젝트 메타데이터 및 도구 설정
└── GEMINI.md           # 프로젝트 에이전트 규칙 (본 파일)
```
