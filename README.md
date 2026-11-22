# 물리와 프로그래밍 6조 팀 프로젝트 (Physics & AI Team Project)

물리 및 인공지능 기법을 접목한 데이터 분석, 시뮬레이션 및 고양이 이미지 분류 팀 프로젝트입니다.

---

## 📁 디렉토리 구조

```text
physics_programing_team_project/
├── .agents/
│   └── rules/
│       └── project_rules.md      # 팀 프로젝트 세부 규칙
├── .venv/                        # 파이썬 가상환경
├── data/
│   ├── raw/                      # 원본 이미지 데이터 (구글 드라이브 동기화)
│   └── processed/                # 전처리 완료된 데이터
├── notebooks/                    # 주피터 노트북 실험 공간
├── src/
│   ├── __init__.py
│   ├── config.py                 # 전역 설정 및 구글 드라이브 폴더 ID
│   ├── dataset.py                # 이미지 데이터셋 로더 (Pillow/Torch)
│   ├── drive_sync.py             # 구글 드라이브(6조 팀플) 연동 및 동기화 모듈
│   └── main.py                   # 프로젝트 시작 및 상태 점검 진입점
├── tests/
│   ├── __init__.py
│   └── test_project_setup.py     # 프로젝트 설정 및 환경 검증 테스트
├── .env.example                  # 환경 변수 템플릿
├── .gitignore                    # Git 제외 파일 목록 (가상환경, 대용량 데이터 등)
├── GEMINI.md                     # Antigravity/Gemini 에이전트 프로젝트 룰
├── AGENTS.md                     # 에이전트 가이드라인
├── pyproject.toml                # 패키지 빌드 및 도구 설정
├── requirements.txt              # 전체 패키지 의존성 목록
└── README.md                     # 본 문서
```

---

## 🚀 시작하기

### 1. 가상환경 활성화
Windows PowerShell 환경:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 패키지 설치
전체 요구 패키지 설치:
```powershell
pip install -r requirements.txt
```
또는 개발 모드로 설치:
```powershell
pip install -e ".[dev]"
```

### 3. 프로젝트 상태 점검 및 실행
```powershell
python src/main.py
```

### 4. 단위 테스트 실행
```powershell
pytest
```

---

## ☁️ Google Drive 연동

본 프로젝트는 Antigravity의 `gdrive` 플러그인과 연동되어 있습니다.

- **팀 프로젝트 폴더**: `물리와 인공지능 / 6조 팀플` (`1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl`)
- **데이터 폴더**: `6조 팀플 / 데이터` (`1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl`)
  - `main_siamese_cat`: `1Q-8uPv-239ygJE3PmwnSl7RdB7ARK3SC`
  - `other_siamese_cat`: `1JcFD6ELcFM3LXj9jeJxANFGfZDBGZKv-`
  - `etc_cats`: `1wvugYpIN8BpsqCzt0dsaR3LWEl4hiKJX`

### 구글 드라이브 데이터 확인 및 동기화 스크립트 실행
```powershell
python src/drive_sync.py
```

CLI 직접 사용 시:
```powershell
# 드라이브 팀 데이터 폴더 확인
gdrive readonly ls 1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl

# 모델 결과물 업로드 (필요 시)
gdrive mutate upload checkpoints/model.pt --parent 1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl
```

---

## 📜 팀 프로젝트 룰 (Rule)
- 코딩 표준, 물리 단위 표기 규칙, AI 재현성 시드 고정 및 데이터 보안 규칙은 [GEMINI.md](file:///d:/study/physics/physics_programing/physics_programing_team_project/GEMINI.md)와 [.agents/rules/project_rules.md](file:///d:/study/physics/physics_programing/physics_programing_team_project/.agents/rules/project_rules.md)에 정의되어 있습니다.
- 대용량 데이터는 Git에 커밋하지 않고 Google Drive를 통해 동기화합니다.
