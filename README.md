# 물리와 프로그래밍 6조 팀 프로젝트 (Physics & AI Team Project)

동일 품종(샴고양이) 내에서 특정 반려동물 개체(`main_siamese_cat`)를 다른 샴고양이(`other_siamese_cat`) 및 기타 품종 고양이(`etc_cats`)와 구별하는 **개체 식별(Fine-Grained Pet Re-ID) AI 파이프라인** 프로젝트입니다.

---

## 📁 디렉토리 구조

```text
physics_programing_team_project/
├── .agents/
│   └── rules/
│       └── project_rules.md      # 팀 프로젝트 세부 규칙
├── .venv/                        # 파이썬 가상환경 (Git 제외)
├── 데이터/                       # 구글 드라이브에서 내려받은 원본 이미지 (Git 제외)
│   ├── main_siamese_cat/         # 구분할 내 반려동물 (77장)
│   ├── other_siamese_cat/        # 다른 샴고양이 21개체 폴더 (572장)
│   └── etc_cats/                 # 기타 품종 고양이 (462장)
├── data/
│   ├── raw/                      # 예비 원본 데이터 폴더
│   └── processed/                # Stage 1 YOLO 크롭 및 패딩 완료 데이터 + manifest.json
├── checkpoints/                  # 학습된 모델 가중치 (best_baseline.pt, yolov8n.pt)
├── outputs/                      # 비교 실험 JSON 및 시각화 그리드 PNG
├── notebooks/                    # 주피터 노트북 실험 공간
├── src/
│   ├── __init__.py
│   ├── config.py                 # 전역 경로, Google Drive ID 및 하이퍼파라미터 설정
│   ├── dataset.py                # EXIF 보정, Letterbox 패딩, 온라인 증강, GroupKFold 분할
│   ├── drive_sync.py             # 구글 드라이브(6조 팀플) 연동 및 동기화 모듈
│   ├── preprocess.py             # Stage 1: EXIF 보정 + YOLOv8n 고양이 크롭 전처리
│   ├── train_baseline.py         # Stage 2: 3종 백본 5-Fold Group CV 비교 및 학습
│   ├── predict.py                # 단일 이미지 판별 추론 CLI
│   ├── test_random_samples.py    # 클래스별 N장 랜덤 추출 평가 및 결과 그리드 생성
│   └── main.py                   # 프로젝트 환경 점검 진입점
├── tests/
│   ├── __init__.py
│   ├── test_project_setup.py     # 프로젝트 설정 및 환경 검증 테스트
│   └── test_pipeline.py          # 전처리, 증강, GroupKFold 누수 방지 단위 테스트
├── .env.example                  # 환경 변수 템플릿
├── .gitignore                    # Git 제외 파일 목록 (가상환경, 대용량 데이터, 출력물 등)
├── GEMINI.md                     # Antigravity/Gemini 에이전트 프로젝트 룰
├── AGENTS.md                     # 에이전트 가이드라인
├── pyproject.toml                # 패키지 빌드 및 도구 설정
├── requirements.txt              # 전체 패키지 의존성 목록
└── README.md                     # 본 문서
```

---

## 🚀 파이프라인 실행 방법

### 1. 환경 준비 및 단위 테스트
```powershell
# 가상환경 활성화 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt

# 단위 테스트 실행 (7개 테스트)
pytest
```

### 2. Stage 1: 데이터 전처리 (EXIF 보정 + YOLO 고양이 크롭)
```powershell
python src/preprocess.py
```

### 3. Stage 2: 5-Fold Group CV 백본 비교 학습
```powershell
# 기본 학습 (main_siamese_cat 온라인 증강 500장 확장)
python src/train_baseline.py --main-target-count 500

# 양성/음성 대칭 증강 비교 예시
python src/train_baseline.py --main-target-count 500 --neg-aug-passes 7
```

### 4. 추론 및 랜덤 샘플 테스트
```powershell
# 단일 이미지 판별
python src/predict.py "판별할_이미지_경로.jpg"

# 클래스별 10장(총 30장) 랜덤 테스트 + 결과 그리드 저장
python src/test_random_samples.py --random

# 테스트 샘플 및 해당 개체를 학습에서 완전 제외한 순수 미학습 평가
python src/test_random_samples.py --random --strict-holdout
```

---

## 🧠 1. 모델 선정 방식

본 문제는 단순 품종 분류가 아니라 **같은 샴고양이 품종 내에서 특정 1개체(내 고양이)를 구별하는 개체 식별(Fine-Grained Pet Re-ID)** 문제입니다.

1. **2-Stage 파이프라인 설계**
   - **Stage 1 (전처리)**: `ImageOps.exif_transpose`로 EXIF 90도 회전 태그를 보정하고 RGBA→RGB로 통일한 뒤, **`YOLOv8n`**(COCO `cat`)으로 고양이 영역만 크롭(여백 12%, 전체 1,111장 중 **94.7% 탐지 성공**)하고 비율을 유지한 `512×512` 정사각형 패딩(`letterbox_square`)을 적용해 바닥·소파 등 배경 편향을 차단했습니다.
   - **Stage 2 (특징 추출 및 판별)**: 소량 데이터(`main` 77장)에서 과적합을 줄이기 위해 사전학습 백본을 동결(Frozen)하고, 성격이 다른 **3종 백본 × 2종 판별 방식**을 동일 조건의 **5-Fold Group CV**로 비교했습니다.
     - **비교 백본**: `MegaDescriptor-S-224`(동물 개체식별 특화 Swin), **`DINOv2`**(`vit_small_patch14_dinov2.lvd142m`, 자기지도학습 ViT), `ConvNeXt-Tiny`(CNN)
     - **판별 방식**: `prototype_cosine`(임베딩 평균 코사인 유사도) vs `linear_probe`(클래스 불균형 가중치 `LogisticRegression`)
     - **검증 원칙**: `other_siamese_cat` 21마리를 개체 단위(`GroupKFold`)로 분할해 **학습 때 한 번도 본 적 없는 샴고양이**에 대한 구분력(**Hard ROC-AUC**)을 핵심 선정 기준으로 삼았습니다.
2. **선정 결과**: 모든 실험 조건에서 **`DINOv2 + linear_probe`**가 가장 높은 Hard ROC-AUC와 안정적인 Fold 편차를 기록해 최종 모델로 선정했습니다.

---

## 🔧 2. 하이퍼파라미터·증강 튜닝 및 성능 변화

| 실험 단계 | 주요 설정 (데이터 · 증강 · 임계값) | Hard AUC (vs 다른 샴) | Overall AUC | Best F1 | Precision / Recall | Unseen FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **실험 1 (초기)** | `main` 30장(총 1,064장), 대칭 증강 7회 | 0.9939 | 0.9966 | 0.9492 | 0.9655 / 0.9333 | 0.17% |
| **실험 2 (데이터 확장)** | **`main` 77장**(총 1,111장), 대칭 증강 7회, `thr=0.8718` | **0.9680** | **0.9819** | **0.8276** | **0.8824** / 0.7792 | **1.22%** |
| **실험 3 (비대칭 증강)** | `main` 77→**500장 증강**, `other/etc` 1회 증강, `thr=0.7019` | 0.9584 | 0.9764 | 0.7619 | 0.7033 / **0.8312** | 4.55% |

- **핵심 관찰**:
  - 실험 1의 매우 높은 수치는 `main` 30장 중 24장이 임베딩 유사도 0.95 이상의 연사 사진이었기 때문이며, 다양한 각도·장면의 사진 47장을 추가한 실험 2에서 과대평가 거품이 빠진 현실적인 성능이 나타났습니다.
  - 실험 3처럼 `main`만 강하게 증강(약 8배)하고 음성 클래스(`other`/`etc`)는 1회만 증강하면 **증강 비대칭**으로 Recall은 오르지만(`0.7792 → 0.8312`) 닮은 다른 샴고양이 오탐(`Unseen FPR: 1.22% → 4.55%`)이 늘어 F1이 소폭 하락했습니다.
- **추가 튜닝 예정 작업**:
  - `LogisticRegression` L2 규제 강도 `C ∈ {0.01, 0.05, 0.1, 0.5, 1.0}` 그리드 탐색
  - 양성/음성 대칭 증강(`--main-target-count 500 --neg-aug-passes 7`) 비교
  - `DINOv2 + ConvNeXt` 임베딩 앙상블 및 추론 시 TTA(원본+좌우반전 평균) 적용

---

## 📊 3. 최종 모델 성능 요약 (`DINOv2 + linear_probe`)

- **5-Fold Group CV (전체 1,111장, 미확인 샴고양이 구분 기준)**:
  - **최고 균형 기록 (실험 2 대칭 증강)**: Hard AUC **`0.9680`** (Fold 평균 `0.9751±0.0205`), Overall AUC **`0.9819`**, F1 **`0.8276`**, Unseen Siamese FPR **`1.22%`**
  - **현재 체크포인트 (실험 3 `main` 500장 확장)**: Hard AUC **`0.9584`**, Overall AUC **`0.9764`**, F1 **`0.7619`**, Recall **`0.8312`**, Unseen Siamese FPR **`4.55%`** (`outputs/baseline_results.json`)
- **30장 랜덤 `Strict Holdout` 테스트** (테스트 30장 및 `other` 10개체 전체를 학습에서 제외 후 812장만 재학습):
  - 전체 정확도 **30/30 (100%)** (`main` 확률 `74.45~97.64%`, `other` `8.45~48.04%`, `etc` `0.01~15.28%`로 뚜렷한 마진 확보)
- **강건성(스트레스 테스트) 검증**:
  - 고양이 영역을 지우고 배경만 남겼을 때 평균 확률 `8.65%`(오판 0%) → **배경 치트키(Shortcut Learning) 없음 확인**
  - `128px + JPEG Q35`로 강하게 화질을 저하시켜도 정답률 `93.3%` 유지 → **4K JPG vs 저화질 PNG 도메인 편향 없음 확인**

---

## 🖼️ 4. 결과 시각화 및 한계에 대한 논의

- **결과 시각화 (`outputs/random_sample_test_grid.png`)**:
  - `Main 사진`, `Other 사진`, `ETC 사진` 3행×10열 그리드로 YOLO 크롭 결과, 예측 확률, `T:MAIN | P:MAIN`(초록 테두리) vs 나머지(빨간 테두리)를 한눈에 확인할 수 있습니다.
- **한계점 및 향후 과제**:
  1. **연사 사진 편중**: `main_siamese_cat` 77장 내에 같은 날짜(`20261118`, `20261124`) 연사 사진이 다수 포함되어 있어, 완전히 새로운 장소·야간 조명·측면 각도에서는 예측 확률이 낮아질 수 있습니다(Leave-One-Scene-Out 테스트에서 미학습 장면 평균 확률 `64~78%` 확인).
  2. **YOLO 미탐지 예외**: 극단적 클로즈업이나 특이 자세 등 약 `5.3%`의 이미지는 YOLO가 탐지하지 못해 원본 전체가 그대로 입력됩니다.
  3. **전용 외부 테스트셋 필요**: 학습 데이터와 완전히 독립된 날짜·환경에서 새로 촬영한 내 고양이 사진과 신규 샴고양이 사진으로 최종 외부 검증을 수행하면 실전 일반화 성능을 보다 정확히 입증할 수 있습니다.

---

## ☁️ Google Drive 연동

본 프로젝트는 Antigravity의 `gdrive` 플러그인과 연동되어 있습니다.

- **팀 프로젝트 폴더**: `물리와 인공지능 / 6조 팀플` (`1SZmXRpx9VWUT0LKS4VNqpTXXFaU4QQwl`)
- **데이터 폴더**: `6조 팀플 / 데이터` (`1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl`)
  - `main_siamese_cat`: `1Q-8uPv-239ygJE3PmwnSl7RdB7ARK3SC`
  - `other_siamese_cat`: `1JcFD6ELcFM3LXj9jeJxANFGfZDBGZKv-`
  - `etc_cats`: `1wvugYpIN8BpsqCzt0dsaR3LWEl4hiKJX`

```powershell
# 구글 드라이브 데이터 확인 및 동기화 스크립트 실행
python src/drive_sync.py
```

---

## 📜 팀 프로젝트 룰 (Rule)
- 코딩 표준, 물리 단위 표기 규칙, AI 재현성 시드 고정 및 데이터 보안 규칙은 [GEMINI.md](GEMINI.md), [AGENTS.md](AGENTS.md), [.agents/rules/project_rules.md](.agents/rules/project_rules.md)에 정의되어 있습니다.
- 대용량 데이터(`데이터/`, `data/processed/`, `checkpoints/`, `outputs/`)는 Git에 커밋하지 않고 Google Drive를 통해 동기화합니다.
