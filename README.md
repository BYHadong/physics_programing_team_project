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

---

## DINOv2 + Linear Probe 중복 수정 실험

Colab에서 전처리본 1,121장(대상 고양이 87장, 다른 샴고양이 572장, 기타 고양이 462장)을 복원했습니다. 개체 ID 간 복제 충돌을 격리한 뒤 기존 E00–E23을 재평가하고, 추가 탐색과 해상도 비교를 거쳐 R448을 저장했습니다. 실험 ID는 `dinov2-duplicate-fix-1121-01`입니다. 소스 ZIP의 SHA-256이 Colab 로그와 일치하고, 결과의 소스 해시도 배포 명세와 일치합니다. [실험 소스](experiments/dinov2_duplicate_fix/src) · [Colab 노트북](notebooks/dinov2_generalization_duplicate_fix.ipynb) · [결과 표](experiments/dinov2_duplicate_fix/results)

### 모델 구조와 전처리

- 기존 백본 비교에서 다른 샴을 구분하는 Hard ROC-AUC가 가장 높았던 자기지도 사전학습 ViT DINOv2-S/14(`vit_small_patch14_dinov2.lvd142m`)를 사용했습니다. 백본을 동결해 소량 데이터에서 학습할 파라미터 수를 줄이고, 384차원 특징 위에 이진 로지스틱 회귀를 적합했습니다. 대상 고양이가 양성이고, 다른 샴과 기타 고양이가 음성입니다. L2 규제, 클래스 가중치 `balanced`, `lbfgs`, 최대 1,000회를 사용했습니다.
- EXIF 방향 보정과 RGB 변환을 거쳐 YOLOv8n으로 고양이를 찾아 12% 여백을 두고 크롭합니다. 미탐지 시 원본 전체를 사용합니다. 비율을 유지해 512×512로 패딩하고, 입력 해상도로 리사이즈한 뒤 ImageNet 평균과 표준편차로 정규화합니다. ZIP 복원 시 정상 크롭을 재사용하며 누락·손상된 크롭을 복구할 수 있습니다.
- 랜덤 크롭, 색상 변화, 회전과 좌우 반전은 각 Fold의 학습용 복사본에만 적용했습니다. 검증과 Holdout에는 리사이즈와 정규화를 적용했습니다.

### 중복 해결과 데이터 분할

1. Manifest 경로와 원본·크롭 파일 및 디코딩 픽셀의 SHA-256을 비교했습니다. 동일 원본이 서로 다른 샴 개체 ID에 들어간 5쌍(10장)을 확인했습니다. 원본·크롭·픽셀 해시를 합해 충돌 증거 15건을 기록했습니다.
2. 충돌 사진 10장을 학습과 평가에서 격리했습니다. 복제로 연결된 개체들의 나머지 사진은 같은 분할 그룹으로 묶었고, 원본과 이전 결과는 보존했습니다. 격리 후 정확 중복 라벨 충돌은 0건이었습니다.
3. 대상 고양이는 촬영일 등 확인된 증거로 31개 그룹을 만들고, 다른 샴은 개체 ID를 기준으로 묶었습니다. 이전 Holdout과 연결된 그룹을 개발셋에서 격리했습니다. 남은 1,111장은 개발 917장(57/475/385)과 고정 Holdout 194장(30/87/77)입니다. 괄호 순서는 대상/다른 샴/기타입니다.
4. 5-Fold `GroupKFold`와 `StratifiedGroupKFold`를 비교해 분할 불균형 점수가 낮은 `GroupKFold`를 채택했습니다(0.0305 대 0.0422). dHash 거리 ≤4와 임베딩 코사인 유사도 ≥0.98에서는 분할 경계를 넘는 후보가 없었습니다. [분할 비교](experiments/dinov2_duplicate_fix/results/splitter_comparison.json) · [유사도 감사](experiments/dinov2_duplicate_fix/results/similarity_review/summary.json)

### 하이퍼파라미터 탐색과 선정

| 항목 | 탐색 설정 | 변경 이유 | 최종 R448 |
| --- | --- | --- | --- |
| L2 역규제 강도 `C` | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0 | 강한 규제의 과적합 억제 효과와 약한 규제의 개체 구분력을 비교 | 0.5 |
| 양성 목표 수와 증강 횟수 | 원본만, 250, 500, 750, 1,000장. 추가 탐색에서 원본만 0회, 목표 수 지정 시 7회 설정 | 양성 부족을 보완하는 효과와 같은 사진을 반복 변형하는 영향 비교 | 원본만, 0회 |
| 음성 증강 횟수 | 0, 1, 3, 5, 7회 | 다른 샴과 기타 음성의 다양성을 높여 오탐 및 재현율 변화 확인 | 5회 |
| 최대 회전 각도 | 0°, 5°, 10°, 15°, 20°, 30° | 촬영 기울기에 대한 강건성과 큰 회전이 외형에 주는 영향 비교 | ±10° |
| 좌우 반전 확률 | 0, 0.25, 0.5, 0.75, 1.0 | 좌우 자세 변화 대응과 비대칭 무늬 정보의 영향 비교 | 0.75 |
| 입력 해상도 | 224, 336, 448px | 얼굴과 무늬의 세부 정보가 구분력과 계산 비용에 주는 영향 비교 | 448px |
| 임계값 정책 | OOF 최대 F1, Precision–Recall 균형, 다른 샴 FPR 목표 1·5·10%와 기존 1.22%, 고정 0.5 | 양성 미탐과 다른 샴 오탐의 균형 비교 | 최대 F1, 0.679063 |

- 시드 42로 기존 24개 ID와 추가 31개 후보를 224px에서 평가했습니다. 항목별 순차 탐색에 C×음성 증강, 양성 목표×회전, 반전×회전 조합을 포함했습니다. 224px 최상위 T29의 설정을 고정해 336·448px를 비교했습니다. 동일 설정을 공유한 행까지 합해 57행이며 고유 CV는 45개였습니다.
- 개발셋에서 각 검증 Fold의 예측을 합친 OOF 지표로 순위를 정했습니다. 가중치는 Hard AUC 50%, F1 18%, 낮은 다른 샴 FPR 15%, 균형 정확도 7%, Recall 5%, 낮은 Fold별 Hard AUC 표준편차 5%였습니다. T29·R336·R448은 각각 0.8668·0.8951·0.9077을 기록했습니다.
- R448의 OOF 최대 F1 임계값 0.679063을 고정하고 개발 917장으로 분류기를 적합한 뒤 Holdout을 평가했습니다. OOF에서 다른 샴 FPR 1% 이하 정책을 적용하면 임계값 0.827427, FPR 0.84%, Recall 54.39%였습니다. 최대 F1 정책의 FPR과 Recall은 각각 4.84%, 82.46%였습니다. [선정 결과](experiments/dinov2_duplicate_fix/results/best_hyperparameters.json) · [후보별 결과](experiments/dinov2_duplicate_fix/results/hyperparameter_results.csv) · [해상도 비교](experiments/dinov2_duplicate_fix/results/resolution_comparison.csv)

### 기존 설정과 최종 모델 성능

기준 E00, 기존 튜닝 E15, 최종 R448을 같은 중복 수정 분할에서 비교했습니다. Hard AUC는 대상 고양이와 다른 샴만으로 계산한 ROC-AUC입니다. 다른 샴 FPR은 다른 샴을 대상으로 오인한 비율입니다.

| 평가 | 모델 | 정확도 | Hard AUC | 전체 ROC-AUC | Precision | Recall | F1 | 다른 샴 FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 개발 OOF 917장 | E00 | 95.75% | 0.9161 | 0.9534 | 68.00% | 59.65% | 0.6355 | 3.37% |
| 개발 OOF 917장 | E15 | 93.68% | 0.8851 | 0.9365 | 49.30% | 61.40% | 0.5469 | 7.58% |
| 개발 OOF 917장 | R448 | **96.40%** | **0.9581** | **0.9765** | 67.14% | **82.46%** | **0.7402** | 4.84% |
| 고정 Holdout 194장 | E00 | **97.42%** | **0.9985** | **0.9988** | **90.32%** | 93.33% | **0.9180** | **2.30%** |
| 고정 Holdout 194장 | E15 | 94.33% | 0.9759 | 0.9852 | 75.68% | 93.33% | 0.8358 | 9.20% |
| 고정 Holdout 194장 | R448 | **94.85%** | 0.9904 | 0.9945 | 76.32% | **96.67%** | 0.8529 | 9.20% |

R448은 개발 OOF에서 E00보다 Hard AUC와 F1이 높았습니다. 고정 Holdout에서는 184/194장을 맞혔고(TN 155, FP 9, FN 1, TP 29), 균형 정확도는 95.59%, Average Precision은 0.9755였습니다. Holdout에서 E00은 R448보다 정확도가 2.58%p 높고 다른 샴 FPR이 6.90%p 낮았습니다. R448은 대상 고양이 미탐을 E00의 2장에서 1장으로 줄였습니다. [동일 분할 비교](experiments/dinov2_duplicate_fix/results/baseline_comparison.csv) · [최종 지표](experiments/dinov2_duplicate_fix/results/final_model_metrics.json)

### 실험의 한계

- 고정 Holdout은 기존 데이터 안에서 격리한 194장입니다. 대상 30장은 한 그룹에 속했고 다른 샴은 두 분할 그룹뿐이라 그룹 단위 신뢰구간을 산출하지 못했습니다. 독립 촬영 데이터의 성능은 측정되지 않았습니다.
- dHash 거리 ≤8과 코사인 유사도 ≥0.95에서 각각 Fold 경계를 넘는 미검토 후보가 2쌍씩 남았습니다.
- 최종 후보와 임계값 선정에 개발 OOF를 사용해 해당 지표에는 탐색의 영향이 포함됩니다. Nested Group CV는 E00/E15 진단에 한정됐습니다. Tesla T4에서 해상도 비교는 448px 약 215초, 224px 약 50초였으며 시간에는 기존 특징 캐시의 영향이 포함됩니다.

### 실행과 테스트

1. [Colab 노트북](notebooks/dinov2_generalization_duplicate_fix.ipynb)을 열고 [검증된 소스 ZIP](experiments/dinov2_duplicate_fix/dinov2_duplicate_fix_source.zip)을 Drive의 `MyDrive/Cat_Project_Backup/dinov2_duplicate_fix_source.zip`에 둡니다. 같은 폴더에 `processed_1121.zip`을 두고 `DATA_ROOT`를 원본 세 클래스 폴더의 공통 상위 경로로 지정합니다.
2. 저장된 모델 평가에는 셀 1(설정·복원)과 셀 3(평가)을 실행합니다. 체크포인트는 `Cat_Project_Backup/checkpoints/dinov2-duplicate-fix-1121-01/final_model.pt`입니다. 셀 3에서는 단일 이미지(`RUN_SINGLE`, `IMAGE_PATH`), 클래스별 무작위 10장(`RUN_RANDOM`), 고정 Holdout 샘플(`RUN_STRICT`), 고정 Holdout 전체(`RUN_RESERVED`)를 선택할 수 있습니다. 평가만 할 때는 `STRICT_DEVELOPMENT=False`를 유지합니다. 무작위 샘플에는 개발 사진이 포함될 수 있습니다.
3. 그래프는 결과 ZIP을 Drive의 `Cat_Project_Backup/results/dinov2-duplicate-fix-1121-01/`에 풀고 셀 1과 4를 실행합니다. 셀 2는 학습용입니다. 저장소 노트북은 실행 로그를 담고 있으며 이미지·HTML 출력은 원본 노트북에 있습니다.

로컬에서 기존 가상환경으로 실험 단위 테스트와 CLI 옵션을 확인할 수 있습니다.

```powershell
Push-Location experiments/dinov2_duplicate_fix
..\..\.venv\Scripts\python.exe -m pytest
..\..\.venv\Scripts\python.exe -m src.test_study_model --help
Pop-Location
```

원본 이미지, 체크포인트, 임베딩 캐시와 상세 감사 파일은 Drive 또는 로컬 출력 경로에서 관리합니다. 결과 ZIP의 전체 내용은 로컬 `outputs/results/dinov2-duplicate-fix-1121-01/`에 풀었습니다.

### Windows 로컬 검증 기록

Colab에서 모델 학습과 평가를 완료했습니다. 이전 Windows 검증에서 실험 단위 테스트는 29개 중 26개가 통과했고, 아래 세 테스트는 임베딩 캐시 파일을 쓰는 단계에서 중단됐습니다. pytest 임시 경로에 두 개의 해시 디렉터리와 임시 파일명이 붙어 Windows 경로 길이 제한에 걸렸고, 파일을 열 때 `FileNotFoundError`가 발생했습니다.

| 실패한 테스트 | 실패 지점 |
| --- | --- |
| `test_hyperparameters.py::test_pipeline_resume_and_inference` | 합성 데이터로 파이프라인을 시작할 때 기본 임베딩 캐시 임시 파일을 열지 못함 |
| `test_hyperparameters.py::test_generalization_pipeline_and_test_modes` | 일반화 실험에서 첫 임베딩 캐시 임시 파일을 열지 못함 |
| `test_study_release.py::test_cache_legacy_shards_and_partial_data_revision` | 기존 캐시를 읽은 뒤 새 캐시 조각의 임시 파일을 열지 못함 |

별도 기존 프로젝트 테스트는 Windows 보안 정책에 따른 Python 확장 모듈 로딩 오류로 수집 단계에서 중단됐습니다. 이번 확인에는 Python 표준 라이브러리만 사용해 ZIP 무결성, 소스 해시, CSV·JSON 지표와 노트북 셀을 대조했습니다.
