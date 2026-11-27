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
├── notebooks/
│   └── hyperparameter_experiment.ipynb  # 1,121장 Colab 실험
├── src/
│   ├── __init__.py
│   ├── config.py                 # 전역 경로, Google Drive ID 및 하이퍼파라미터 설정
│   ├── dataset.py                # EXIF 보정, Letterbox 패딩, 온라인 증강, GroupKFold 분할
│   ├── drive_sync.py             # 구글 드라이브(6조 팀플) 연동 및 동기화 모듈
│   ├── preprocess.py             # Stage 1: EXIF 보정 + YOLOv8n 고양이 크롭 전처리
│   ├── prepare_dataset.py        # 기존 크롭 재사용 및 1,121장 백업 준비
│   ├── experiment_splits.py      # Manifest 대조, 그룹 감사, 고정 분할
│   ├── embedding_cache.py        # 동결 DINOv2 특징 벡터 저장 및 재사용
│   ├── linear_probe.py           # 분류기 학습, 검증 지표, 임계값 보정
│   ├── tune_hyperparameters.py    # E00, 후보 비교, 최종 체크포인트
│   ├── analyze_results.py         # 실험 결과 PNG 변환
│   ├── train_baseline.py         # Stage 2: 3종 백본 5-Fold Group CV 비교 및 학습
│   ├── predict.py                # 단일 이미지 판별 추론 CLI
│   ├── test_random_samples.py    # 클래스별 N장 랜덤 추출 평가 및 결과 그리드 생성
│   └── main.py                   # 프로젝트 환경 점검 진입점
├── tests/
│   ├── __init__.py
│   ├── test_project_setup.py     # 프로젝트 설정 및 환경 검증 테스트
│   ├── test_pipeline.py          # 기존 전처리, 증강 및 GroupKFold 테스트
│   ├── test_prepare_dataset.py   # 증분 전처리와 백업 보존
│   ├── test_embedding_cache.py    # 캐시 키 및 재사용
│   └── test_hyperparameters.py    # 튜닝, 임계값, 추론과 사진 그리드
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

# 단위 테스트 실행
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

> 아래 1~4절은 팀의 초기 모델 선정과 1,111장까지의 실험 기록입니다.
> 1,121장 실험의 실행 방법과 보고서는 [별도 절](#동결-dinov2-linear-probe-하이퍼파라미터-실험)에 있습니다.

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

## 동결 DINOv2 Linear Probe 하이퍼파라미터 실험

[Colab 노트북](notebooks/hyperparameter_experiment.ipynb)은 Drive 연결부터 전처리 준비,
테스트, 후보 비교, 최종 학습, 추론과 PNG 저장까지 한 개의 Python 셀에서 실행합니다.
원본 수량은 87장·572장·462장으로 총 1,121장입니다. 이전 1,111장 크롭을
재사용할 수 있으면 신규 10장만 기존 EXIF·YOLO·Letterbox 방식으로 처리하고
별도 `processed_1121.zip`에 저장합니다. 완성된 1,121장 크롭도 재사용할 수 있습니다.
기존 `processed.zip`을 읽기 전용으로 두고, `dinov2-1121-01`에서 같은 분할로
E00과 후보를 새로 측정합니다. 노트북은 작업 브랜치의 `src/`와 `tests/`를 가져옵니다.

### 기준과 탐색 범위

| 단계 | 변경 조건 | 후보 |
| --- | --- | --- |
| E00 | 양성·음성 대칭 기준 | 원본 1회 + 추가 증강 7회, C=1.0, ±15°, 반전 확률 0.5 |
| 정규화 | LogisticRegression C | 0.01, 0.05, 0.1, 0.5, 1.0 |
| 음성 증강 | C 선택 뒤 음성 추가 패스 비교 | 1, 3, 5, 7 |
| 양성 증강 | 음성 조건 선택 뒤 양성 원본 포함 목표 장수 | 250, 500, 750 |
| 회전 | 양방향 최대 회전 | ±5°, ±15°, ±30° |
| 좌우 반전 | 반전 확률 | 0.0, 0.25, 0.5, 0.75, 1.0 |
| 조합 | 상위 양성 수량에서 음성 증강 대안 확인 후 상위 후보를 한 변수씩 비교 | 최대 3개 추가 실험 |

E00은 README 실험 2의 ‘대칭 증강 7회’를 각 원본에 일곱 개의 증강본을 추가하는 의미로
구현합니다. 기존 CLI의 `--main-target-count 500 --neg-aug-passes 7`은 양성을 Fold마다
500장에 맞추므로 이 방식과 일치하지 않습니다. 과거 실행 설정과 분할은 체크포인트나
코드에서 확인할 수 없으므로 당시 수치는 참고용으로 두고, E00을 현재 분할에서 새로
측정합니다. 기존 3종 백본 비교는 `src/train_baseline.py`에서 계속 실행할 수 있습니다.

모든 신규 후보에서 `vit_small_patch14_dinov2.lvd142m` 동결 특징과 224px 입력,
기본 배치 32, 기존 색상·블러·JPEG 증강, L2, `class_weight="balanced"`,
`lbfgs`, 1,000회 반복 상한, 시드 42를 사용합니다. C가 작을수록 L2 정규화가
강합니다. 정규화 단계와 조합 단계는 Hard ROC-AUC를 우선하고, 동률일 때
다른 샴고양이 FPR이 낮고 F1이 높으며 Fold Hard AUC 표준편차가 작은 후보를
앞에 둡니다. 그 순위는 실제 평가한 후보에 적용됩니다. 증강 단계의 F1과 오탐을
해석할 때는 실제 Fold별 양성·음성 학습 수량, Precision과 Recall, 목표 FPR에서의
Recall을 함께 확인합니다. 좌우 반전 단계는 개체의 비대칭 특징이 보존되는 정도를
검토할 수 있도록 각 후보의 Hard AUC와 Recall을 함께 보여줍니다.

탐색 목록을 바꾸려면 `--search-space search_space.json`을 사용합니다. 키는
`c`, `main_target_count`, `neg_aug_passes`, `rotation_degrees`,
`hflip_probability`이며 값은 후보 목록입니다. 예를 들어
`{"c": [0.05, 0.1, 1.0], "hflip_probability": [0, 0.5, 1.0]}`은 두 단계의 목록만 변경합니다.
선택 임계값 정책은 `--threshold-policy`에서 지정합니다.

### 분할, 점수, 임계값

- 데이터가 허용하면 처음에 고정 Group Holdout을 분리합니다. E00과 모든 후보는
  나머지 개발 데이터의 동일한 5-Fold를 사용합니다. 다른 샴고양이는 개체 그룹을
  공유하고, 대상 고양이는 확인된 촬영 날짜나 Manifest의 장면·연사 ID를 공유합니다.
  같은 원본 경로와 동일 픽셀은 같은 그룹에 넣습니다.
- 확인 가능한 정보가 없는 대상 사진은 원본 이미지 단위로 분할합니다. 유사 이미지
  후보와 분할 사이의 중복 가능성은 `data_audit.json`과 `fold_indices.json`에 남깁니다.
  유사 이미지 후보는 dHash 해밍 거리 4 이하로 기록됩니다. 후보에는 서로 다른
  사진도 포함될 수 있습니다.
- 학습 인덱스에만 증강을 사용합니다. Fold 검증과 고정 Holdout에는 원본 크롭의
  클린 임베딩을 사용합니다. 본 데이터 안에서 정한 Holdout으로 검증하며, 별도의
  외부 데이터셋은 사용하지 않습니다. 고정 Holdout을 구성할 그룹이 부족하면 개발
  CV 성능과 그 한계만 보고합니다. 양성과 다른 샴고양이가 함께 없는 Fold의
  Hard AUC는 공란으로 남기고 평균·표준편차의 유효 Fold 수를 표시합니다.
- Hard ROC-AUC는 대상과 기타 샴고양이만 사용합니다. Overall ROC-AUC는
  기타 품종도 음성에 포함합니다. FPR의 분모는 실제 다른 샴고양이 사진 수입니다.
  Balanced Accuracy는 민감도와 특이도의 산술 평균이고 PR 요약은 Average Precision입니다.
  혼동행렬은 `[[TN, FP], [FN, TP]]` 순서이며 Fold 표준편차는 모집단 표준편차입니다.
- 네 정책은 개발 Fold의 OOF 점수로 보정합니다. `max_f1`은 F1 최대,
  `precision_recall_balance`는 참 양성을 찾는 후보 중 Precision과 Recall의 차이 최소,
  `min_siamese_fpr`은 FPR 최소 후 Recall 최대, `target_siamese_fpr`은
  FPR 1.22% 이하에서 Recall 최대입니다. 같은 점수는 한꺼번에 판정하므로 정확히
  1.22%인 임계값이 없을 수 있습니다. 기본값은 `max_f1`이고 Holdout 평가 시
  이미 선정된 임계값을 적용합니다.
- 일반 랜덤 테스트는 학습 이미지를 포함할 수 있습니다. Strict Holdout은 선택된
  샘플과 검증 가능한 그룹(다른 샴고양이 개체 전체 포함)을 제외하고, 체크포인트의
  C, 증강 수량, 회전, 반전 조건을 사용해 재학습합니다. 임계값은 남은 학습 그룹의
  내부 OOF에서 다시 계산합니다. 선택한 그룹이 이전 하이퍼파라미터 탐색에 포함됐다면
  이 테스트의 하이퍼파라미터 선정 과정에도 그 그룹이 관여한 것입니다.
  `--reserved-cohort`를 두 랜덤 명령에 함께 주면 동일 체크포인트의 사전 Holdout에서
  같은 시드로 비교할 수 있습니다.

### 설치와 Colab 실행

작업 브랜치는 `feature/00110111100101`입니다. 로컬에서 실행하려면 Python 3.10
이상을 설치한 뒤 프로젝트를 받습니다. Windows PowerShell의 예시입니다.

```powershell
git clone --branch feature/00110111100101 https://github.com/BYHadong/physics_programing_team_project.git
cd physics_programing_team_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt "ruff==0.11.13"
python -m pytest -v
python -m ruff check src tests
```

Colab에서 [실험 노트북](notebooks/hyperparameter_experiment.ipynb)을 열고 T4 GPU를
선택해 Python 셀 하나를 실행합니다. `DATA_ROOT`는 세 클래스 폴더의 공통 상위 경로,
`BACKUP`은 접근 권한이 있는 Drive 백업 폴더입니다. 노트북은 실행한 커밋 SHA를
출력합니다. 같은 커밋을 재현하려면 `GIT_REF`에 그 SHA를 넣으세요. Git 참조를
바꿀 때 이미 체크아웃된 폴더가 있으면 `PROJECT`에 별도 Colab 경로를 지정하세요.

전처리 준비를 CLI에서 따로 실행하려면 `python -m src.prepare_dataset --help`로
원본·전처리·백업 경로 옵션을 확인하세요. 학습은 `python -m src.tune_hyperparameters`,
발표용 PNG 생성은 `python -m src.analyze_results`입니다. 두 명령에 같은 백업
경로와 `--run-id dinov2-1121-01`을 사용합니다. 같은 코드·데이터·환경의 실행은
완료 결과와 임베딩을 재사용합니다.

### Drive에서 최종 모델과 결과 불러오기

실험을 실행한 계정의 **내 드라이브 → Cat_Project_Backup**에 아래 파일이 저장됩니다.
팀원은 폴더 접근 권한을 받은 뒤 Colab의 `BACKUP`을 해당 Drive 경로로 지정합니다.
Drive 폴더를 내려받아 프로젝트에 두는 경우 기본 제외 이름인 `Cat_Project_Backup/`을
그대로 쓸 수 있습니다.

```text
Cat_Project_Backup/
├── processed_1121.zip
├── checkpoints/dinov2-1121-01/final_model.pt
├── results/dinov2-1121-01/
│   ├── best_hyperparameters.json
│   ├── final_model_metrics.json
│   └── baseline_comparison.csv
├── graphs/dinov2-1121-01/
└── embeddings/
```

Colab에서 작업 브랜치의 코드와 의존성을 준비한 뒤 아래처럼 체크포인트를 지정하면
저장 모델로 이미지 한 장을 판별할 수 있습니다. `image`를 판별할 이미지 경로로 바꾸세요.

```python
from pathlib import Path
import subprocess
import sys
from google.colab import drive

drive.mount("/content/drive")
project = Path("/content/physics_programing_team_project_github")
backup = Path("/content/drive/MyDrive/Cat_Project_Backup")
checkpoint = backup / "checkpoints" / "dinov2-1121-01" / "final_model.pt"
image = Path("/content/example.jpg")
subprocess.run(
    [sys.executable, "src/predict.py", str(image), "--checkpoint", str(checkpoint)],
    cwd=project,
    check=True,
)
```

일반 랜덤 판별은 `src/test_random_samples.py --random --seed 42 --checkpoint <파일>`로
실행합니다. Strict Holdout은 같은 명령에 `--strict-holdout`을 추가합니다.
원본 경로는 `--source`, 복원한 전처리 폴더는 `--processed`로 전달하고,
기존 크롭으로 검사하려면 `--reuse-processed`를 추가하세요.
Strict Holdout을 실행하면 제외 조건에 맞추어 선형 분류기를 다시 학습합니다.

### 선정 하이퍼파라미터와 1,121장 실험 결과

1,121장 실험의 E00은 기존 1,111장 실험과 별도로 측정합니다. 공유된 탐색 결과에
따르면 조합 재검증 후 최종 후보는 **E15**입니다.

| 실험 단계 | 탐색 항목 | 선정 조건 |
| --- | --- | --- |
| E01–E05 | LogisticRegression C | 0.01 |
| E06–E09 | 음성 증강 | 원본 1회에 추가 증강 5회 |
| E10–E12 | 대상 고양이 학습 수량 | Fold별 원본 포함 750장 |
| E13–E15 | 회전 범위 | ±30° |
| E16–E20 | 좌우 반전 확률 | 50% |
| E21–E23 | 상위 조건 조합 | E15 유지 |

E15의 좌우 반전 확률은 기본값인 50%이며, E18은 같은 확률을 다시 비교하는 후보입니다.
E15는 탐색한 후보 중 가장 높은 순위를 받은 설정입니다. 임계값과 성능 수치는 아래
Drive 보고서에서 확인합니다.

| 필요한 값 | 파일 및 항목 |
| --- | --- |
| 선정 실험 ID와 C, 증강 수량, 회전, 반전 확률 | `best_hyperparameters.json`의 `experiment`, `parameters` |
| 선정 임계값과 개발 CV 지표 | `best_hyperparameters.json`의 `validation_metrics` |
| E00과 최종 모델의 분할별 성능 | `baseline_comparison.csv` |
| 고정 Holdout 지표와 샘플 수 | `final_model_metrics.json`의 `reserved_group_holdout`, `holdout_index_count` |

Drive를 마운트한 Colab에서 아래 셀을 실행하면 기존 결과를 읽을 수 있습니다.

```python
import json
from pathlib import Path
from IPython.display import display
import pandas as pd

folder = Path("/content/drive/MyDrive/Cat_Project_Backup/results/dinov2-1121-01")
best = json.loads((folder / "best_hyperparameters.json").read_text(encoding="utf-8"))
report = json.loads((folder / "final_model_metrics.json").read_text(encoding="utf-8"))
print("선정 실험", best["experiment"])
print("하이퍼파라미터", best["parameters"])
print("판별 임계값", best["validation_metrics"]["threshold"])
print("고정 Holdout 이미지 수", report["holdout_index_count"])
policies = pd.DataFrame.from_dict(best["validation_threshold_policies"], orient="index")
display(policies[["threshold", "precision", "recall", "f1", "unseen_siamese_fpr"]])
display(pd.read_csv(folder / "baseline_comparison.csv"))
```

### E15 단독 학습

선정된 E15만 학습하려면 기존 1,121장 전처리 폴더와 원본 폴더를 지정합니다.
설정은 C 0.01, 음성 추가 증강 5회, 양성 목표 750장, 회전 ±30°, 반전 확률 50%입니다.
동일한 5-Fold Group CV에서 임계값을 보정하고, 개발 데이터로 최종 분류기를 학습합니다.
고정 Holdout이 있는 경우 해당 사진은 학습과 임계값 보정에서 제외합니다.
이 실행은 원래 탐색과 별도 ID를 사용하며 호환되는 Drive 임베딩 캐시를 재사용합니다.

```sh
python -m src.train_e15 --source /path/to/raw --processed outputs/processed_1121 \
  --output /path/to/Cat_Project_Backup --run-id dinov2-e15-1121-01
```

체크포인트는 `checkpoints/dinov2-e15-1121-01/final_model.pt`, 검증과 최종 평가 결과는
`results/dinov2-e15-1121-01/`에 저장됩니다. 추론은 `src/predict.py --checkpoint`로
이 체크포인트를 지정합니다. 완료된 실행은 동일한 ID로 재실행하면 저장 결과를 사용합니다.

### 결과 경로

- `embeddings/`에는 데이터, Manifest, 백본 가중치, 전처리, 시드·환경으로 구분한
  임베딩 조각이 저장됩니다. C 변경은 재추출 없이 처리됩니다. 수량 증가 시에는
  추가분, 회전 변경 시에는 필요한 새 각도의 특징, 반전 확률 변경 시에는 실제 반전
  결과가 달라져 캐시에 없는 특징만 추출합니다.
- `experiments/<run-id>/E00/result.json`부터 각 후보의 네 가지 임계값, OOF 점수,
  Fold 지표와 소요 시간을 저장합니다.
- `results/<run-id>/`에는 `hyperparameter_results.csv`, `hyperparameter_results.json`,
  `fold_results.csv`, `best_hyperparameters.json`, `baseline_comparison.csv`,
  `final_model_metrics.json`, `experiment_config.json`, `environment_versions.txt`,
  `fold_indices.json`, `data_audit.json`이 저장됩니다.
- `checkpoints/<run-id>/final_model.pt`에는 개발 데이터로 재학습한 선형 분류기,
  백본 식별자 및 가중치 해시, 증강 설정과 추론 임계값이 저장됩니다.
- `graphs/<run-id>/`에는 후보별 성능, 기준·최종 비교, Hard/Overall ROC,
  PR, 혼동행렬, Fold 성능 PNG가 저장됩니다. Colab의 두 랜덤 그리드와 샘플별 JSON은
  같은 폴더의 `sample_tests/` 아래 각 실행별로 저장됩니다.

## 1,121장 DINOv2 최종 검증

[최종 Colab 노트북](notebooks/dinov2_generalization.ipynb)과 실행용
`dinov2_final_review_source.zip`을 사용합니다. ZIP에는 실제 Python 소스와
파일별 SHA-256 목록이 있습니다. 1번 셀은 ZIP의 파일과 실행 폴더를 대조하고
실행 소스 ZIP 해시를 출력합니다. 현재 변경 사항을 GitHub에 올리기 전에도
이 ZIP으로 실행할 수 있습니다.

1. Drive의 `MyDrive/Cat_Project_Backup`에 다음을 준비합니다.
   - `dinov2_final_review_source.zip`
   - 기존 `processed_1121.zip`
   - 기존 `results/dinov2-1121-01/hyperparameter_results.csv` 또는 `.json`
   - 기존 Holdout을 비교하려면 같은 폴더의 `fold_indices.json`과
     `experiment_config.json`
2. Colab에서 GPU 런타임을 선택하고 노트북을 연 뒤 셀 1부터 실행합니다.
   원본 폴더는 `DATA_ROOT`, 기존 실험 경로는 `LEGACY_RESULTS`에서 바꿀 수 있습니다.
3. 셀 2는 감사, 기존 후보 재평가, 추가 탐색, 최종 학습을 실행합니다.
   사진 검토만 먼저 할 때는 `AUDIT_ONLY=True`로 설정합니다.
4. 셀 3은 저장된 모델 테스트입니다. `RUN_SINGLE`, `RUN_RANDOM`, `RUN_STRICT`,
   `RUN_RESERVED`로 고르고, 셀 4에서 비교표와 PNG를 만듭니다.
   학습된 모델의 테스트만 할 때는 1번과 3번을 실행합니다.

### 데이터와 캐시

기본 입력은 87·572·462장으로 총 1,121장입니다. ZIP의 경로·CRC, JPEG 디코딩,
512×512 RGB 크롭, Manifest와 원본의 일대일 대응을 확인합니다. 정상 ZIP에서는
Manifest의 바이트와 순서를 유지하고 YOLO를 호출하지 않습니다. 누락·손상·검출된
변경이 있으면 기존 크롭을 남겨 필요한 사진만 다시 처리한 뒤 별도 수리 ZIP에
저장합니다. 기존 ZIP에 원본 해시가 없으면 당시 원본의 바이트 일치는 확인할 수
없습니다. 처음 확인한 현재 원본과 크롭의 해시는 별도 무결성 파일에 남기고,
이후 변경을 확인하는 데 사용합니다.

동결된 `vit_small_patch14_dinov2.lvd142m`의 특징에는 이미지당 L2 정규화를
적용합니다. 표본이나 전체 Fold에 맞춰 추정하는 변환은 없습니다. C, Fold, 분류
임계값은 특징 계산 키를 바꾸지 않습니다. 새 공용 캐시는 가중치, 해상도, 실행
환경, 이미지·크롭 해시와 실제 증강을 확인합니다. 증강 난수는 기존 방식의
Manifest 행 번호와 시드에 의존합니다. 이전 캐시는 Manifest·크롭, 변환 코드와
환경 서명이 호환되는 경우에 읽습니다. 단계 로그의 캐시 건수는 특징 요청 횟수라
같은 벡터의 여러 사용이 포함됩니다. 새 계산 건수는 실제로 생성한 벡터 수입니다.

### 분할과 데이터 감사

서로 다른 다른 샴고양이 개체 폴더에 바이트가 같은 원본이 복제된 경우
`--quarantine-other-copies`로 해당 복제본을 학습·검증·평가에서 제외할 수
있습니다. 연결된 개체의 나머지 사진은 동일 분할 그룹에 둡니다. 원본과
Manifest는 그대로 유지하고, 격리 내역은 `data_audit.json`에 기록합니다.
기존 실행과 분할이 달라지므로 새 RUN_ID를 사용하세요.

원본·크롭 SHA-256, 디코딩된 크롭 픽셀 해시, dHash, DINOv2 코사인, EXIF·파일명,
개체 폴더를 이용합니다. 수동 촬영 기록을 가장 우선하며, 자동 확인에서는 EXIF
촬영일을 Manifest 촬영일과 파일명 날짜보다 우선합니다. 근거 없는 양성과
개체 ID를 확인할 수 없는 다른 샴고양이는 제외합니다. 실제 유사 사진은 수동 검토
없이 서로 다른 개체로 합치지 않습니다.

- `data_audit.json`에 확인된 양성 그룹 수, `fold_indices.json`에 개발·평가 그룹과
  샘플 수, `splitter_comparison.json`에 GroupKFold와 StratifiedGroupKFold의
  분포를 저장합니다.
- 분할을 가로지르는 dHash ≤ 4 및 코사인 ≥ 0.98 미검토 쌍이 있으면 학습이
  대기합니다. `similarity_review/priority_cross_partition.csv`와 이미지 목록을
  확인합니다. `REVIEW_CSV`는 `left,right,decision` 열에 `duplicate` 또는
  `different`를 씁니다. `SESSIONS_CSV`는 `processed_rel_path,session_id` 열을
  사용합니다. 검토나 데이터 분할이 변경되면 새 RUN_ID를 선택합니다.
- 이전 Holdout의 어떤 사진도 새 개발 데이터에서 사용하지 않으며, 연결된 확인
  그룹 전체를 격리합니다. 새 그룹에 맞춰 평가 범위가 확대돼도 비교표에서 기존
  204장 중 평가 가능한 부분을 따로 기록합니다. 양성과 다른 샴고양이가 모두
  들어가는 유효 CV가 불가능하면 학습을 중단합니다.
- 그룹을 보존하고 수가 충분한 범위에서 최대 5 Fold로 평가합니다. 촬영일이
  같으면 넓게 묶이는 경우가 있습니다. 촬영 날짜만으로 독립성을 보증할 수
  없고, 강한 유사도 기준을 벗어나는 누수 가능성도 남습니다.

### 후보, 임계값, 해상도

보관 파일의 E00–E23 실제 설정을 복원하고 20개의 고유 조건을 같은 새 개발
Fold로 재평가합니다. CSV와 JSON을 모두 읽을 수 있으면 두 설정을 대조합니다.
설정이 빠지거나 서로 다르면 해당 ID를 보고하고 탐색을 중단합니다. 기존 수치는
`historical_metrics_audit.json`에서 원본 파일에 근거한 관측값과 함께 표시합니다.
랜덤 테스트의 모델을 저장된 메타데이터로 확인할 수 없으면 참고값의 검증 상태를
비워 둡니다.

E15 기준은 C 0.01, 양성 목표 750장, 음성 추가 5회, 회전 ±30°, 좌우 반전 0.5입니다.
양성 목표 수량이 활성화되면 양성 추가 패스 7은 적용되지 않습니다. 목표 750 또는
1,000은 학습 Fold의 원본과 파생 증강을 합한 장수이며, 독립 사진 수는 별도입니다.
작은 C는 L2 정규화를 더 강하게 적용합니다.

| 탐색 변수 | 추가 후보 |
| --- | --- |
| C | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1 |
| 양성 학습 목표 | 원본만, 250, 500, 750, 1000 |
| 음성 추가 증강 | 0, 1, 3, 5, 7 |
| 최대 회전 각도 | 0°, 5°, 10°, 15°, 20°, 30° |
| 반전 확률 | 0, 0.25, 0.5, 0.75, 1 |

기본은 단일 변수 28개와 제한된 조합 최대 3개입니다. `ADDITIONAL_BUDGET`을
낮추면 먼저 지정된 변수부터 예산을 사용합니다. 같은 설정의 별칭은 결과표에
남기고 재학습을 생략합니다. 총 예정은 최대 57행이며, 기존 조건 중복으로
고유 CV는 최대 53개입니다. 최대 5 Fold면 후보 Logistic Regression 적합은
최대 265회입니다. 고정 평가용 재학습과 원본 축소 진단, 가능한 경우 E00/E15
두 조건으로 제한한 Nested Group CV는 별도입니다. 실제 시간은 임베딩 캐시와
입력 해상도에 따라 달라지며, CUDA 추출 피크·추출 시간과 Logistic Regression
적합 시간을 따로 저장합니다.

선정 점수는 실험 시작 전에 고정합니다. Hard ROC-AUC 50%, F1 18%,
`1−Siamese FPR` 15%, Balanced Accuracy 7%, Recall 5%, `1−Fold Hard AUC 표준편차`
5%입니다. OOF에서 F1 최대, Precision–Recall 균형, FPR 제한(1%, 1.22%, 5%, 10%)
과 고정 0.5를 비교합니다. 제한 안에서 양성을 한 장도 찾지 못하면 결과에 표시합니다.
Holdout 점수로 임계값이나 순위를 수정하지 않습니다. Nested 진단은 처음부터
고정한 E00과 E15 두 후보에만 적용하며 전체 튜닝의 선택 편향을 정량화하지는 않습니다.

224·336·448이 모두 512픽셀 크롭을 공유합니다. DINOv2의 동적 입력과 위치
임베딩 보간을 실제 forward로 확인합니다. 첫 입력 224에서 최적 조건을 찾고
같은 조건으로 다른 해상도를 검증합니다. 학습에만 증강하며,
`augmentation_review.png`에서 학습 Fold 얼굴·귀·눈 주변의 변경을 살펴볼 수 있습니다.
원본 그룹 수가 허용할 때 E15에서 양성 촬영 그룹 하나를 통째로 줄인 비교를 기록합니다.

### 테스트, 저장 경로, 한계

학습 시드는 기본 42입니다. `TEST_SEED=None`이면 테스트가 난수 시드를 생성해 JSON에
기록합니다. 같은 시드는 같은 이미지 선택을 재현합니다. 일반 랜덤은 학습 사진을
포함할 수 있고, Reserved Holdout은 저장된 임계값과 모델을 사용합니다. Strict
Holdout은 동일 격리 집합·설정의 분류기를 재사용하며, 새 개발 그룹을 시험할 때
그 그룹 전체를 제외하고 나머지에서 임계값을 다시 보정합니다. 기존 개발 그룹을
제외해 재학습한 결과는 이전 하이퍼파라미터 탐색의 영향을 받을 수 있습니다.

새 외부 데이터의 세 클래스 폴더를 `EXTERNAL_SOURCE`로 지정하면 현재 체크포인트로
전체 평가합니다. EXIF 방향을 보정한 RGB 픽셀의 SHA-256으로 정확 중복을 막으며, 명확히 분리된 촬영 세션의 확보와
활용 여부는 외부 데이터 수집자가 확인해야 합니다. 기존 204장 Holdout은 이미
검토한 자료입니다. 개발 OOF는 선정에 사용한 성능입니다. Holdout에 충분한
촬영·개체 그룹이 있으면 그룹 부트스트랩 95% 구간을 저장합니다. 이 구간은
고정 모델·임계값과 주어진 그룹 분포에 조건부이며, 외부 데이터에서의 성능을
보장하지 않습니다.

```bash
D=/content/drive/MyDrive
B="$D/Cat_Project_Backup"
R=dinov2-final-review-1121-01
P=outputs/processed_1121
TRAIN_SEED=42 python -m src.study_data --source "$D" --archive "$B/processed_1121.zip" --output "$P"
TRAIN_SEED=42 python -m src.generalization_study --source "$D" --processed "$P" --output "$B" --run-id "$R"
python -m src.test_study_model --mode random --source "$D" --processed "$P" --output "$B" --run-id "$R" --random-seed
python -m src.plot_generalization --processed "$P" --output "$B" --run-id "$R"
```

기존 `python src/predict.py "이미지.jpg"`,
`python src/test_random_samples.py --random`,
`python src/test_random_samples.py --random --strict-holdout` 엔트리도 유지됩니다.

| Drive 상대 경로 | 용도 |
| --- | --- |
| `results/<RUN_ID>/legacy_conditions.csv`, `legacy_reevaluation.csv` | 원래 설정, 동일 조건의 ID, 새 Fold의 24개 비교 행 |
| `results/<RUN_ID>/additional_search.csv`, `fold_results.csv`, `resolution_comparison.csv` | 추가 후보·Fold·해상도 및 시간/메모리 |
| `results/<RUN_ID>/baseline_comparison.csv`, `final_model_metrics.json` | E00·E15·최종 모델, 예전 평가 부분집합 및 그룹 단위 불확실성 |
| `results/<RUN_ID>/threshold_comparison.csv`, `best_hyperparameters.json` | FPR 목표와 최종 후보 설정 |
| `checkpoints/<RUN_ID>/final_model.pt`, `strict_checkpoints/` | 최종 추론용 체크포인트와 Strict 재학습 캐시 |
| `experiments/<RUN_ID>/`, `embeddings/` | 후보별 즉시 저장 결과와 특징 벡터 |
| `graphs/<RUN_ID>/`, `graphs/<RUN_ID>/sample_tests/` | 발표용 PNG, 판별 이미지와 JSON |

환경·소스 해시와 Fold가 같으면 동일 RUN_ID에서 완료 후보가 재사용됩니다.
데이터·환경·설정이 달라지면 새 RUN_ID를 사용하세요. 실제 Drive 사진과 T4에서의
E15 대비 변화, GPU 피크와 외부 일반화 성능은 새 실험에서 측정합니다.
코드 결함과 수정 결과는 [최종 검토 보고서](REVIEW.md)에 기록했습니다.
