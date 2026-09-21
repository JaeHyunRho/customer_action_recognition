# 온디바이스 Skeleton 기반 고객 행동 인식

- 대상 논문: 노재현, 「온디바이스에서의 Skeleton 기반 고객 행동 인식 모델 개발」, 한신대학교 석사학위논문, 2025
- 구성: YOLOv11n-Pose (자세추정) → StrongSORT (추적) → ST-GCN / LSTM (행동 분류)
- 상태: **MERL Shopping Dataset 106개 영상으로 학습·검증 완료**

```
영상/카메라 → YOLOv11n-Pose → 머리 5개 키포인트 제거(12개) → StrongSORT
           → 트랙별 30프레임 버퍼 → ST-GCN 또는 LSTM → 5개 행동 라벨
```

- 인식 행동: `Reach To Shelf`, `Retract From Shelf`, `Hand In Shelf`, `Inspect Product`, `Inspect Shelf`

---

## 1. 핵심 결과

| 항목 | 본 구현 | 논문 |
|---|---|---|
| 자세추정 mAP-pose 50 | **0.927** | 0.90 |
| 자세추정 mAP-pose 50-95 | **0.885** | 0.66 |
| 행동 인식 정확도 (ST-GCN) | 0.701 | 0.93 |
| 행동 인식 정확도 (LSTM) | 0.694 | 0.85 |
| 자세추정 재학습 기여 | **+12.7%p** | — |
| 실시간 처리 속도 | 23.9 fps | 28~32 fps |
| 프로세스 메모리 | 1.32 GB | 5.8 GB |

- 상세 분석: [docs/RESULTS.md](docs/RESULTS.md)
- 논문 초안: [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md)

---

## 2. 빠른 시작

### 2.1 준비된 산출물

| 파일 | 내용 |
|---|---|
| `.venv/` | CUDA 동작하는 Jetson 전용 PyTorch 환경 |
| `weights/custom_yolo11n_pose.pt` | 상단 시점 재학습 자세추정 (mAP50 0.927) |
| `weights/yolo11n-pose.engine` | 기존 모델 TensorRT 엔진 |
| `runs/action/stgcn_custom/best.pt` | ST-GCN 행동 모델 (0.701) |
| `runs/action/lstm_custom/best.pt` | LSTM 행동 모델 (0.694) |
| `data/raw/merl/` | 영상 106개 + 라벨 106개 (1.8 GB) |
| `data/processed/merl_custom_primary.npz` | skeleton 134,898 프레임 |
| `figures/` | 논문용 그림 22종 + 주석 영상 |

- **Jetson에서는 `.venv/bin/python`을 사용할 것** (시스템 `python3`는 CPU 전용, §5 참조)

### 2.2 영상 파일로 실행

```bash
.venv/bin/python tools/run_recognition.py \
  --source data/videos/your_topview.mp4 \
  --pose-weights weights/custom_yolo11n_pose.pt \
  --weights runs/action/stgcn_custom/best.pt \
  --save-video runs/demo/annotated.mp4 \
  --save-csv runs/demo/actions.csv \
  --save-summary runs/demo/summary.json \
  --no-display
```

- 간편 실행: `bash scripts/run_video.sh data/videos/your_topview.mp4`

| 출력 | 내용 |
|---|---|
| `annotated.mp4` | skeleton, 트랙 ID, 행동 라벨 |
| `actions.csv` | 프레임별 `track_id, bbox, action, score` |
| `summary.json` | fps, 단계별 지연, 메모리, 행동별 카운트 |

- 행동 모델 없이도 실행 가능 (자세추정 + 추적만 수행)
- `--no-display`: SSH 등 화면 없는 환경용

### 2.3 카메라로 실행

```bash
# 연결 확인
.venv/bin/python -c "from car.utils.video import list_cameras; print(list_cameras())"

# USB 웹캠
bash scripts/run_camera.sh 0

# Jetson CSI(MIPI) 카메라
bash scripts/run_camera.sh 0 --csi

# IP 카메라
bash scripts/run_camera.sh "rtsp://user:pw@192.168.0.10:554/stream1"

# 화면 없는 환경
bash scripts/run_camera.sh 0 --no-display
```

- 결과는 `runs/live/<타임스탬프>/`에 저장
- CSI 카메라는 GStreamer `nvarguscamerasrc` 경유 (V4L2로는 열리지 않음)
- Jetson 튜닝 프로파일: `--config configs/jetson.yaml`

### 2.4 천장 카메라 설치 시 확인 사항

| 항목 | 기준 |
|---|---|
| 높이 | 어깨·팔·엉덩이가 모두 보일 것 (머리만 보이면 인식 불가) |
| 구도 | 선반이 프레임 상단에 오도록 배치 |
| 조명 | 역광 및 천장 조명 반사 회피 |
| 각도 | 학습 데이터와 다르면 §3의 자체 데이터 학습 권장 |

---

## 3. 학습

### 3.1 MERL Shopping 데이터 준비

- 이미 `data/raw/merl/`에 준비됨. 다른 장치에서 새로 받으려면:

```bash
mkdir -p data/raw/merl && cd data/raw/merl
BASE=https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset
curl -LO $BASE/Videos_MERL_Shopping_Dataset.zip    # 1.7 GB
curl -LO $BASE/Labels_MERL_Shopping_Dataset.zip
unzip -q Videos_MERL_Shopping_Dataset.zip && unzip -q Labels_MERL_Shopping_Dataset.zip
rm -rf __MACOSX *.zip
mv Videos_MERL_Shopping_Dataset videos && mv Labels_MERL_Shopping_Dataset labels
```

| 항목 | 값 |
|---|---|
| 구조 | `videos/{피험자}_{세션}_crop.mp4`, `labels/{피험자}_{세션}_label.mat` |
| 라벨 형식 | MATLAB `tlabs` 셀 배열, 행동별 `[시작, 끝]` 프레임 |
| 공식 split | 학습 1–20 (60개), 검증 21–26 (18개), 테스트 27–41 (28개) |

```bash
.venv/bin/python tools/extract_skeletons.py --merl data/raw/merl \
  --out data/processed/merl_skeletons.npz --fps 10 --keep-background \
  --set pose.weights=weights/custom_yolo11n_pose.pt
```

- 공식 split 사용: `--split train|val|test|trainval`

### 3.2 자체 촬영 영상 준비

- 행동별 폴더에 클립 배치:

```
data/videos/labeled/
  Reach To Shelf/     clip1.mp4 clip2.mp4
  Hand In Shelf/      clip3.mp4
```

```bash
.venv/bin/python tools/extract_skeletons.py --video-dir data/videos/labeled \
  --out data/processed/custom_skeletons.npz
```

- 클립 하나에 행동 하나면: `--video PATH --label "Reach To Shelf"`
- **한 명만 등장하는 영상이면 단일 트랙 병합 필수** (검출 끊김 시 윈도우 생성 실패)

```python
from car.datasets.sequence import SkeletonStore, keep_primary_track
s = keep_primary_track(SkeletonStore.load("data/processed/custom_skeletons.npz"))
s.save("data/processed/custom_primary.npz")
```

### 3.3 행동 모델 학습

```bash
# 두 모델 학습 + 비교
bash scripts/train_all.sh data/processed/merl_custom_primary.npz

# 개별 학습
.venv/bin/python -m car.train.train_action \
  --store data/processed/merl_custom_primary.npz --model stgcn \
  --set train.balance_ratio=1.5
```

| 출력 | 내용 |
|---|---|
| `best.pt` | 최고 성능 체크포인트 |
| `summary.json` | 정확도, F1, 혼동행렬, 설정 |
| `confusion_matrix.png` | 혼동행렬 |
| `history.png` | 학습 곡선 |

### 3.4 자세추정 모델 학습

```bash
# 1. 대형 모델로 12관절 자동 라벨링
.venv/bin/python tools/autolabel_pose.py --videos data/raw/merl/videos \
  --out data/processed/pose_dataset --teacher weights/yolo11x-pose.pt --fps 1

# 2. YOLOv11n-Pose 파인튜닝
.venv/bin/python -m car.train.train_pose \
  --data data/processed/pose_dataset/dataset.yaml \
  --set pose_train.imgsz=480 pose_train.batch_size=24
```

| 항목 | 값 |
|---|---|
| 교사 모델 | YOLOv11x-Pose |
| 채택 기준 | 신뢰도 ≥ 0.6, 확신 관절 ≥ 8개 |
| 생성 라벨 | 10,347장 (기각 4,385장) |
| 수렴 | 15 epoch |

### 3.5 평가 및 시각화

```bash
# 모델 비교 (논문 표 4.4 형식)
.venv/bin/python -m car.train.evaluate \
  --checkpoint runs/action/stgcn_custom/best.pt runs/action/lstm_custom/best.pt \
  --store data/processed/merl_custom_primary.npz --out runs/eval

# 추적 기여도 평가 (논문 표 2.2 형식)
.venv/bin/python tools/eval_tracking.py --merl data/raw/merl --clips 12

# 영상 기반 그림 (몽타주, 타임라인, 자세추정 비교)
.venv/bin/python tools/make_figures.py \
  --video data/raw/merl/videos/1_2_crop.mp4 \
  --csv figures/1_2_actions.csv --out figures

# 분석 그림 15종 (통계, ablation, 런타임)
.venv/bin/python tools/make_analysis_figures.py --out figures
```

**영상 기반 그림** (`tools/make_figures.py`)

| 파일 | 내용 |
|---|---|
| `fig_actions_montage.png` | 행동별 대표 프레임 5장 |
| `fig_timeline.png` | 시간축 예측 vs 정답 |
| `fig_pose_comparison.png` | 기존 vs 재학습 자세추정 |
| `fig_confusion_*.png` | 혼동행렬 |
| `fig_history_*.png` | 학습 곡선 |

**분석 그림** (`tools/make_analysis_figures.py`)

| 파일 | 내용 |
|---|---|
| `fig_skeleton_layout.png` | 12관절 배치와 인접 행렬 |
| `fig_head_removal.png` | 안면 관절 제거 전후 |
| `fig_class_distribution.png` | 클래스 분포 실측 vs 논문 |
| `fig_track_continuity.png` | 추적 구간 길이 분포 |
| `fig_ablation_normalize.png` | 정규화 방식 비교 |
| `fig_ablation_rotation.png` | 회전 증강 범위 비교 |
| `fig_ablation_split.png` | 분할 단위 비교 + 중복 구조 |
| `fig_pose_training.png` | 자세추정 mAP 학습 곡선 |
| `fig_pose_effect.png` | 자세추정에 따른 데이터·정확도 변화 |
| `fig_class_f1.png` | 클래스별 F1 |
| `fig_model_tradeoff.png` | 파라미터 vs 정확도 vs 지연 |
| `fig_runtime_stages.png` | 단계별 지연 분해 |
| `fig_runtime_fps.png` | 백엔드별 처리 속도 |
| `fig_quantization.png` | int8 양자화 영향 |
| `fig_sequence_examples.png` | 행동별 손목 궤적 |

---

## 4. 데이터 없이 파이프라인 검증

```bash
.venv/bin/python tools/make_demo_skeletons.py --out data/processed/demo_skeletons.npz
.venv/bin/python -m car.train.train_action --store data/processed/demo_skeletons.npz \
  --model stgcn --set train.epochs=30
```

- `car/synthetic.py`가 상단 시점 5가지 행동의 관절 움직임을 합성
- **테스트 픽스처이며 데이터가 아님.** 정확도는 배관 검증 의미만 있음

---

## 5. Jetson 환경

### 5.1 검증 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin NX 16GB (Super) |
| L4T / JetPack | 36.4.4 / JetPack 6.2 |
| Python / PyTorch | 3.10.12 / 2.9.1 (Jetson 전용 휠) |
| CUDA / TensorRT | 12.6.68 / 10.3.0.30 |
| OpenCV | 5.0 (JetPack 제공) |
| 전력 모드 | 25 W |

- 논문 환경은 Jetson AGX Xavier 32GB / CUDA 11.4 → fps·메모리 직접 비교 불가

### 5.2 GPU 활성화 (중요)

- **pip 기본 `torch`는 Jetson에서 GPU를 인식하지 못함** (`cuda.is_available()` → `False`)
- 코드는 동작하나 전부 CPU로 실행되어 매우 느림

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True여야 함
```

- 새 장치 설치: `bash scripts/setup_jetson.sh`
- 수동 설치 시 주의사항:

```bash
python3 -m venv --system-site-packages .venv
# python3-venv 없으면: pip install --user virtualenv && python3 -m virtualenv --system-site-packages .venv

# ⚠️ --extra-index-url로 PyPI를 함께 주면 pip가 PyPI의 CPU 휠을 선택함
.venv/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.9.1 torchvision==0.24.1

# ⚠️ libcudss.so.0 누락 문제
.venv/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  nvidia-cudss-cu12

.venv/bin/pip install --no-deps ultralytics ultralytics-thop
.venv/bin/pip install filelock typing-extensions sympy networkx jinja2 fsspec pillow \
  numpy scipy pandas PyYAML matplotlib tqdm psutil py-cpuinfo polars nvidia-ml-py pytest
```

| JetPack | CUDA | 인덱스 |
|---|---|---|
| 6.x | 12.6 | `jp6/cu126` |
| 6.x | 12.8 | `jp6/cu128` |
| 5.x | 11.8 | `jp5/cu118` |

- cuDSS는 `site-packages/nvidia/cu12/lib`에 설치되어 동적 링커가 찾지 못함
- 본 저장소는 `_car_jetson_boot.py`와 동명 `.pth`로 인터프리터 시작 시 선로딩 처리

### 5.3 측정 성능

**자세추정 단독** (640 px, 4인 프레임)

| 백엔드 | 지연 | fps |
|---|---|---|
| PyTorch FP32 | 101 ms | 9.9 |
| PyTorch FP16 | 89 ms | 11.2 |
| TensorRT FP16 | 45 ms | 22.3 |
| Custom pose FP32 | 40 ms | 24.8 |

**전체 파이프라인** (920×680)

| 구성 | fps | 프레임당 |
|---|---|---|
| Custom pose + LSTM | **23.9** | 37 ms |
| Custom pose + ST-GCN | 22.7 | 44 ms |
| 기존 pose + ST-GCN (TensorRT) | 18.6 | 54 ms |

### 5.4 성능 튜닝 순서

| 순위 | 항목 | 명령 | 효과 |
|---|---|---|---|
| 1 | 전력 모드 | `sudo nvpmodel -m 0 && sudo jetson_clocks` | 최대 |
| 2 | TensorRT 변환 | `tools/export_models.py --pose ... --format engine --half` | 2.25배 |
| 3 | 해상도 축소 | `--set pose.imgsz=480` | 1.3배 |
| 4 | 분류 주기 | `--set runtime.action_every=3` | 소폭 |
| 5 | FP16 | `--set pose.half=true` | 1.1배 |
| 6 | 모델 선택 | LSTM 사용 | 1.05배 |

- 모니터링: `jtop`

---

## 6. 배포용 변환

```bash
# ONNX
.venv/bin/python tools/export_models.py --checkpoint runs/action/stgcn_custom/best.pt --format onnx

# TensorRT (Jetson 권장)
.venv/bin/python tools/export_models.py --pose weights/custom_yolo11n_pose.pt --format engine --half

# int8 동적 양자화
.venv/bin/python tools/export_models.py --checkpoint runs/action/lstm_custom/best.pt --format torch-int8
```

| 경로 | 대상 | 비고 |
|---|---|---|
| TensorRT FP16/INT8 | Jetson **(권장)** | GPU 사용 |
| int8 TFLite | 논문 재현 | `onnx2tf` + `tensorflow` 필요 |
| PyTorch dynamic int8 | 툴체인 없는 CPU | Linear/LSTM만 대상 |

**int8 양자화 측정 결과**

| 모델 | 크기 | 정확도 | CPU 지연 |
|---|---|---|---|
| LSTM fp32 → int8 | 1.42 → **0.39 MB** | 0.746 → 0.748 | 36 → **73.6 ms** |
| ST-GCN fp32 → int8 | 4.13 → 4.13 MB | 0.757 → 0.757 | 376 → 402 ms |

- LSTM은 크기 73% 감소, 정확도 유지, **CPU 지연 2배 증가**
- ST-GCN은 Conv2d 기반이라 동적 양자화 효과 없음
- 상세: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 7. 논문 미기재 항목과 구현 결정

| 항목 | 논문 | 본 구현 | 근거 |
|---|---|---|---|
| Skeleton 정규화 | 미기재 | 몸통 기준 | 0.849 vs bbox 0.831 vs 없음 0.644 |
| 학습 데이터 증강 | 미기재 | 회전 ±30° | 0.839 vs ±180° 0.784 vs 없음 0.730 |
| train/val 분할 단위 | "70/30"만 | 클립 단위 | 윈도우 단위는 25프레임 중복 누수 |
| 윈도우 라벨 | 미기재 | 마지막 프레임 | 실시간 추론 조건과 일치 |
| grouped conv 그룹 수 | "grouped conv" | 2 | 파라미터 922,301개 역산 |
| LSTM Dense 폭 | 미기재 | 32 | 파라미터 352,775개 역산 |
| StrongSORT ReID | 미기재 | HSV 히스토그램 | 온디바이스 2차 신경망 배제 |
| pose 라벨 제작 | "YOLO 포맷" | 대형 모델 pseudo-labeling | 126,350장 수작업 비현실적 |

**파라미터 수 대조**

| 모델 | 논문 | 본 구현 | 차이 |
|---|---|---|---|
| LSTM | 352,775 | 353,549 | +0.2% |
| ST-GCN | 922,301 | 1,018,413 | +10.4% |

- 상세: [docs/PAPER_GAPS.md](docs/PAPER_GAPS.md)

---

## 8. 프로젝트 구조

```
car/
  config.py          설정 dataclass (# CHOICE = 논문 미기재 항목)
  keypoints.py       COCO-17 → 12관절, 머리 제거, 스켈레톤 그래프
  synthetic.py       합성 top-view 고객 (테스트 픽스처)
  datasets/
    merl.py          MERL .mat 라벨 파서, 공식 split
    extract.py       영상 → skeleton store
    sequence.py      store 저장, 윈도우 생성, 단일 트랙 병합
  models/
    graph.py         ST-GCN 인접행렬 분할
    stgcn.py         논문 그림 3.1 구조
    lstm.py          논문 그림 3.2 구조
  pose/
    estimator.py     YOLOv11n-Pose 래퍼 + 머리 제거
    tracker.py       StrongSORT 직접 구현
  pipeline/
    normalize.py     스켈레톤 정규화 3종
    buffer.py        트랙별 30프레임 버퍼 + 평활화
    realtime.py      end-to-end 실행기
  train/             train_pose, train_action, evaluate
  export/            ONNX / TensorRT / TFLite / int8
  utils/             video, viz, metrics
tools/               CLI 진입점
configs/             default.yaml, jetson.yaml
docs/                RESULTS, PAPER_DRAFT, PAPER_GAPS, EXPERIMENTS, DEPLOYMENT, PIPELINE
```

---

## 9. 테스트

```bash
.venv/bin/python -m pytest tests/ -v
```

- 42개 테스트: 키포인트 처리, 그래프 구성, 정규화 불변성, 추적기 상태 전이, 버퍼 동작, 모델 입출력, 설정 로딩, 메트릭 계산

---

## 10. 문서

| 문서 | 내용 |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | 실험 절차, 결과, 논문 대비 변경점 요약 |
| [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md) | 저널 투고용 국문 논문 초안 |
| [docs/PAPER_GAPS.md](docs/PAPER_GAPS.md) | 논문 미기재 항목 전체 목록 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 설계 선택 실험 상세 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 엣지 배포, 카메라 연결, 튜닝 |
| [docs/PIPELINE.md](docs/PIPELINE.md) | 단계별 구현 대응표 |
