# 온디바이스 Skeleton 기반 고객 행동 인식

노재현, 「온디바이스에서의 Skeleton 기반 고객 행동 인식 모델 개발」(한신대학교 석사학위논문, 2025)의
실행 가능한 구현입니다. 자세추정은 **YOLOv11n-Pose**, 추적은 **StrongSORT**, 행동 분류는
**ST-GCN / LSTM**을 사용합니다.

```
영상/카메라 → YOLOv11n-Pose → 머리 5개 키포인트 제거(12개) → StrongSORT
           → 트랙별 30프레임 버퍼 → ST-GCN 또는 LSTM → 5개 행동 라벨
```

인식 대상 행동 5가지: `Reach To Shelf`, `Retract From Shelf`, `Hand In Shelf`,
`Inspect Product`, `Inspect Shelf`

---

## 1. 빠른 시작

이 저장소는 **바로 실행 가능한 상태**입니다. 이미 준비된 것:

- `.venv/` — CUDA가 동작하는 Jetson 전용 PyTorch 환경
- `weights/yolo11n-pose.pt`, `weights/yolo11n-pose.engine` — TensorRT 엔진 포함
- `runs/action/{stgcn,lstm}/best.pt` — 합성 데이터로 학습한 행동 모델 (§3 참고)

**Jetson에서는 `.venv/bin/python`을 쓰십시오.** 시스템 `python3`는 CPU 전용
PyTorch라 느립니다. 자세한 이유는 §4에 있습니다.

### 영상 파일로 실행 (카메라 없을 때)

top-view 고객 영상을 `data/videos/`에 넣고:

```bash
.venv/bin/python tools/run_recognition.py \
  --source data/videos/your_topview.mp4 \
  --pose-weights weights/yolo11n-pose.engine \
  --weights runs/action/stgcn/best.pt \
  --save-video runs/demo/annotated.mp4 \
  --save-csv runs/demo/actions.csv \
  --save-summary runs/demo/summary.json \
  --no-display
```

한 줄로는:

```bash
bash scripts/run_video.sh data/videos/your_topview.mp4
```

`--no-display`는 SSH 등 화면 없는 환경용입니다. 결과물은 세 가지입니다.

| 파일 | 내용 |
|---|---|
| `annotated.mp4` | skeleton, 트랙 ID, 행동 라벨이 그려진 영상 |
| `actions.csv` | 프레임별 `track_id, bbox, action, score` |
| `summary.json` | fps, 단계별 지연시간, 메모리, 행동별 카운트 |

**행동 모델 체크포인트가 없어도 실행됩니다.** 이 경우 자세추정과 추적만 수행하고
경고를 출력합니다. 파이프라인 연결과 속도를 먼저 확인할 때 유용합니다.

### 새 장치에 설치할 때

```bash
bash scripts/setup_jetson.sh      # Jetson (§4에 주의사항)
pip install -r requirements.txt   # 데스크탑
```

### 카메라로 실행 (다른 사람이 카메라 연결했을 때)

```bash
# 연결된 카메라 확인
.venv/bin/python -c "from car.utils.video import list_cameras; print(list_cameras())"

# USB 웹캠
bash scripts/run_camera.sh 0

# Jetson CSI(MIPI) 카메라 — OpenCV 기본 경로로는 안 열리므로 --csi 필수
bash scripts/run_camera.sh 0 --csi

# IP 카메라
bash scripts/run_camera.sh "rtsp://user:pw@192.168.0.10:554/stream1"

# 화면 없는 환경(SSH)이면 --no-display를 덧붙이십시오
bash scripts/run_camera.sh 0 --no-display
```

`scripts/run_camera.sh`는 `/dev/video*`를 먼저 확인하고, 카메라가 없으면
USB인지 CSI인지 구분해 안내합니다. 결과는 `runs/live/<타임스탬프>/`에
영상·CSV·요약으로 저장됩니다.

해상도와 프레임레이트 조정:

```bash
.venv/bin/python tools/run_recognition.py --source 0 \
  --set runtime.camera_width=1920 runtime.camera_height=1080 runtime.camera_fps=30
```

Jetson에 맞춰 둔 프로파일도 있습니다.

```bash
.venv/bin/python tools/run_recognition.py --config configs/jetson.yaml --source 0
```

천장 카메라 설치 시 확인할 것: 사람 전신이 프레임에 들어오는 높이,
선반이 프레임 상단에 오도록 배치, 역광 회피. 학습 데이터와 카메라 각도가
다르면 정확도가 크게 떨어지므로, 새 설치 환경에서는 §4의 자체 데이터 학습을
권장합니다.

---

## 2. 학습

### 2-1. 데이터 준비: MERL Shopping (논문 방식)

논문은 MERL Shopping Dataset(106개 영상, 각 약 2분)을 10 fps로 분할해
126,350장을 만들었습니다. 데이터셋은 MERL 웹사이트에서 직접 받아야 합니다.

```
data/raw/merl/
  videos/   1_1_crop.mp4 ...
  labels/   1_1_label.mat ...
```

skeleton 추출:

```bash
.venv/bin/python tools/extract_skeletons.py --merl data/raw/merl \
  --out data/processed/merl_skeletons.npz --fps 10
```

### 2-2. 데이터 준비: 직접 촬영한 영상

행동별 폴더에 짧은 클립을 넣으면 됩니다.

```
data/videos/labeled/
  Reach To Shelf/     clip1.mp4 clip2.mp4
  Hand In Shelf/      clip3.mp4
  Inspect Product/    ...
```

```bash
.venv/bin/python tools/extract_skeletons.py --video-dir data/videos/labeled \
  --out data/processed/custom_skeletons.npz
```

클립 하나에 행동 하나면 `--video PATH --label "Reach To Shelf"`도 됩니다.

### 2-3. 행동 모델 학습

```bash
.venv/bin/python -m car.train.train_action --store data/processed/merl_skeletons.npz --model stgcn
.venv/bin/python -m car.train.train_action --store data/processed/merl_skeletons.npz --model lstm
```

두 모델을 학습하고 논문 Table 4.4 형식으로 비교까지 한 번에:

```bash
bash scripts/train_all.sh data/processed/merl_skeletons.npz
EPOCHS=50 bash scripts/train_all.sh data/processed/merl_skeletons.npz   # 짧게
```

논문 Table 4.2 그대로 300 epoch, batch 128, Adam, CCE, dropout 0.25, early stopping을
기본값으로 씁니다. 결과는 `runs/action/{stgcn,lstm}/`에 저장됩니다.
`best.pt`, `summary.json`, `confusion_matrix.png`, `history.png`.

### 2-4. 평가 및 모델 비교

```bash
.venv/bin/python -m car.train.evaluate \
  --checkpoint runs/action/stgcn/best.pt runs/action/lstm/best.pt \
  --store data/processed/merl_skeletons.npz --out runs/eval
```

논문 Table 4.4 형식의 비교표를 출력합니다.

### 2-5. Custom YOLOv11n-Pose 학습 (선택)

논문은 top-view 데이터로 YOLOv11-Pose를 재학습해 mAP50을 0.84→0.90으로 올렸습니다.
논문이 "YOLO 포맷으로 레이블 제작"이라고만 밝힌 부분을, 큰 pose 모델을 교사로 쓰는
pseudo-labeling으로 구현했습니다.

```bash
# 1) 큰 모델로 자동 라벨링 (머리 키포인트 제거된 12개 포맷)
.venv/bin/python tools/autolabel_pose.py --videos data/raw/merl/videos \
  --out data/processed/pose_dataset --teacher yolo11x-pose.pt --fps 2

# 2) YOLOv11n-Pose 파인튜닝
.venv/bin/python -m car.train.train_pose --data data/processed/pose_dataset/dataset.yaml

# 3) 학습된 가중치로 실행
.venv/bin/python tools/run_recognition.py --source data/videos/x.mp4 \
  --pose-weights weights/custom_yolo11n_pose.pt
```

---

## 3. 데이터 없이 파이프라인 검증

MERL 데이터도 카메라도 없는 상태에서 코드 전체가 도는지 확인하려면:

```bash
.venv/bin/python tools/make_demo_skeletons.py --out data/processed/demo_skeletons.npz
.venv/bin/python -m car.train.train_action --store data/processed/demo_skeletons.npz \
  --model stgcn --set train.epochs=30
.venv/bin/python tools/make_demo_video.py --out data/videos/demo_topview.mp4
```

`car/synthetic.py`가 천장 시점에서 본 5가지 행동의 관절 움직임을 합성합니다.
**이것은 테스트 픽스처이지 데이터가 아닙니다.** 여기서 나온 정확도는 배관이
연결됐다는 뜻일 뿐, 실제 고객 인식 성능과 무관합니다.

---

## 4. Jetson에서 사용하기

### 이 저장소가 검증된 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin NX 16GB (Super) |
| L4T / JetPack | 36.4.4 / JetPack 6.2 |
| Python | 3.10.12 |
| PyTorch | **2.9.1 (Jetson 전용 휠, CUDA 12.6) — GPU 동작 확인** |
| CUDA 드라이버 | 12.6.68 |
| TensorRT | 10.3.0.30 |
| OpenCV | 5.0 (JetPack 제공) |

논문은 Jetson AGX Xavier 32GB / CUDA 11.4 환경입니다. 보드가 다르므로
논문 표 4.5의 fps·메모리 수치와 직접 비교되지 않습니다.

### ⚠️ GPU를 쓰려면 Jetson 전용 PyTorch가 필요합니다

**이 보드에서 가장 먼저 걸리는 문제입니다.** pip가 기본으로 설치하는 `torch`는
Jetson 드라이버와 CUDA 버전이 맞지 않아 `torch.cuda.is_available()`이 조용히
`False`를 반환합니다. 코드는 그대로 돌지만 전부 CPU에서 실행됩니다.

이 저장소는 프로젝트 안의 `.venv`에 Jetson 전용 휠을 설치해 두었습니다.
**GPU를 쓰려면 `.venv/bin/python`으로 실행하십시오.**

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
.venv/bin/python tools/run_recognition.py --source data/videos/your_clip.mp4
```

시스템 `python3`는 CPU 전용 torch를 쓰므로 느립니다. 둘 다 동작하지만 속도가 다릅니다.

### 다른 Jetson에 새로 설치할 때

```bash
bash scripts/setup_jetson.sh
```

이 스크립트가 하는 일과, 직접 할 때 주의할 점:

```bash
python3 -m venv --system-site-packages .venv     # JetPack의 OpenCV/TensorRT 재사용
# python3-venv가 없으면: pip install --user virtualenv && python3 -m virtualenv --system-site-packages .venv

# ⚠️ --extra-index-url로 PyPI를 함께 주면 pip가 PyPI의 같은 버전 CPU 휠을
#    골라버립니다. Jetson 인덱스만 쓰고 --no-deps로 설치하십시오.
.venv/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.9.1 torchvision==0.24.1

.venv/bin/pip install --no-deps ultralytics ultralytics-thop
.venv/bin/pip install filelock typing-extensions sympy networkx jinja2 fsspec pillow \
  numpy scipy pandas PyYAML matplotlib tqdm psutil py-cpuinfo polars nvidia-ml-py pytest
```

JetPack 버전별 인덱스: JetPack 6/CUDA 12.6은 `jp6/cu126`,
CUDA 12.8은 `jp6/cu128`, JetPack 5는 `jp5/cu118`입니다.

**cuDSS 문제.** Jetson torch 휠은 `libcudss.so.0`을 요구하는데 JetPack이
설치해 주지 않습니다. 그대로 두면 `import torch`가 실패합니다.

```bash
.venv/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  nvidia-cudss-cu12
```

설치해도 라이브러리가 `site-packages/nvidia/cu12/lib`에 들어가 동적 링커가 찾지
못합니다. 이 저장소는 `.venv/lib/python3.10/site-packages/_car_jetson_boot.py`와
같은 이름의 `.pth` 파일로 인터프리터 시작 시 미리 로드하도록 처리했습니다.
수동으로 하려면 `LD_LIBRARY_PATH`에 해당 경로를 넣으십시오.

### 측정된 성능 (Orin NX, 25W 모드, GPU)

**자세추정 단독** — 4명이 있는 640 입력 이미지, 15회 평균.

| 백엔드 | 지연 | fps |
|---|---|---|
| PyTorch FP32 | 101 ms | 9.9 |
| PyTorch FP16 | 89 ms | 11.2 |
| **TensorRT FP16 엔진** | **45 ms** | **22.3** |
| PyTorch FP32, imgsz=480 | 78 ms | 12.9 |

**전체 파이프라인** — 자세추정 + StrongSORT + 버퍼링, 960×540 영상 120프레임.

| 백엔드 | fps | 프레임당 |
|---|---|---|
| PyTorch FP32 | 11.5 | 87 ms |
| **TensorRT FP16 엔진** | **18.3** | **55 ms** |

TensorRT 변환이 자세추정을 2.25배, 전체 파이프라인을 1.6배 빠르게 했습니다.
엔진은 이미 만들어 두었습니다.

```bash
.venv/bin/python tools/run_recognition.py --source data/videos/x.mp4 \
  --pose-weights weights/yolo11n-pose.engine
```

**전력 모드가 25W로 제한되어 있습니다.** 위 수치는 최대 성능이 아닙니다.
아래 튜닝의 첫 항목을 적용하면 더 올라갑니다.

### 성능 튜닝 순서

효과가 큰 순서입니다. 바꾸기 전후로 `tools/benchmark.py`로 측정하십시오.

1. **전력 모드 최대화 — 가장 큰 효과.** 비밀번호가 필요해 이 저장소에서는
   적용하지 못했습니다. 직접 실행하십시오.
   ```bash
   sudo nvpmodel -q          # 현재 모드 확인 (이 보드는 25W)
   sudo nvpmodel -m 0        # 최대 성능
   sudo jetson_clocks        # 클럭 고정 (재부팅하면 풀립니다)
   ```
2. **TensorRT 엔진** — pose 모델이 병목이므로 여기가 가장 효과적입니다. §5 참조.
3. **입력 해상도 축소** — `--set pose.imgsz=480` (측정값 기준 약 1.3배)
4. **행동 분류 주기** — `--set runtime.action_every=3`
   (30프레임 버퍼는 매 프레임 크게 바뀌지 않습니다)
5. **FP16** — `--set pose.half=true` (측정값 기준 약 1.1배, 전력 모드를 올리면 더 큼)
6. **모델 선택** — LSTM이 ST-GCN보다 파라미터가 약 66만 개 적습니다

모니터링은 `jtop`으로 합니다.

### CSI 카메라

USB 웹캠과 달리 CSI(MIPI) 카메라는 GStreamer `nvarguscamerasrc`를 거쳐야 합니다.
`--csi` 플래그가 이를 처리합니다. 파이프라인은
`car/utils/video.py`의 `csi_gstreamer_pipeline()`에서 조정할 수 있습니다.
카메라가 천장에 뒤집혀 설치된 경우 `flip_method`를 바꾸십시오.

---

## 5. 배포용 변환

```bash
.venv/bin/python tools/export_models.py --checkpoint runs/action/stgcn/best.pt --format onnx
.venv/bin/python tools/export_models.py --checkpoint runs/action/stgcn/best.pt \
  --format tensorrt --precision fp16
.venv/bin/python tools/export_models.py --pose weights/yolo11n-pose.pt --format engine --half
```

**논문은 int8 TFLite로 양자화합니다.** TFLite는 일반 엣지 보드에는 타당하지만
Jetson에서는 GPU를 놀립니다. 이 저장소는 두 경로를 모두 지원하되 Jetson 기본값은
TensorRT입니다. 자세한 선택 기준은 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)에 있습니다.

---

## 6. 벤치마크

논문 Table 4.5(fps, RAM, GPU 메모리)에 대응하는 수치를 측정합니다.

```bash
.venv/bin/python tools/benchmark.py --source data/videos/demo_topview.mp4 \
  --frames 300 --out runs/bench/orin_nx.json
```

데스크탑에서도 같은 명령을 돌리면 논문 Fig. 4.3의 데스크탑 대 온디바이스 비교를
재현할 수 있습니다.

---

## 7. 논문에 없어서 직접 정한 것들

논문이 명시하지 않은 항목은 모두 근거와 함께 기록했습니다. 전체 목록과 실험 결과는
[docs/PAPER_GAPS.md](docs/PAPER_GAPS.md)와 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)에
있고, 코드에서는 `car/config.py`의 `# CHOICE` 주석으로 표시됩니다. 주요 항목:

| 항목 | 논문 | 이 구현 | 근거 |
|---|---|---|---|
| skeleton 정규화 | 언급 없음 | 몸통 기준(`torso`) | 측정: 0.849 vs bbox 0.831 vs 없음 0.644 |
| 학습 데이터 증강 | 언급 없음 | 회전 ±30°, 스케일 ±15%, 좌우 반전 | 측정: 0.839 vs ±180° 0.784 vs 없음 0.730 |
| train/val 분할 단위 | "70/30"만 명시 | 클립 단위 | 프레임 단위는 윈도우 중첩으로 누수 |
| grouped conv 그룹 수 | "grouped convolution" | 2 | 논문 파라미터 수 922,301개에서 역산 |
| StrongSORT ReID | 명시 없음 | HSV 색 히스토그램 | 온디바이스에 두 번째 망을 안 얹음 |
| 윈도우 라벨 결정 | 언급 없음 | 마지막 프레임 | 실시간 추론 조건과 일치 |
| pose 라벨 제작 방법 | "YOLO 포맷 이용" | 대형 pose 모델 pseudo-labeling | 126,350장 수작업은 비현실적 |

측정값은 합성 데이터(§3) 기준이며 실제 고객 인식 정확도가 아닙니다.
설계 선택의 방향을 정하기 위한 것입니다. 전체 결과는
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)에 있습니다.

### 파라미터 수 대조

| 모델 | 논문 | 이 구현 | 차이 |
|---|---|---|---|
| LSTM | 352,775 | 353,549 | +0.2% |
| ST-GCN | 922,301 | 1,018,413 | +10.4% |

LSTM은 논문 구조(3층 × 128채널 + Dense)로 거의 일치합니다. ST-GCN은 논문이 grouped
convolution의 그룹 수와 시간축 커널 크기를 밝히지 않아 정확히 맞추지 못했습니다.
`stgcn.groups`와 `stgcn.temporal_kernel`로 조정할 수 있습니다.

---

## 8. 프로젝트 구조

```
car/
  config.py          설정 dataclass + YAML (# CHOICE 주석이 논문 미기재 항목)
  keypoints.py       COCO-17 → 12개 관절, 머리 제거, 스켈레톤 그래프, OKS 가중치
  synthetic.py       합성 top-view 고객 (테스트 픽스처)
  datasets/
    merl.py          MERL Shopping .mat 라벨 파서
    extract.py       영상 → skeleton store
    sequence.py      store 저장 + 윈도우 Dataset
  models/
    graph.py         ST-GCN 인접행렬 분할
    stgcn.py         논문 Fig. 3.1 구조
    lstm.py          논문 Fig. 3.2 구조
  pose/
    estimator.py     YOLOv11n-Pose 래퍼 + 머리 제거
    tracker.py       StrongSORT 직접 구현 (Kalman, cascade, IoU, EMA)
  pipeline/
    normalize.py     스켈레톤 정규화 3종
    buffer.py        트랙별 30프레임 버퍼 + 예측 평활화
    realtime.py      end-to-end 실행기
  train/             train_pose, train_action, evaluate
  export/            ONNX / TensorRT / TFLite / int8
  utils/             video(파일·웹캠·CSI·RTSP), viz, metrics
tools/               실행 스크립트 (CLI 진입점)
configs/default.yaml 기본 설정
docs/                PAPER_GAPS, EXPERIMENTS, DEPLOYMENT
```

## 9. 테스트

```bash
.venv/bin/python -m pytest tests/ -v
```

42개 테스트가 키포인트 처리, 그래프 구성, 정규화의 불변성, 추적기 상태 전이,
버퍼 동작, 모델 입출력, 설정 로딩, 메트릭 계산을 검증합니다.

## 10. 더 읽을 것

| 문서 | 내용 |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | 단계별 상세, 논문 그림 2.1과 코드의 대응 |
| [docs/PAPER_GAPS.md](docs/PAPER_GAPS.md) | 논문 미기재 항목과 구현 결정 전체 목록 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 설계 선택을 정한 실험과 런타임 측정 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 엣지 배포, 카메라 연결, 속도 튜닝, 문제 해결 |
