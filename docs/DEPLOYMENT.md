# 배포 가이드

이 프로젝트를 엣지 장치에 올리고, 카메라를 붙이고, 속도를 끌어올리는 방법입니다.

---

## 1. 이 저장소가 검증된 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin NX 16GB (Super) |
| L4T / JetPack | 36.4.4 / JetPack 6.2 |
| OS | Ubuntu 22.04, Kernel 5.15.148-tegra |
| Python | 3.10.12 |
| CUDA 드라이버 | 12.6.68 |
| cuDNN | 9.3.0 |
| TensorRT | 10.3.0.30 |
| OpenCV | 5.0 (JetPack 제공) |

논문 환경은 Jetson AGX Xavier 32GB / CUDA 11.4입니다. 보드가 다르므로
논문 표 4.5의 fps·메모리 수치와 직접 비교되지 않습니다.

---

## 2. PyTorch: GPU를 쓰려면 Jetson 전용 휠이 필요합니다

이것이 이 보드에서 가장 먼저 걸리는 문제입니다.

pip가 기본으로 설치하는 `torch`는 x86/일반 CUDA 빌드라서 Jetson 드라이버와 맞지
않고, `torch.cuda.is_available()`이 조용히 `False`를 반환합니다. 코드는 그대로
돌지만 전부 CPU에서 실행됩니다.

확인:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False`라면 아래처럼 전용 휠을 설치하십시오.

### 권장: 가상환경에 설치

시스템 PyTorch를 덮어쓰면 다른 프로젝트가 깨질 수 있으므로 분리합니다.

```bash
cd ~/customer_action_recognition
python3 -m venv --system-site-packages .venv

.venv/bin/pip install \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --extra-index-url https://pypi.org/simple \
  torch==2.9.1 torchvision==0.24.1

.venv/bin/pip install --no-deps ultralytics ultralytics-thop
.venv/bin/pip install matplotlib tqdm psutil pyyaml scipy pandas polars py-cpuinfo pytest

.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
```

`--system-site-packages`를 쓰는 이유는 JetPack이 제공하는 OpenCV와 TensorRT를
그대로 쓰기 위해서입니다. 이후 모든 명령을 `.venv/bin/python`으로 실행하십시오.

### JetPack 버전별 인덱스

| JetPack | CUDA | 인덱스 |
|---|---|---|
| 6.x | 12.6 | `https://pypi.jetson-ai-lab.io/jp6/cu126` |
| 6.x | 12.8 | `https://pypi.jetson-ai-lab.io/jp6/cu128` |
| 5.x | 11.8 | `https://pypi.jetson-ai-lab.io/jp5/cu118` |

보드의 CUDA 버전은 `jetson_release` 또는 `nvcc --version`으로 확인합니다.

---

## 3. 전력 모드 — 가장 효과가 큰 설정

Jetson은 기본적으로 낮은 전력 모드로 출하됩니다. 최대 성능 모드로 바꾸면
다른 어떤 최적화보다 큰 차이가 납니다.

```bash
sudo nvpmodel -q              # 현재 모드 확인
sudo nvpmodel -m 0            # 최대 성능 (Orin NX는 25W 또는 MAXN)
sudo jetson_clocks            # 클럭 고정
sudo jetson_clocks --show     # 확인
```

재부팅하면 `jetson_clocks`는 풀립니다. 상시 적용하려면 systemd 서비스로 등록하십시오.

발열을 확인하면서 쓰십시오. 팬 없는 케이스에서 MAXN으로 장시간 돌리면
스로틀링이 걸려 오히려 느려집니다. `jtop`으로 온도를 봅니다.

---

## 4. 카메라 연결

### USB 웹캠 (V4L2)

```bash
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext    # 지원 해상도/fps 확인

python3 tools/run_recognition.py --source 0 --weights runs/action/stgcn/best.pt
```

연결된 카메라를 코드로 확인:

```bash
python3 -c "from car.utils.video import list_cameras; print(list_cameras())"
```

MJPG를 지원하는 카메라라면 YUYV보다 높은 fps가 나옵니다. 필요하면
`car/utils/video.py`의 `FrameSource.__init__`에서 `CAP_PROP_FOURCC`를 설정하십시오.

### CSI (MIPI) 카메라

USB와 달리 CSI 카메라는 OpenCV의 기본 경로로 열리지 않습니다. GStreamer의
`nvarguscamerasrc`를 거쳐야 하며, `--csi` 플래그가 이를 처리합니다.

```bash
# 카메라 인식 확인
ls /dev/video*
gst-inspect-1.0 nvarguscamerasrc

# 단독 테스트
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! nvegltransform ! nveglglessink

# 파이프라인 실행
python3 tools/run_recognition.py --source 0 --csi --weights runs/action/stgcn/best.pt
```

천장에 뒤집어 설치한 경우 `car/utils/video.py`의 `csi_gstreamer_pipeline()`에서
`flip_method`를 바꾸십시오 (0=없음, 2=180°, 1/3=90°).

### IP 카메라 (RTSP)

```bash
python3 tools/run_recognition.py --source "rtsp://user:pw@192.168.0.10:554/stream1"
```

지연이 크면 카메라 쪽 인코더 설정을 낮추고 sub-stream을 쓰십시오.

### 천장 카메라 설치 시 확인할 것

- 고객 전신이 프레임에 들어오는 높이인지. 상단 시점에서도 어깨·팔·엉덩이가
  보여야 합니다. 머리만 보이면 모델이 쓸 정보가 없습니다.
- 선반이 프레임 상단에 오도록 배치. 학습 데이터(MERL)와 구도가 비슷할수록 좋습니다.
- 역광과 천장 조명 반사 회피.
- **학습 데이터와 카메라 각도가 다르면 정확도가 떨어집니다.** 새 환경에서는
  README §2-2의 자체 데이터 학습을 권장합니다.

---

## 5. 속도 올리기

효과가 큰 순서입니다.

### 5.1 전력 모드 (§3) — 가장 먼저

### 5.2 입력 해상도 축소

```bash
python3 tools/run_recognition.py --source 0 --set pose.imgsz=480
```

640→480은 픽셀 수가 약 0.56배이므로 pose 단계가 그만큼 빨라집니다.
상단 시점에서 사람이 작게 나오면 검출률이 떨어지므로 320 아래로는 권장하지 않습니다.

### 5.3 행동 분류 주기 조정

```bash
python3 tools/run_recognition.py --source 0 --set runtime.action_every=3
```

30프레임 버퍼는 한 프레임마다 크게 바뀌지 않으므로 매 프레임 분류할 필요가 없습니다.
3프레임마다 분류해도 체감 지연은 거의 없습니다.

### 5.4 FP16

GPU가 활성화된 경우에만 의미가 있습니다.

```bash
python3 tools/run_recognition.py --source 0 --set pose.half=true
```

### 5.5 TensorRT 엔진

pose 모델이 파이프라인의 병목이므로 여기를 변환하는 것이 가장 효과적입니다.

```bash
# pose 모델 (첫 빌드는 수 분 걸립니다)
python3 tools/export_models.py --pose weights/yolo11n-pose.pt --format engine --half

# 행동 모델
python3 tools/export_models.py --checkpoint runs/action/stgcn/best.pt \
  --format tensorrt --precision fp16

# 변환된 pose 엔진으로 실행
python3 tools/run_recognition.py --source 0 \
  --pose-weights weights/yolo11n-pose.engine
```

엔진은 **빌드한 장치에서만 동작합니다.** 다른 Jetson으로 옮기면 다시 빌드해야 합니다.

### 5.6 모델 선택

LSTM이 ST-GCN보다 파라미터가 약 60만 개 적습니다. 논문에서는 정확도가 8%p 낮지만
fps는 더 높습니다. 어느 쪽이 맞는지는 용도에 달려 있습니다.

```bash
python3 tools/run_recognition.py --source 0 --weights runs/action/lstm/best.pt
```

### 5.7 추적기 경량화

```bash
--set tracker.appearance=none        # 색 히스토그램 계산 생략
--set tracker.enabled=false          # 추적 자체를 끔 (다중 인물 추적 불가)
```

### 5.8 측정

바꾸기 전후로 측정하십시오. 추측은 자주 틀립니다.

```bash
python3 tools/benchmark.py --source data/videos/your_clip.mp4 --frames 300 \
  --out runs/bench/before.json
```

---

## 6. 양자화: TFLite인가 TensorRT인가

논문은 float32 PyTorch 모델을 **int8 TFLite**로 변환합니다. 일반 엣지 보드에서는
타당한 선택이지만, Jetson에서는 TFLite가 GPU를 쓰지 않으므로 하드웨어를 놀립니다.

| 상황 | 권장 |
|---|---|
| Jetson (GPU 사용 가능) | **TensorRT FP16** |
| Jetson, 정확도 여유 있음 | TensorRT INT8 + 실제 데이터 캘리브레이션 |
| 논문 결과 재현 | int8 TFLite |
| GPU 없는 ARM 보드 | int8 TFLite 또는 PyTorch dynamic int8 |
| 툴체인 설치 불가 | PyTorch dynamic int8 (추가 설치 없음) |

TFLite 경로는 추가 패키지가 필요합니다.

```bash
pip install onnx2tf tensorflow onnx_graphsurgeon sng4onnx
python3 tools/export_models.py --checkpoint runs/action/lstm/best.pt \
  --format tflite --calibration data/processed/merl_skeletons.npz
```

`--calibration`에 실제 skeleton store를 주십시오. 무작위 값으로 캘리브레이션하면
int8 양자화가 정확도를 크게 잃습니다.

행동 모델은 35만~102만 파라미터로 작아서 양자화 이득이 제한적입니다.
pose 모델을 먼저 변환하십시오.

---

## 7. 서비스로 상시 실행

```ini
# /etc/systemd/system/customer-action.service
[Unit]
Description=Customer Action Recognition
After=network.target

[Service]
Type=simple
User=alooh
WorkingDirectory=/home/alooh/customer_action_recognition
ExecStart=/home/alooh/customer_action_recognition/.venv/bin/python \
  tools/run_recognition.py --source 0 --no-display \
  --weights runs/action/stgcn/best.pt \
  --save-csv /var/log/customer_actions.csv
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now customer-action
journalctl -u customer-action -f
```

CSV가 계속 커지므로 `logrotate`를 함께 설정하십시오.

---

## 8. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `torch.cuda.is_available()` False | §2 — Jetson 전용 휠 설치 |
| 카메라가 안 열림 | `ls /dev/video*`, CSI면 `--csi`, 권한은 `video` 그룹 |
| 사람이 검출되지 않음 | `--set pose.conf=0.2`, 카메라가 너무 높지 않은지 |
| 라벨이 안 뜸 | 30프레임 버퍼가 채워졌는지 (화면에 `buffering N/30` 표시) |
| 라벨이 계속 바뀜 | `--set runtime.smooth_window=9 runtime.min_action_conf=0.5` |
| 트랙 ID가 자꾸 바뀜 | `--set tracker.max_age=45 tracker.max_iou_distance=0.8` |
| 메모리 부족 | `--set pose.imgsz=480 train.batch_size=32`, 스왑 확인 |
| 느림 | §5 순서대로, 그리고 `tools/benchmark.py`로 측정 |
| `imshow` 오류 (SSH) | `--no-display` 사용 |
