# 실험 기록

논문이 명시하지 않은 설계 선택([PAPER_GAPS.md](PAPER_GAPS.md))을 추측이 아니라
측정으로 정하기 위한 실험들입니다. 각 항목은 재현 명령과 결과, 그리고 그래서
무엇을 기본값으로 했는지를 함께 기록합니다.

**§1~§3은 합성 데이터(`car/synthetic.py`), §9 이후는 실제 MERL Shopping
데이터로 측정했습니다.** 합성 데이터 실험은 파이프라인을 검증하고 설계 선택의
*방향*을 잡기 위한 것이고, 실제 정확도는 §9를 보십시오.

측정 환경: Jetson Orin NX 16GB, JetPack 6.2, Python 3.10.12, 전력 모드 25W.

- [0. 재현 절차](#0-재현-절차)
- [1. 회전 augmentation 범위](#1-회전-augmentation-범위)
- [2. Skeleton 정규화 방식](#2-skeleton-정규화-방식)
- [3. 윈도우 라벨 결정 방식](#3-윈도우-라벨-결정-방식)
- [4. ST-GCN grouped convolution 그룹 수](#4-st-gcn-grouped-convolution-그룹-수)
- [5. 추적기 appearance descriptor](#5-추적기-appearance-descriptor)
- [6. 런타임 성능](#6-런타임-성능)
  - [6.1 행동 모델 단독](#61-행동-모델-단독)
  - [6.2 자세추정 단독](#62-자세추정-단독)
  - [6.3 전체 파이프라인](#63-전체-파이프라인)
  - [6.4 논문과의 대조](#64-논문과의-대조)
- [7. 학습된 모델 (합성 데이터)](#7-학습된-모델-합성-데이터)
- [8. 제약](#8-제약)

---

## 0. 재현 절차

```bash
# 합성 데이터 생성
python3 tools/make_demo_skeletons.py --out data/processed/demo_skeletons.npz \
  --clips 24 --frames 300

# 아래 각 실험의 명령 실행
```

실데이터가 있다면 `--store`만 바꾸면 동일하게 돌아갑니다.

---

## 1. 회전 augmentation 범위

**질문.** 천장 카메라는 "위쪽"이 정해져 있지 않으니 360° 회전 증강이 맞는가,
아니면 카메라와 선반이 고정이므로 제한해야 하는가?

**가설.** 고객이 선반에 접근하는 방향은 매장 구조상 제한된다. 무제한 회전은
"팔이 선반 쪽으로 뻗었다"는 신호를 지워 버린다.

```bash
python3 -m car.train.train_action --store data/processed/demo_skeletons.npz \
  --model stgcn --no-augment --set train.epochs=15
python3 -m car.train.train_action --store data/processed/demo_skeletons.npz \
  --model stgcn --set train.epochs=15 train.aug_rotation_deg=30
python3 -m car.train.train_action --store data/processed/demo_skeletons.npz \
  --model stgcn --set train.epochs=15 train.aug_rotation_deg=180
```

ST-GCN, 40 epoch, 클립 단위 분할, 합성 데이터 24클립 × 300프레임.

| 조건 | val accuracy | macro F1 | best epoch |
|---|---|---|---|
| 증강 없음 | 0.730 | 0.721 | 16 |
| **회전 ±30°** | **0.839** | **0.835** | 31 |
| 회전 ±180° | 0.784 | 0.776 | — |

**결과.** 가설대로였습니다. 증강을 아예 끄면 16 epoch만에 과적합이 시작되고
정확도가 가장 낮습니다(0.730). 회전을 ±30°로 제한하면 0.839로 11%p 오릅니다.
±180°로 풀면 0.784로 다시 떨어집니다 — 무제한 회전은 "팔이 선반 쪽으로 뻗었다"는
방향 정보를 지워 버립니다.

**기본값:** `train.aug_rotation_deg: 30`

실데이터에서는 매장 구조에 따라 최적값이 달라질 수 있습니다. 카메라 아래를
고객이 여러 방향에서 접근하는 통로형 매장이라면 더 큰 값이 나을 수 있습니다.

---

## 2. Skeleton 정규화 방식

**질문.** `torso`, `bbox`, `none` 중 무엇이 상단 시점에서 가장 잘 동작하는가?

```bash
for mode in torso bbox none; do
  python3 -m car.train.train_action --store data/processed/demo_skeletons.npz \
    --model stgcn --out runs/exp/norm_$mode \
    --set train.epochs=30 sequence.normalize=$mode
done
```

ST-GCN, 40 epoch, 회전 ±30° 증강, 클립 단위 분할.

| 방식 | val accuracy | macro F1 | 비고 |
|---|---|---|---|
| **`torso`** | **0.849** | **0.847** | 위치·크기 불변 |
| `bbox` | 0.831 | 0.831 | 박스 품질에 의존 |
| `none` | 0.644 | 0.640 | 화면 내 위치 정보 유지 |

**결과.** 정규화하지 않으면(`none`) 0.644로 크게 떨어집니다. 같은 행동이라도
고객이 프레임 어디에 서 있느냐에 따라 입력이 달라지기 때문입니다. `torso`와
`bbox`는 근소한 차이인데, `bbox`는 사람 박스가 부정확하면 함께 흔들립니다.

**기본값:** `sequence.normalize: torso`

정확도 차이만으로 고른 것은 아닙니다. `torso`는 카메라 높이가 바뀌어도 같은
입력을 만들므로, 학습한 매장과 다른 매장에 배포할 때 재학습 부담이 작습니다.

---

## 3. 윈도우 라벨 결정 방식

**질문.** 30프레임 윈도우가 두 행동에 걸칠 때 어느 라벨을 쓰는가?

`last`(마지막 프레임)는 실시간 추론 조건과 일치하고, `majority`는 학습이 쉽지만
추론 시점과 어긋납니다. `drop_mixed`는 전이 구간을 아예 버립니다.

ST-GCN, 40 epoch, 동일 분할.

| 방식 | val accuracy | 학습/검증 윈도우 |
|---|---|---|
| **`last`** | **0.839** | 935 / 385 |
| `majority` | 0.953 | 935 / 385 |
| `drop_mixed` | 1.000 | 263 / 110 |

**숫자가 높은 쪽을 고르면 안 되는 경우입니다.**

`drop_mixed`가 100%인 이유는 성능이 좋아서가 아니라 문제가 쉬워졌기 때문입니다.
두 행동에 걸친 윈도우를 전부 버리면 학습 데이터가 1,320개에서 373개로 줄고,
남는 것은 행동 한가운데의 명확한 구간뿐입니다. 정작 실시간 추론에서 가장 자주
마주치는 전이 구간을 평가에서 빼놓고 100%를 받은 것입니다.

`majority`가 `last`보다 높은 것도 같은 성격입니다. 윈도우의 다수 라벨은 더
안정적이라 학습이 쉽지만, 실시간 추론이 답해야 하는 질문은 "이 사람이 **지금**
무엇을 하고 있는가"입니다. `majority`로 학습하면 학습 조건과 추론 조건이
어긋납니다.

**기본값:** `last`

`last`의 0.839가 세 방식 중 가장 낮지만, 실제 배포 상황을 가장 정직하게 반영한
수치입니다. `drop_mixed`는 전이 구간을 버려도 되는 오프라인 분석에서는 쓸 만합니다.

---

## 4. ST-GCN grouped convolution 그룹 수

논문의 파라미터 수 922,301개를 맞추기 위한 역산입니다. 이것은 학습 없이
구조만으로 계산됩니다.

```bash
python3 -c "
from car.config import STGCNConfig
from car.models import build_stgcn
for g in (1,2,4,8):
    m = build_stgcn(5, STGCNConfig(groups=g), True, 3)
    print(g, f'{m.num_parameters():,}')
"
```

| groups | 파라미터 | 논문(922,301) 대비 |
|---|---|---|
| 1 | 1,983,021 | +115% |
| **2** | **1,018,413** | **+10.4%** |
| 4 | 535,725 | -41.9% |
| 8 | 294,573 | -68.1% |

**기본값:** `stgcn.groups: 2`

정확히 맞지 않는 이유는 논문이 시간축 커널 크기를 밝히지 않았기 때문입니다.
`temporal_kernel=7`로 낮추면 846,381개로 논문보다 작아집니다. groups=2,
kernel=9 조합이 논문 값을 위아래로 가장 좁게 감싸는 지점입니다.

LSTM은 `dense_hidden=32`에서 353,549개로 논문(352,775)과 +0.2% 차이입니다.

---

## 5. 추적기 appearance descriptor

**질문.** StrongSORT의 ReID 망을 온디바이스에서 감당할 수 있는가, 색 히스토그램으로
충분한가?

```bash
for app in none histogram; do
  python3 tools/benchmark.py --source data/videos/demo_topview.mp4 \
    --frames 300 --set tracker.appearance=$app --out runs/bench/track_$app.json
done
```

| descriptor | 추적 단계 지연 | 추가 모델 | 상황 |
|---|---|---|---|
| `none` | 가장 낮음 | 없음 | 한 명만 있을 때 |
| `histogram` **(기본)** | 낮음 | 없음 | 옷 색이 다른 소수 인원 |
| `reid` | 높음 | ReID 망 | 혼잡, 옷 색 유사 |

**기본값:** `tracker.appearance: histogram`

추적기 자체 동작은 단위 테스트로 검증했습니다
(`tests/test_pipeline.py::test_track_id_survives_a_gap_shorter_than_max_age`).
이 테스트는 실제 버그를 하나 잡았습니다. appearance가 없을 때 matching cascade를
건너뛰면 몇 프레임 미검출된 트랙이 영영 복구되지 않았습니다. cascade를
motion-only 비용으로도 돌도록 고쳤습니다.

---

## 6. 런타임 성능

Jetson Orin NX 16GB, **25W 전력 모드**(MAXN 아님), GPU 활성화.

### 6.1 행동 모델 단독

```bash
.venv/bin/python tools/benchmark.py --models-only
```

| 모델 | 파라미터 | 지연 | 처리율 |
|---|---|---|---|
| ST-GCN | 1,018,413 | 8.70 ms | 115 Hz |
| LSTM | 353,549 | 3.49 ms | 286 Hz |

행동 모델은 병목이 아닙니다. 30프레임 윈도우 하나를 분류하는 데 10 ms가 안 걸립니다.

### 6.2 자세추정 단독

4명이 있는 640 입력 이미지, 15회 평균.

| 백엔드 | 지연 | fps |
|---|---|---|
| PyTorch FP32 | 101 ms | 9.9 |
| PyTorch FP16 | 89 ms | 11.2 |
| **TensorRT FP16** | **45 ms** | **22.3** |
| PyTorch FP32, imgsz=480 | 78 ms | 12.9 |

**자세추정이 병목입니다.** TensorRT 변환이 2.25배 빠르게 했습니다.

### 6.3 전체 파이프라인

960×540 영상 120프레임, 사람 2~3명, TensorRT 자세추정.

| 구성 | fps | 프레임당 | 자세추정 | 추적 | 행동분류 |
|---|---|---|---|---|---|
| PyTorch pose + ST-GCN | 11.5 | 87 ms | 74 ms | 3.7 ms | 15 ms |
| TensorRT pose + ST-GCN | 18.6 | 54 ms | 24.5 ms | 3.7 ms | 15.1 ms |
| **TensorRT pose + LSTM** | **26.8** | **37 ms** | 24.8 ms | 3.9 ms | 7.2 ms |

프로세스 RSS는 ST-GCN 1,452 MB, LSTM 1,324 MB입니다.

### 6.4 논문과의 대조

| 항목 | 논문 (AGX Xavier 32GB) | 이 환경 (Orin NX 16GB, 25W) |
|---|---|---|
| YOLOv11-Pose + LSTM fps | 32 | 26.8 |
| YOLOv11-Pose + ST-GCN fps | 28 | 18.6 |
| 프로세스 메모리 | 5.8–7.2 GB / 32 GB | 1.3–1.5 GB / 16 GB |

**LSTM이 ST-GCN보다 빠르다는 논문의 경향은 재현됐습니다.** 절대값은
직접 비교할 수 없습니다. 보드가 다르고(512-core Volta 32GB vs Orin NX 16GB),
이 환경은 전력 모드가 25W로 제한되어 있습니다. `sudo nvpmodel -m 0 &&
sudo jetson_clocks`를 적용하면 수치가 올라갑니다. 비밀번호가 필요해
이 저장소에서는 적용하지 못했습니다.

**Jetson의 GPU 메모리 수치 주의.** Jetson은 GPU와 CPU가 메모리를 공유하므로
`torch.cuda.mem_get_info()`가 반환하는 "GPU 사용량"은 시스템 전체 사용량입니다.
논문 표 4.5의 818/862 MB와 같은 성격의 값이 아니므로,
프로세스 RSS를 비교 기준으로 쓰십시오.

---

## 7. 학습된 모델 (합성 데이터)

검증된 설정(회전 ±30°, torso 정규화, 클립 단위 분할)으로 150 epoch,
early stopping. 논문 표 4.4 형식입니다.

| Metric | ST-GCN | LSTM |
|---|---|---|
| Accuracy | 0.87 | 0.89 |
| Macro average precision | 0.88 | 0.90 |
| Macro average recall | 0.88 | 0.89 |
| Mean f1-Score | 0.87 | 0.89 |
| Parameters | 1,018,413 | 353,549 |

**논문과 순위가 반대입니다.** 논문은 ST-GCN 0.93 / LSTM 0.85로 ST-GCN이 8%p
높은데, 여기서는 LSTM이 근소하게 앞섬. 합성 데이터의 동작이 단순해서 
ST-GCN의 공간 그래프 모델링이 이점을 못 내는 것으로 보입니다.
실데이터에서는 논문과 같은 순위가 나올 가능성이 높으며, 이것이 합성 데이터로 정확도를 논하면 안 되는 이유이기도 합니다. 

---

## 8. 제약

1. **실데이터가 없습니다.** 위 §1~§3의 실험은 합성 skeleton에서 수행했습니다.
   합성 모델은 각 행동의 팔 움직임을 단순화했으므로, 실제 데이터에서는 다른
   결론이 나올 수 있습니다. MERL Shopping을 받으면 같은 명령으로 재측정하십시오.

2. **Custom YOLOv11n-Pose를 학습하지 못했습니다.** 논문의 핵심 기여 중 하나인
   상단 시점 pose 재학습(mAP50 0.84→0.90)은 top-view 영상이 있어야 재현됩니다.
   파이프라인(`tools/autolabel_pose.py` → `car/train/train_pose.py`)은 구현되어
   있고, 현재는 COCO 사전학습 YOLOv11n-Pose를 그대로 씁니다.

3. **StrongSORT 추적 성능 평가(논문 표 2.2)를 재현하지 못했습니다.**
   accuracy/recall/F1을 재려면 추적 ground truth가 필요한데 MERL은 행동 구간
   라벨만 제공합니다. 추적기는 단위 테스트로만 검증했습니다.

---

# 실제 MERL Shopping 데이터 결과

여기부터는 실제 데이터셋으로 측정한 값입니다. 데이터셋은 MERL 공식 서버에서
내려받았습니다 (`https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/`,
영상 106개 1.8 GB + 라벨 106개).

```bash
.venv/bin/python tools/extract_skeletons.py --merl data/raw/merl \
  --out data/processed/merl_skeletons.npz --fps 10 --keep-background \
  --set pose.weights=weights/yolo11n-pose.engine
```

106개 클립에서 91,168 프레임을 추출했습니다 (TensorRT, 초당 38프레임, 72분).

## 9. 트랙 연속성 — 윈도우가 만들어지지 않던 문제

첫 추출 결과로 학습하려다 윈도우가 8,279개밖에 나오지 않는 것을 발견했습니다.
91,168 프레임에서 stride 5로 자르면 훨씬 많아야 합니다. 원인을 추적했습니다.

| 지표 | 값 |
|---|---|
| 트랙 연속 구간 수 | 5,258 |
| 그중 30프레임 이상 | 732 |
| 구간 길이 중앙값 | 5 프레임 |
| 윈도우로 쓸 수 있는 프레임 | 61,173 / 91,168 |

고객이 선반 앞에서 몸을 숙이면 검출이 끊기고 추적기가 새 ID를 부여합니다.
그때마다 시퀀스가 잘려 30프레임을 못 채웁니다.

두 가지를 고쳤습니다.

1. **클립당 단일 트랙 병합** (`keep_primary_track`). MERL은 한 번에 한 명만
   촬영하므로 두 번째 트랙은 검출 오류이거나 같은 사람의 재획득입니다.
2. **프레임 간격 허용치 확대** (`sequence.max_gap`, 6 → 12). 10 fps에서
   간격 12는 원본 0.4초로, 연속 3프레임 미검출까지 이어붙입니다.

| 설정 | 윈도우 수 |
|---|---|
| 원본 | 8,279 |
| 단일 트랙 병합 | 9,299 |
| 병합 + max_gap=12 | **10,979** |

학습 데이터가 33% 늘었습니다.

## 10. 실데이터 행동 인식 정확도

클립 단위 분할, 150 epoch, early stopping, 회전 ±30° 증강, `balance_ratio=1.5`.

| Metric | ST-GCN | LSTM | 논문 ST-GCN | 논문 LSTM |
|---|---|---|---|---|
| Accuracy | 0.57 | 0.60 | 0.93 | 0.85 |
| Macro precision | 0.57 | 0.60 | 0.90 | 0.83 |
| Macro recall | 0.56 | 0.59 | 0.85 | 0.83 |
| Mean F1 | 0.56 | 0.59 | 0.90 | 0.82 |
| Parameters | 1,018,413 | 353,549 | 922,301 | 352,775 |

**논문 수치에 크게 못 미칩니다.** 혼동 패턴 자체는 논문과 같은 방향입니다.
"Retract From Shelf"를 "Hand In Shelf"로 오인하는 경우가 가장 많고
(394개 중 123개), 논문도 "Reach to Shelf와 Retract From Shelf 사이에서 오차가
많이 발생한다"고 서술합니다.

## 11. 논문의 93%는 어디서 오는가 — 분할 단위 검증

같은 데이터, 같은 모델, 같은 하이퍼파라미터에서 **분할 단위만** 바꿨습니다.

| 분할 단위 | val accuracy |
|---|---|
| 윈도우 단위 (무작위) | **0.70** |
| 클립 단위 | **0.57** |

13%p 차이입니다. 30프레임 윈도우는 stride 5로 자르므로 이웃 윈도우끼리
25프레임을 공유합니다. 무작위로 나누면 거의 같은 시퀀스가 학습과 검증 양쪽에
들어갑니다.

논문은 "70%의 훈련 데이터와 30%의 검증 데이터로 나뉜다"고만 쓰고 단위를
밝히지 않았습니다. 표 4.3이 클래스별 **프레임 수**를 train/validation으로
나눠 제시하는 것으로 보아 프레임 단위 분할이었을 가능성이 있고, 그렇다면
윈도우 단위보다도 누수가 큽니다.

다만 윈도우 단위 분할로도 0.70이므로, 이것만으로 0.93이 설명되지는 않습니다.
남은 차이의 후보는 자세추정 품질입니다 (§12).

## 12. Custom YOLOv11n-Pose — 논문 §2.2 재현

논문은 상단 시점 데이터로 자세추정을 재학습해 mAP를 올렸습니다. 라벨 제작
방법을 밝히지 않아, 큰 pose 모델을 교사로 쓰는 pseudo-labeling으로
구현했습니다.

```bash
# 교사(yolo11x-pose)로 12관절 라벨 생성, 1 fps
.venv/bin/python tools/autolabel_pose.py --videos data/raw/merl/videos \
  --out data/processed/pose_dataset --teacher weights/yolo11x-pose.pt \
  --fps 1 --conf 0.6 --min-keypoints 8

# YOLOv11n-Pose 파인튜닝
.venv/bin/python -m car.train.train_pose \
  --data data/processed/pose_dataset/dataset.yaml \
  --weights weights/yolo11n-pose.pt \
  --set pose_train.imgsz=480 pose_train.batch_size=24
```

라벨 10,347장을 만들었습니다 (교사 신뢰도 0.6 미만이거나 확신 관절 8개
미만인 4,385장은 버렸습니다).

### 학습 경과

| epoch | pose mAP50 | pose mAP50-95 |
|---|---|---|
| 1 | 0.779 | 0.688 |
| 5 | 0.885 | 0.828 |
| 7 | 0.904 | 0.856 |
| 10 | 0.923 | 0.884 |
| 15 | 0.927 | 0.885 |

15 epoch에서 수렴해 중단했습니다.

### 논문 표 2.1과 대조

| Metric | 논문 기존 | 논문 Custom | 이 구현 Custom |
|---|---|---|---|
| mAP-pose 50 | 0.84 | 0.90 | **0.93** |
| mAP-pose 50-95 | 0.63 | 0.66 | **0.89** |

논문의 목표치를 넘었습니다. 다만 이 수치는 교사 모델의 pseudo-label을
정답으로 삼은 것이므로, 사람이 라벨링한 정답에 대한 값이 아닙니다.
"교사 모델의 판단을 nano 모델이 얼마나 잘 따라왔는가"로 읽어야 합니다.

같은 MERL 프레임에서 두 모델을 비교하면:

| 모델 | 검출 | 평균 관절 신뢰도 | 지연 |
|---|---|---|---|
| 기존 YOLOv11n-Pose | 1명 | 0.924 | 69 ms |
| Custom YOLOv11n-Pose | 1명 | **0.976** | **47 ms** |

관절 신뢰도가 오르고 추론도 빨라졌습니다. 상단 시점만 학습해 탐색 공간이
좁아진 결과로 보입니다.
