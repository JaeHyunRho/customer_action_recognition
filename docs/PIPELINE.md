# 파이프라인 상세

논문 그림 2.1의 구조도를 코드 어디에서 구현했는지, 그리고 각 단계가 무엇을
입출력하는지 정리합니다.

- [단계별 설명](#단계별-설명)
  - [1. 자세추정 — `car/pose/estimator.py`](#1-자세추정--carposeestimatorpy)
  - [2. 머리 키포인트 제거 — `car/keypoints.py`](#2-머리-키포인트-제거--carkeypointspy)
  - [3. 추적 — `car/pose/tracker.py`](#3-추적--carposetrackerpy)
  - [4. 버퍼링 — `car/pipeline/buffer.py`](#4-버퍼링--carpipelinebufferpy)
  - [5. 정규화 — `car/pipeline/normalize.py`](#5-정규화--carpipelinenormalizepy)
  - [6. 행동 분류 — `car/models/`](#6-행동-분류--carmodels)
  - [7. 평활화 — `car/pipeline/buffer.py :: smoothed`](#7-평활화--carpipelinebufferpy--smoothed)
- [데이터 흐름 (학습)](#데이터-흐름-학습)
- [확장 지점](#확장-지점)

```
영상/카메라 프레임 (BGR, H×W×3)
      │
      ▼  car/pose/estimator.py :: PoseEstimator.infer
YOLOv11n-Pose
      │   boxes (N,4) xyxy · scores (N,) · keypoints (N,17,3)
      ▼  car/keypoints.py :: drop_head
머리 5개 키포인트 제거
      │   keypoints (N,12,3)
      ▼  car/pose/tracker.py :: StrongSort.update
StrongSORT (Kalman → matching cascade → IoU matching)
      │   TrackOutput(track_id, box, score, keypoints) 목록
      ▼  car/pipeline/buffer.py :: BufferBank.update
트랙별 30프레임 링 버퍼
      │   버퍼가 찬 트랙만 통과
      ▼  car/pipeline/normalize.py :: normalize_sequence
몸통 기준 정규화 + 저신뢰 관절 0 처리
      │   (T=30, V=12, C=3) → (C, T, V)
      ▼  car/models/{stgcn,lstm}.py
ST-GCN 또는 LSTM
      │   logits (B, 5) → softmax
      ▼  car/pipeline/buffer.py :: BufferBank.smoothed
최근 5개 예측 다수결
      │
      ▼
행동 라벨 + 신뢰도
```

---

## 단계별 설명

### 1. 자세추정 — `car/pose/estimator.py`

YOLOv11n-Pose를 ultralytics로 감쌉니다. 두 종류의 가중치를 투명하게 처리합니다.

- COCO 사전학습 가중치: 17개 관절을 예측 → 추론 후 머리 5개를 제거
- 커스텀 상단시점 가중치: 처음부터 12개 관절

`resolve_device()`가 CUDA 사용 가능 여부를 확인하고 불가하면 CPU로 내려갑니다.
Jetson에서 torch 빌드가 드라이버와 맞지 않으면 여기서 조용히 CPU로 떨어지므로,
`describe()`가 출력하는 `device=`를 확인하십시오.

### 2. 머리 키포인트 제거 — `car/keypoints.py`

논문 §2.1과 그림 2.4. 코, 양 눈, 양 귀(COCO 인덱스 0–4)를 버리고 12개 관절만
남깁니다. 상단 시점에서는 머리가 어깨·몸통을 가려 오히려 방해가 됩니다.

논문의 "어깨 및 팔 부분 skeleton에 가중치 0.5를 추가"는 두 곳에서 구현됩니다.

- 자세추정 학습: `oks_sigmas()`가 해당 관절의 OKS 시그마를 `1/(1+0.5)`배로
  줄입니다. 시그마가 작을수록 그 관절의 오차에 손실이 더 민감해집니다.
- 특징 가중: `keypoint_weights()`가 좌표에 곱할 1.5배 가중치를 제공합니다.

### 3. 추적 — `car/pose/tracker.py`

StrongSORT를 외부 패키지 없이 직접 구현했습니다. boxmot이나 torchreid를 쓰면
torch 사본이 하나 더 들어와 16 GB Jetson에서는 손해입니다.

논문 그림 2.7의 구성 요소가 모두 있습니다.

| 구성 | 구현 |
|---|---|
| Kalman filter predict/update | `KalmanBoxFilter`, NSA 가중 측정 잡음 |
| Matching cascade | `_matching_cascade`, 트랙 나이별 우선순위 |
| IoU matching | `_iou_match`, 최근 트랙만 대상 |
| Tentative → Confirmed | `n_init=3` 연속 매칭 |
| Deleted | `max_age` 프레임 미갱신 |
| EMA feature bank | `Track.update`, `ema_alpha=0.9` |

appearance descriptor는 세 가지 중 고를 수 있습니다: `histogram`(기본, HSV 색
히스토그램), `reid`(외부 ReID 망), `none`(모션만).

**주의할 지점.** appearance가 없어도 matching cascade는 돌아야 합니다.
cascade를 건너뛰면 몇 프레임 미검출된 트랙이 IoU 단계에도 못 들어가(최근
트랙만 대상) 영영 복구되지 않습니다. 이 버그는 단위 테스트
`test_track_id_survives_a_gap_shorter_than_max_age`가 잡았고, cascade가
motion-only 비용으로도 동작하도록 고쳤습니다.

### 4. 버퍼링 — `car/pipeline/buffer.py`

논문의 "30 프레임의 시간적 정보를 버퍼링"입니다. 추적기가 있으므로 버퍼는
**트랙별**입니다. 프레임 단위 버퍼 하나를 쓰면 두 고객의 자세가 한 시퀀스에
섞입니다.

`batch_inputs()`가 버퍼가 찬 모든 트랙을 하나의 배치로 모아 forward pass를
한 번만 돌립니다. 매장에 세 명이 있어도 분류는 한 번입니다.

버퍼가 아직 30프레임이 안 찼거나 유효 관절 비율이 30% 미만이면 `None`을
반환해 분류를 건너뜁니다. 화면에는 `buffering N/30`으로 표시됩니다.

### 5. 정규화 — `car/pipeline/normalize.py`

논문에 없는 부분입니다. 세 가지 방식의 비교와 선택 근거는
[PAPER_GAPS.md §1.1](PAPER_GAPS.md)에 있습니다.

기본값 `torso`는 엉덩이 중점으로 평행이동하고, 몸통 길이와 어깨 너비 중 큰 값의
클립 중앙값으로 나눕니다. 프레임마다 다시 스케일하지 않는 이유는, 그렇게 하면
팔을 뻗을 때 몸 전체가 수축하는 것처럼 보여 동작 정보가 사라지기 때문입니다.

### 6. 행동 분류 — `car/models/`

두 모델 모두 `(N, C, T, V)` 입력과 `(N, 5)` 출력을 갖습니다.
런타임 코드가 어느 쪽인지 신경 쓰지 않아도 되도록 인터페이스를 맞췄습니다.

**ST-GCN** (논문 그림 3.1): grouped convolution 기반 GCN + TCN 6층,
채널 64/64/128/128/256/256, 각 블록은
`GroupedConv → BN → LeakyReLU → GroupedConv → BN → Dropout → LeakyReLU`.
공간 그래프는 `car/models/graph.py`가 spatial 3분할로 만듭니다.

**LSTM** (논문 그림 3.2): 3층 × 128채널 + `Dense → LeakyReLU → Dropout → Dense`.
입력은 각 시점의 평탄화된 스켈레톤 36차원(12관절 × 3채널)입니다.

### 7. 평활화 — `car/pipeline/buffer.py :: smoothed`

논문에 없습니다. 프레임마다 argmax를 그대로 쓰면 "Reach To Shelf"와
"Hand In Shelf"처럼 시각적으로 비슷한 행동 사이에서 라벨이 떨립니다.
최근 5개 예측의 다수결을 취하고, 신뢰도는 승자 예측들의 평균으로 계산합니다.

---

## 데이터 흐름 (학습)

```
MERL 영상 + .mat 라벨          자체 촬영 영상 (행동별 폴더)
        │                              │
        └──────────┬───────────────────┘
                   ▼  tools/extract_skeletons.py
         YOLOv11n-Pose + StrongSORT
                   │
                   ▼  car/datasets/sequence.py :: SkeletonStore
         skeletons.npz  (프레임 단위 평탄 테이블)
                   │
                   ▼  build_windows(window=30, stride=5)
         (clip, track) 연속 구간을 30프레임 윈도우로 절단
                   │
                   ▼  car/train/train_action.py
         클립 단위 train/val 분할 → 학습 → best.pt
```

`SkeletonStore`가 프레임 단위 평탄 테이블인 이유는, 윈도우 길이나 stride를
바꿀 때 영상에서 skeleton을 다시 뽑지 않아도 되게 하기 위해서입니다.
윈도우는 `__getitem__` 시점에 잘립니다.

윈도우는 같은 `(clip, track)` 안에서만 만들어지고, 프레임 번호 간격이 6을
넘으면 구간을 끊습니다. 10 fps 샘플링에서 한두 프레임 미검출은 허용하되,
긴 공백을 가로지르는 시퀀스는 만들지 않기 위해서입니다.

---

## 확장 지점

| 하고 싶은 것 | 손댈 곳 |
|---|---|
| 행동 클래스 추가 | `car/config.py :: ACTION_CLASSES` |
| 다른 pose 모델 | `car/pose/estimator.py`, `PoseResult`만 맞추면 됩니다 |
| 다른 추적기 | `car/pose/tracker.py`, `update()` 시그니처 유지 |
| 새 행동 모델 | `car/models/`에 추가 후 `build_model()`에 등록 |
| 새 정규화 방식 | `car/pipeline/normalize.py :: normalize_sequence` |
| 행동 발생 시 알림 | `car/pipeline/realtime.py :: run`의 예측 루프 |
| 다른 카메라 종류 | `car/utils/video.py :: FrameSource` |
