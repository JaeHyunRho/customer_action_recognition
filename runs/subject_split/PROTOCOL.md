# Subject-split experiment protocol (E1, E2)

Received from the user on 2026-10-07 and copied verbatim below. Every run under
`runs/subject_split/` follows it. Choices the text leaves open are listed at the
end under "Implementation choices".

---

Jetson의 저장소 루트에서 .venv/bin/python으로 실행합니다. 데이터는 data/raw/merl입니다. 조건은 논문의 Xavier, TFLite, 70/30, mAP, 93%와 85%가 아니라 지금 구현을 따릅니다. 이 실험의 정확도만 논문에 씁니다. 기존 수치 0.949, 0.701, 0.694, +12.7%p, 윈도우 25,627, mAP 0.927과, 기존 custom_yolo11n_pose.pt로 뽑은 skeleton은 사용하지 않습니다.

먼저 고칠 코드
autolabel_pose.py는 폴더의 영상을 전부 라벨링하고, 프레임을 20% 확률로 검증에 넣습니다. 입력은 피험자 1–20만 받게 하고, 검증은 아래의 영상 단위 규칙으로 나누게 고칩니다.

train_action.py의 --split-by clip과 window는 쓰지 않습니다. 학습 파일로만 가중치를 갱신하고, 조기 종료는 검증 파일의 accuracy만 보며, 테스트 파일은 각 seed의 best checkpoint로 한 번만 평가하게 고칩니다.

normalize.py의 torso 척도 주석은 “clip”으로 되어 있으나, 실제 중앙값은 넘겨받은 30장입니다. 주석을 윈도우 단위로 고칩니다. 계산은 바꾸지 않습니다.

1. Custom pose 재학습
파일명은 {피험자}_{회차}입니다. 기존 pose 라벨에 피험자 21 이상이 있으면 다시 학습합니다.

학습과 pose 검증은 피험자 1–20, 60편만 사용합니다. 21–26과 27–41은 넣지 않습니다. 60편을 이름순으로 정렬한 뒤 seed 42로 영상을 섞어 20%를 검증으로 뺍니다. 한 영상의 프레임은 한쪽에만 들어갑니다. 같은 피험자의 다른 회차가 pose 학습에 남을 수 있습니다. 이 분할은 pose checkpoint 선택에만 쓰므로, 행동 인식 테스트 누수로는 보지 않습니다.

라벨 조건은 유지합니다. 교사는 yolo11x-pose, 검출 신뢰도 0.6 이상, 확신 관절 8개 이상, 프레임당 1명입니다. 머리 5개는 라벨을 저장하기 전에 제거해 12관절로 학습합니다. 어깨·팔 OKS σ는 기존대로 1.5로 나눕니다. mAP를 남기면 teacher agreement로 적고, 평가 σ가 표준 COCO 표가 아니라고 적습니다.

2. Skeleton 추출
새 Custom 가중치로 뽑습니다. extract_skeletons.py --split train|val|test를 사용합니다. 학습은 피험자 1–20(60편), 검증은 21–26(18편), 테스트는 27–41(28편)이고, 파일은 셋으로 나눕니다. 샘플링은 10 fps입니다. 프레임 인덱스는 iter_sampled_frames와 같이 step = round(원본fps / 10), idx % step == 0입니다.

COCO nano(weights/yolo11n-pose.pt)와 교사 x(weights/yolo11x-pose.pt)도 같은 분할로 뽑습니다. 둘은 17관절입니다. 분류기에 넣기 전에 drop_head로 12관절이 됐는지 로그로 확인합니다. Custom과 관절 수가 다르면 E2는 무효입니다.

추출 직후 keep_primary_track을 적용합니다. 프레임마다 박스 면적이 가장 큰 검출만 남기고, 남은 행의 track id는 모두 1로 바꿉니다. 이 입력을 쓰면 StrongSORT의 ID는 평가에 들어가지 않습니다. MERL은 한 사람 촬영이므로 이 규칙을 유지하고, 추적 on/off는 하지 않습니다.

3. 윈도우와 채점
모델 학습용 윈도우는 build_windows 그대로입니다. 같은 클립에서 원본 프레임 번호 차이가 1 이상 12 이하인 검출만 한 구간으로 잇습니다. 빠진 프레임은 보간하거나 0으로 채우지 않습니다. 그보다 크게 끊기면 구간을 나누고, 검출이 30개 미만이면 윈도우를 만들지 않습니다. 한 윈도우는 저장된 검출 30개입니다. 라벨은 마지막 검출, stride는 학습·검증·테스트 모두 5입니다.

결측이 없으면 이 30개는 약 3초입니다. 검출과 검출 사이마다 원본 프레임 간격이 최대 12까지 허용되므로, 그 간격이 29번이면 최장 약 11.6초입니다. 원본 프레임 간격이 정상 간격 3보다 큰 윈도우의 비율과, 윈도우가 실제로 덮는 시간을 보고합니다.

주 정확도는 윈도우마다 argmax만 사용합니다. 최근 5개 다수결과 신뢰도 0.4은 화면 출력입니다. 주 수치에 넣지 않습니다. 그 후처리를 보조로 보고할 때만, 0.4 미만으로 표시하지 않은 윈도우는 오답입니다.

train.balance_ratio=1.5는 학습에만 적용합니다. 설정 기본값은 0이므로 반드시 지정합니다. 검증과 테스트는 원래 클래스 분포입니다. 증강은 회전 ±30°, 스케일 ±15°, 좌우 반전입니다. 반전 시 왼쪽·오른쪽 관절을 서로 바꾸는 기존 코드를 유지합니다. --no-augment는 쓰지 않습니다.

정규화는 윈도우 30장의 torso 중앙값입니다. 클립 전체가 아니고, 결정 이후 프레임은 쓰이지 않습니다. 실시간 버퍼도 쌓인 30장에 같은 함수를 호출합니다. 실시간 버퍼는 max_gap으로 구간을 자르지 않습니다.

epoch와 조기 종료는 검증 파일만 보고 정합니다. 테스트는 학습이 끝난 뒤 한 번만 평가합니다. seed는 42, 43, 44입니다. 테스트 점수가 높은 seed를 고르지 않습니다.

지표는 accuracy와 macro-F1이고, 혼동행렬을 저장합니다. 피험자별 accuracy를 seed마다 저장합니다. 주 신뢰구간은 피험자별 정확도를 seed 3개에서 평균한 15명으로 bootstrap합니다. seed별 평균도 표에 남깁니다. ST-GCN과 LSTM은 피험자별 정확도 차이를 paired bootstrap해서 차이의 구간을 냅니다. 각 모델의 구간이 겹치는지만으로 우열을 판단하지 않습니다.

end-to-end의 분모는 npz 행이나 학습용 stride 5 윈도우 목록이 아닙니다. 각 테스트 영상의 .mat를 frame_labels로 읽고, 10 fps 샘플 인덱스에서 stride 5마다 끝 프레임을 잡습니다. 겹친 구간은 frame_labels와 같이 행동 순서에서 나중 클래스가 덮어씁니다. 마지막 프레임이 다섯 행동 중 하나인 끝 프레임만 분모에 넣습니다. 그 끝 프레임마다, 그 프레임에서 끝나며 max_gap 12를 만족하는 최근 검출 30장으로 윈도우를 직접 만듭니다. 학습용 윈도우 목록에서 같은 끝 프레임을 찾아 매칭하지 않습니다. 끝 프레임 검출이 없거나 30장이 안 되면 오답입니다. 배경 구간에서 행동을 잘못 출력하는 경우는 이 분모에 없으므로 한계로 적습니다.

저장 파일은 모델마다 summary.json, 혼동행렬, 테스트 피험자별 accuracy입니다.

4. E1
새 Custom skeleton으로 ST-GCN과 LSTM을 위 규칙대로 학습하고, 테스트 피험자 27–41에서 평가합니다.

각 seed의 LSTM best checkpoint를 dynamic weight-only INT8로 변환해, 같은 테스트에 argmax로 평가합니다. ST-GCN은 이 변환으로 크기가 변하지 않으므로 정확도 유지 주장을 붙이지 않습니다.

5. E2
COCO nano, 교사 x, Custom nano를 비교합니다. 분류기 입력은 모두 12관절입니다.

교집합은 세 pose가 모두 잡은 동일 프레임만 사용합니다. 학습 교집합은 피험자 1–20, 검증 교집합은 21–26, 테스트 교집합은 27–41입니다. 테스트 프레임은 학습에 넣지 않습니다. 교집합을 만든 다음, 같은 seed로 balance_ratio=1.5를 학습에만 적용합니다. 개수만 같고 프레임이 다른 비교는 무효입니다. 조기 종료는 검증 교집합의 accuracy만 봅니다. 클래스별 학습 윈도우 수를 적습니다. 소수 클래스가 너무 적으면 이 실험만 피험자 bootstrap 신뢰구간을 냅니다. 이 교집합은 검출이 쉬운 프레임이라 정확도가 낙관적으로 나옵니다.

end-to-end 정확도는 3절의 정답 끝 프레임 전체가 분모이고, 검출 실패와 30장 미달은 오답입니다. 교집합 정확도와 end-to-end 정확도를 함께 보고합니다. Custom과 교사가 비슷하면 기여는 상단 시점 적응이 아니라 경량 증류입니다.

pose 지연은 Jetson Orin NX에서 같은 입력 크기(기본 640 px)로 세 모델 모두 잽니다. 파이프라인 fps도 새 Custom 가중치로 한 번 다시 잽니다.

17관절과 12관절도 비교합니다. 대상은 Custom pose가 아니라 교사 x의 skeleton입니다. 같은 프레임으로 분류기를 두 번 학습합니다. 한 번은 17관절, 한 번은 머리 5개를 뺀 12관절입니다. 17관절 그래프는 기존 COCO17_EDGES를 그대로 씁니다. 결과는 COCO 계열 교사 pose에서 머리 관절을 빼면 분류가 어떻게 달라지는지만 적습니다. 머리 5개의 신뢰도가 이미 낮아 0으로 지워졌는지도 확인합니다.

파라미터 수, 분류기 단독 지연, FP32 파일 크기, LSTM의 dynamic weight-only INT8 파일 크기는 구조가 같으면 다시 재지 않습니다.

---

## Implementation choices (where the text leaves room)

| Item | Choice | Reason |
|---|---|---|
| Arm OKS sigma / 1.5 | Written to `kpt_oks_sigmas` in the pose dataset yaml, so it applies to the loss and to validation | The earlier patch of `ultralytics.utils.metrics.OKS_SIGMA` had no effect on a 12-keypoint model; the protocol asks for the division |
| Pose training | 480 px, batch 24, up to 40 epochs, patience 12, start from `weights/yolo11n-pose.pt` | Arguments of the previous run (`args.yaml`); that run was stopped by hand at 15 epochs, this one runs to its own end |
| New pose weights | `weights/custom_yolo11n_pose_s1_20.pt` | The existing `custom_yolo11n_pose.pt` is left in place and not used |
| Pose settings at extraction | `conf 0.35`, `iou 0.5`, `imgsz 640`, FP32, identical for the three pose models | Config defaults |
| Teacher skeletons | Extracted once with 17 joints; the 12-joint store is `drop_head` of the same rows | Guarantees identical frames for the 17-vs-12 comparison |
| Action training | epochs <= 120, patience 20, batch 128, Adam 1e-3, default augmentation (rotation 30 deg, scale 15%, jitter 0.01, flip 0.5), `train.balance_ratio=1.5`, FP32 | Settings of the existing real-data runs (`runs/train_custom.log`) |
| Classifiers and seeds in E2 | ST-GCN and LSTM, seeds 42/43/44, as in E1 | Section 3 fixes the seeds for all runs |
| End-frame grid (end-to-end) | 10 fps sample index j with j % 5 == 0 | "stride 5마다"; frames too early to have 30 detections count as errors |
| End-to-end classifier for a pose model | The classifier trained on that pose model's full training store | It is the system one would deploy with that pose model |
| Latency benchmark | `tools/benchmark.py --frames 500` on test clip 27_2 from source frame 144, where a person is in view throughout; 640 px, FP32 | The classifier then runs on every frame, so the pipeline figure is not flattered by empty frames |
