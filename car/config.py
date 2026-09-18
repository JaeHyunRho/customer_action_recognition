"""Typed configuration objects plus YAML loading/merging.

Every default here is either taken straight from the thesis or, where the
thesis is silent, chosen deliberately and flagged with a ``# CHOICE`` comment
so the gap between "what the paper says" and "what we decided" stays visible.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------
# Action label space
# --------------------------------------------------------------------------
#: The five shopping actions the thesis recognises (Table 4.3).
ACTION_CLASSES: Tuple[str, ...] = (
    "Reach To Shelf",
    "Retract From Shelf",
    "Hand In Shelf",
    "Inspect Product",
    "Inspect Shelf",
)

#: The thesis' confusion matrices (Fig. 4.2) show a sixth "Standing" column,
#: which is the MERL background class. Enabling it is a config switch.
BACKGROUND_CLASS: str = "Standing"


def action_classes(include_background: bool = False) -> Tuple[str, ...]:
    return ACTION_CLASSES + (BACKGROUND_CLASS,) if include_background else ACTION_CLASSES


@dataclass
class PoseConfig:
    """YOLOv11n-Pose settings (thesis §2.2)."""

    weights: str = "weights/yolo11n-pose.pt"
    #: Stock COCO weights used for bootstrapping / pseudo-labelling.
    base_weights: str = "yolo11n-pose.pt"
    imgsz: int = 640
    conf: float = 0.35          # CHOICE: thesis does not state a threshold
    iou: float = 0.5            # CHOICE
    max_det: int = 10           # CHOICE: a retail aisle rarely holds more
    device: str = "auto"
    half: bool = False          # set True on a CUDA-enabled Jetson build
    head_removal: bool = True   # thesis §2.1 / Fig. 2.4
    upper_body_bonus: float = 0.5  # thesis §2.1 "가중치 0.5"
    #: Inference backend: pytorch | onnx | tensorrt | openvino
    backend: str = "pytorch"


@dataclass
class TrackerConfig:
    """StrongSORT settings (thesis §2.3, Fig. 2.7)."""

    enabled: bool = True
    max_age: int = 30           # CHOICE: one buffer length worth of frames
    n_init: int = 3             # thesis: "3회 연속 매칭 성공 시 Confirmed"
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.2
    nn_budget: int = 100
    ema_alpha: float = 0.9      # StrongSORT feature EMA
    mc_lambda: float = 0.98     # motion/appearance blend
    only_position: bool = False
    #: Appearance descriptor: none | histogram | reid
    appearance: str = "histogram"  # CHOICE: ReID net is optional on-device
    reid_weights: Optional[str] = None


@dataclass
class SequenceConfig:
    """How skeleton frames become model inputs."""

    window: int = 30            # thesis §2 "30 프레임의 시간적 정보를 버퍼링"
    train_stride: int = 5       # CHOICE: 6x fewer near-duplicate windows
    infer_stride: int = 1       # CHOICE: predict every frame at inference
    fps: int = 10               # thesis: dataset sampled at 10 fps
    #: bbox | torso | none
    normalize: str = "torso"    # CHOICE, see car/pipeline/normalize.py
    #: Append per-joint frame-to-frame motion to the feature vector.
    use_velocity: bool = False  # CHOICE: off to match the paper's 3 channels
    #: Largest frame-number jump that still counts as one continuous run.
    #: At 10 fps from a 30 fps source the natural step is 3, so 12 tolerates
    #: three consecutive missed detections before a sequence is cut.
    max_gap: int = 12           # CHOICE, see docs/EXPERIMENTS.md
    #: Drop a window if fewer than this fraction of joints are confident.
    min_valid_ratio: float = 0.3
    conf_threshold: float = 0.2


@dataclass
class STGCNConfig:
    """Fine-tuned ST-GCN (thesis §3.1, Fig. 3.1)."""

    in_channels: int = 3        # (x, y, score)
    channels: Tuple[int, ...] = (64, 64, 128, 128, 256, 256)
    groups: int = 2             # CHOICE: thesis says "grouped convolution", not how many
    temporal_kernel: int = 9    # CHOICE: the ST-GCN default
    dropout: float = 0.25       # thesis Table 4.2
    leaky_slope: float = 0.1    # CHOICE
    edge_importance: bool = True


@dataclass
class LSTMConfig:
    """Fine-tuned LSTM (thesis §3.2, Fig. 3.2)."""

    hidden_size: int = 128      # thesis: "각 LSTM 레이어는 128개의 채널"
    num_layers: int = 3         # thesis Fig. 3.2 shows three stacked layers
    dense_hidden: int = 32      # CHOICE: tuned so the param count matches Table 4.4
    dropout: float = 0.25       # thesis Table 4.2
    leaky_slope: float = 0.1


@dataclass
class TrainConfig:
    """Action-model training (thesis Table 4.2)."""

    epochs: int = 300
    batch_size: int = 128
    lr: float = 1e-3            # CHOICE: Adam default, thesis only says "Adam"
    weight_decay: float = 1e-4  # CHOICE
    optimizer: str = "adam"
    loss: str = "cce"
    early_stop_patience: int = 30   # CHOICE: thesis only says "Enabled"
    early_stop_metric: str = "val_acc"
    val_split: float = 0.3      # thesis: 70/30 train/validation
    num_workers: int = 4
    seed: int = 42
    amp: bool = False
    #: Weight the loss by inverse class frequency.
    class_balanced: bool = False
    #: Cap each class's training windows at this multiple of the rarest class.
    #: MERL Shopping is genuinely imbalanced -- "Inspect Shelf" has ~2.9x the
    #: frames of "Hand In Shelf" -- while the thesis' Table 4.3 reports a near
    #: even split, so it must have balanced somehow. 0 disables capping.
    balance_ratio: float = 0.0
    # -- augmentation (the thesis mentions none; these are all CHOICE) ----
    #: Max random rotation in degrees. A ceiling camera has no canonical "up",
    #: but a fixed camera over a fixed shelf does constrain approach angle, so
    #: full 360 rotation throws away usable signal. Measured in docs/EXPERIMENTS.md.
    aug_rotation_deg: float = 30.0
    aug_scale: float = 0.15
    aug_jitter: float = 0.01
    aug_flip_prob: float = 0.5


@dataclass
class PoseTrainConfig:
    """Custom YOLOv11n-Pose fine-tuning (thesis §2.2)."""

    epochs: int = 100           # CHOICE: thesis does not state pose epochs
    batch_size: int = 16        # CHOICE: fits 16 GB Orin NX at 640 px
    imgsz: int = 640
    lr0: float = 1e-3
    patience: int = 25
    pose_weight: float = 12.0   # ultralytics default
    kobj_weight: float = 2.0
    freeze: Optional[int] = None


@dataclass
class PathsConfig:
    root: str = "."
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    videos: str = "data/videos"
    weights: str = "weights"
    runs: str = "runs"


@dataclass
class RuntimeConfig:
    """Live inference behaviour."""

    source: str = "0"           # webcam index, file path or RTSP URL
    action_model: str = "stgcn"  # stgcn | lstm
    action_weights: str = "runs/action/stgcn/best.pt"
    #: Only emit a label once its softmax score clears this.
    min_action_conf: float = 0.4
    #: Majority-vote smoothing over the last N predictions per track.
    smooth_window: int = 5
    display: bool = True
    save_video: Optional[str] = None
    save_csv: Optional[str] = None
    max_fps: Optional[float] = None
    #: Run the classifier every N frames instead of every frame.
    action_every: int = 1
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    #: Use a Jetson CSI camera through GStreamer instead of V4L2.
    csi_camera: bool = False


@dataclass
class Config:
    classes: Tuple[str, ...] = ACTION_CLASSES
    include_background: bool = False
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    stgcn: STGCNConfig = field(default_factory=STGCNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pose_train: PoseTrainConfig = field(default_factory=PoseTrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # -- derived ----------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def label_names(self) -> Tuple[str, ...]:
        if self.include_background and BACKGROUND_CLASS not in self.classes:
            return tuple(self.classes) + (BACKGROUND_CLASS,)
        return tuple(self.classes)

    @property
    def num_keypoints(self) -> int:
        return 12 if self.pose.head_removal else 17

    @property
    def in_channels(self) -> int:
        base = 3  # x, y, score
        return base * 2 if self.sequence.use_velocity else base

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: Any, data: Any) -> Any:
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type
        if dataclasses.is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, value)
        elif isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load a YAML config on top of the dataclass defaults.

    ``overrides`` accepts dotted keys, e.g. ``{"pose.conf": 0.5}``.
    """
    data: Dict[str, Any] = dataclasses.asdict(Config())
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_update(data, loaded)
    if overrides:
        patch: Dict[str, Any] = {}
        for dotted, value in overrides.items():
            cursor = patch
            parts = dotted.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        data = _deep_update(data, patch)

    cfg = Config(
        classes=tuple(data.get("classes", ACTION_CLASSES)),
        include_background=bool(data.get("include_background", False)),
        pose=_build(PoseConfig, data.get("pose", {})),
        tracker=_build(TrackerConfig, data.get("tracker", {})),
        sequence=_build(SequenceConfig, data.get("sequence", {})),
        stgcn=_build(STGCNConfig, data.get("stgcn", {})),
        lstm=_build(LSTMConfig, data.get("lstm", {})),
        train=_build(TrainConfig, data.get("train", {})),
        pose_train=_build(PoseTrainConfig, data.get("pose_train", {})),
        paths=_build(PathsConfig, data.get("paths", {})),
        runtime=_build(RuntimeConfig, data.get("runtime", {})),
    )
    return cfg


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Turn ``["pose.conf=0.5", "train.epochs=50"]`` into a dict."""
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out
