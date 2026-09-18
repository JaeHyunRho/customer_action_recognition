"""YOLOv11n-Pose wrapper with the thesis' head-removal preprocessing.

Handles both weight flavours transparently:

* stock COCO weights that predict 17 keypoints, from which the five facial
  landmarks are dropped after inference, and
* a custom model fine-tuned on 12-keypoint top-view labels, whose output is
  already head-free.

Backends: PyTorch (default), ONNX Runtime and TensorRT, all routed through
ultralytics so the call site never changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import PoseConfig
from ..keypoints import drop_head

LOGGER = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Detections for one frame."""

    boxes: np.ndarray       # (N, 4) xyxy in pixels
    scores: np.ndarray      # (N,) person confidence
    keypoints: np.ndarray   # (N, V, 3) x, y, score in pixels
    frame_shape: Tuple[int, int]  # (height, width)

    def __len__(self) -> int:
        return int(len(self.boxes))

    @property
    def empty(self) -> bool:
        return len(self.boxes) == 0

    @staticmethod
    def blank(num_keypoints: int, frame_shape: Tuple[int, int]) -> "PoseResult":
        return PoseResult(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            keypoints=np.zeros((0, num_keypoints, 3), dtype=np.float32),
            frame_shape=frame_shape,
        )


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string, degrading to CPU when CUDA is unusable.

    On a Jetson this is the line that matters: a PyTorch build compiled for a
    newer CUDA than the installed driver reports ``is_available() == False``,
    and we would rather run slowly on CPU than crash.
    """
    import torch

    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class PoseEstimator:
    """Thin, allocation-light wrapper around ``ultralytics.YOLO``."""

    def __init__(self, cfg: Optional[PoseConfig] = None, weights: Optional[str] = None) -> None:
        from ultralytics import YOLO

        self.cfg = cfg or PoseConfig()
        path = weights or self.cfg.weights
        if not Path(path).exists():
            fallback = self.cfg.base_weights
            LOGGER.warning(
                "pose weights %s not found; falling back to stock %s", path, fallback
            )
            path = fallback

        self.weights_path = str(path)
        self.device = resolve_device(self.cfg.device)
        self.model = YOLO(self.weights_path, task="pose")

        # Only a .pt model can be moved between devices. An exported engine or
        # ONNX graph is already bound to one, and `.to()` on it raises; the
        # device is passed per call at predict time instead.
        self.is_native = Path(self.weights_path).suffix.lower() in {".pt", ".pth", ""}
        if self.is_native and self.device.startswith("cuda"):
            try:
                self.model.to(self.device)
            except Exception as exc:  # pragma: no cover - device-specific
                LOGGER.warning("could not move model to %s (%s); using CPU", self.device, exc)
                self.device = "cpu"

        # A TensorRT engine carries its own precision, chosen when it was built.
        self.half = bool(self.cfg.half) and self.device.startswith("cuda") and self.is_native
        self._native_keypoints: Optional[int] = None

    # ------------------------------------------------------------------
    @property
    def num_keypoints(self) -> int:
        """Keypoint count *after* preprocessing."""
        if self._native_keypoints == 12:
            return 12
        return 12 if self.cfg.head_removal else 17

    # ------------------------------------------------------------------
    def infer(self, frame: np.ndarray, verbose: bool = False) -> PoseResult:
        """Run pose estimation on a single BGR frame."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            device=self.device,
            half=self.half,
            verbose=verbose,
        )
        return self._postprocess(results[0], (h, w))

    def infer_batch(self, frames: Sequence[np.ndarray], verbose: bool = False) -> List[PoseResult]:
        if not frames:
            return []
        results = self.model.predict(
            list(frames),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            device=self.device,
            half=self.half,
            verbose=verbose,
        )
        return [
            self._postprocess(r, frames[i].shape[:2])
            for i, r in enumerate(results)
        ]

    # ------------------------------------------------------------------
    def _postprocess(self, result, frame_shape: Tuple[int, int]) -> PoseResult:
        kp_obj = getattr(result, "keypoints", None)
        box_obj = getattr(result, "boxes", None)
        if kp_obj is None or box_obj is None or kp_obj.data is None or len(box_obj) == 0:
            return PoseResult.blank(self.num_keypoints, frame_shape)

        kpts = kp_obj.data.detach().cpu().numpy().astype(np.float32)  # (N, V, 3)
        boxes = box_obj.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = box_obj.conf.detach().cpu().numpy().astype(np.float32)

        if kpts.ndim != 3 or kpts.shape[0] == 0:
            return PoseResult.blank(self.num_keypoints, frame_shape)

        self._native_keypoints = int(kpts.shape[1])
        if self.cfg.head_removal and kpts.shape[1] == 17:
            kpts = drop_head(kpts)

        return PoseResult(
            boxes=boxes,
            scores=scores,
            keypoints=kpts,
            frame_shape=frame_shape,
        )

    # ------------------------------------------------------------------
    def warmup(self, imgsz: Optional[int] = None) -> None:
        """One dummy forward so the first real frame is not the slow one."""
        size = int(imgsz or self.cfg.imgsz)
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self.infer(dummy)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("warmup failed: %s", exc)

    def describe(self) -> str:
        return (
            f"YOLO pose | weights={Path(self.weights_path).name} "
            f"device={self.device} half={self.half} "
            f"keypoints={self.num_keypoints} imgsz={self.cfg.imgsz}"
        )
