"""Per-track 30-frame skeleton buffers plus prediction smoothing.

The thesis buffers 30 frames of skeleton before classifying (§2). With a
tracker in the loop that buffer has to be per person, not per frame, otherwise
two shoppers in the aisle get their poses interleaved into one sequence.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .normalize import add_velocity, normalize_sequence, to_model_input, valid_ratio


@dataclass
class TrackBuffer:
    """Rolling skeleton history for a single tracked person."""

    track_id: int
    window: int
    keypoints: Deque[np.ndarray] = field(default_factory=deque)
    boxes: Deque[np.ndarray] = field(default_factory=deque)
    frames: Deque[int] = field(default_factory=deque)
    last_seen: int = -1
    predictions: Deque[Tuple[int, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.keypoints = deque(self.keypoints, maxlen=self.window)
        self.boxes = deque(self.boxes, maxlen=self.window)
        self.frames = deque(self.frames, maxlen=self.window)

    def push(self, kpts: np.ndarray, box: np.ndarray, frame_idx: int) -> None:
        self.keypoints.append(np.asarray(kpts, dtype=np.float32))
        self.boxes.append(np.asarray(box, dtype=np.float32))
        self.frames.append(int(frame_idx))
        self.last_seen = int(frame_idx)

    @property
    def ready(self) -> bool:
        return len(self.keypoints) >= self.window

    @property
    def filled(self) -> int:
        return len(self.keypoints)

    def sequence(self) -> np.ndarray:
        """Stack the buffer into ``(T, V, 3)``."""
        return np.stack(list(self.keypoints), axis=0)

    def box_sequence(self) -> np.ndarray:
        return np.stack(list(self.boxes), axis=0)


class BufferBank:
    """Owns one :class:`TrackBuffer` per active track id."""

    def __init__(
        self,
        window: int = 30,
        normalize: str = "torso",
        conf_threshold: float = 0.2,
        min_valid_ratio: float = 0.3,
        use_velocity: bool = False,
        max_idle: int = 60,
        smooth_window: int = 5,
    ) -> None:
        self.window = int(window)
        self.normalize = normalize
        self.conf_threshold = float(conf_threshold)
        self.min_valid_ratio = float(min_valid_ratio)
        self.use_velocity = bool(use_velocity)
        self.max_idle = int(max_idle)
        self.smooth_window = int(smooth_window)
        self.buffers: Dict[int, TrackBuffer] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        track_id: int,
        keypoints: np.ndarray,
        box: np.ndarray,
        frame_idx: int,
    ) -> TrackBuffer:
        buf = self.buffers.get(track_id)
        if buf is None:
            buf = TrackBuffer(track_id=track_id, window=self.window)
            buf.predictions = deque(maxlen=max(1, self.smooth_window))
            self.buffers[track_id] = buf
        buf.push(keypoints, box, frame_idx)
        return buf

    def prune(self, frame_idx: int, alive_ids: Optional[List[int]] = None) -> None:
        """Drop buffers whose track has been gone too long."""
        stale = [
            tid
            for tid, buf in self.buffers.items()
            if frame_idx - buf.last_seen > self.max_idle
            or (alive_ids is not None and tid not in alive_ids
                and frame_idx - buf.last_seen > self.window)
        ]
        for tid in stale:
            self.buffers.pop(tid, None)

    # ------------------------------------------------------------------
    def model_input(
        self,
        track_id: int,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Return a ``(C, T, V)`` array, or ``None`` when not classifiable yet."""
        buf = self.buffers.get(track_id)
        if buf is None or not buf.ready:
            return None

        seq = buf.sequence()
        if valid_ratio(seq, self.conf_threshold) < self.min_valid_ratio:
            return None

        norm = normalize_sequence(
            seq,
            mode=self.normalize,
            boxes=buf.box_sequence() if self.normalize == "bbox" else None,
            frame_size=frame_size,
            conf_threshold=self.conf_threshold,
        )
        if self.use_velocity:
            norm = add_velocity(norm)
        return to_model_input(norm)

    def batch_inputs(
        self,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Collect every ready buffer into one batch for a single forward pass."""
        ids: List[int] = []
        tensors: List[np.ndarray] = []
        for tid in sorted(self.buffers):
            arr = self.model_input(tid, frame_size=frame_size)
            if arr is not None:
                ids.append(tid)
                tensors.append(arr)
        if not tensors:
            return [], None
        return ids, np.stack(tensors, axis=0)

    # ------------------------------------------------------------------
    def record_prediction(self, track_id: int, label: int, score: float) -> None:
        buf = self.buffers.get(track_id)
        if buf is not None:
            buf.predictions.append((int(label), float(score)))

    def smoothed(self, track_id: int) -> Optional[Tuple[int, float]]:
        """Majority vote over recent predictions, scored by mean confidence.

        Single-frame argmax flickers between visually similar actions such as
        "Reach To Shelf" and "Hand In Shelf"; voting over the last few windows
        removes most of that without adding noticeable latency.
        """
        buf = self.buffers.get(track_id)
        if buf is None or not buf.predictions:
            return None
        labels = [lbl for lbl, _ in buf.predictions]
        winner, _ = Counter(labels).most_common(1)[0]
        scores = [s for lbl, s in buf.predictions if lbl == winner]
        return winner, float(np.mean(scores))

    def __len__(self) -> int:
        return len(self.buffers)


def sliding_windows(
    seq: np.ndarray,
    window: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index pairs covering ``seq`` with a sliding window."""
    n = len(seq)
    if n < window:
        return []
    return [(s, s + window) for s in range(0, n - window + 1, max(1, stride))]
