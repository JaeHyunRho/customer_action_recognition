"""Skeleton sequence storage and the torch Dataset the action models train on.

Skeletons live in a single ``.npz`` shard per split::

    keypoints  (N, V, 3)  float32, pixel coordinates
    boxes      (N, 4)     float32, xyxy
    labels     (N,)       int64
    clip_ids   (N,)       int32, index into ``clip_names``
    track_ids  (N,)       int32
    frames     (N,)       int32, source frame index
    clip_names (M,)       str
    frame_size (N, 2)     int32, width and height

Windows are cut lazily at ``__getitem__`` time, so changing the window length
or the stride does not mean re-extracting skeletons from video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..pipeline.normalize import add_velocity, normalize_sequence, to_model_input, valid_ratio

LOGGER = logging.getLogger(__name__)


# ==========================================================================
# Storage
# ==========================================================================
@dataclass
class SkeletonStore:
    """Flat, frame-indexed skeleton table."""

    keypoints: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    clip_ids: np.ndarray
    track_ids: np.ndarray
    frames: np.ndarray
    clip_names: List[str]
    frame_size: np.ndarray

    def __len__(self) -> int:
        return int(len(self.keypoints))

    @property
    def num_keypoints(self) -> int:
        return int(self.keypoints.shape[1])

    # -- io -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keypoints=self.keypoints.astype(np.float32),
            boxes=self.boxes.astype(np.float32),
            labels=self.labels.astype(np.int64),
            clip_ids=self.clip_ids.astype(np.int32),
            track_ids=self.track_ids.astype(np.int32),
            frames=self.frames.astype(np.int32),
            clip_names=np.array(self.clip_names, dtype=object),
            frame_size=self.frame_size.astype(np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SkeletonStore":
        data = np.load(str(path), allow_pickle=True)
        return cls(
            keypoints=data["keypoints"],
            boxes=data["boxes"],
            labels=data["labels"],
            clip_ids=data["clip_ids"],
            track_ids=data["track_ids"],
            frames=data["frames"],
            clip_names=[str(x) for x in data["clip_names"]],
            frame_size=data["frame_size"],
        )

    @classmethod
    def from_records(cls, records: Sequence[Dict]) -> "SkeletonStore":
        """Build a store from per-frame dicts emitted during extraction."""
        if not records:
            raise ValueError("no skeleton records to store")
        names: List[str] = []
        name_to_idx: Dict[str, int] = {}
        for rec in records:
            name = rec["clip"]
            if name not in name_to_idx:
                name_to_idx[name] = len(names)
                names.append(name)
        return cls(
            keypoints=np.stack([r["keypoints"] for r in records]).astype(np.float32),
            boxes=np.stack([r["box"] for r in records]).astype(np.float32),
            labels=np.array([r["label"] for r in records], dtype=np.int64),
            clip_ids=np.array([name_to_idx[r["clip"]] for r in records], dtype=np.int32),
            track_ids=np.array([r.get("track_id", 0) for r in records], dtype=np.int32),
            frames=np.array([r["frame"] for r in records], dtype=np.int32),
            clip_names=names,
            frame_size=np.array(
                [r.get("frame_size", (0, 0)) for r in records], dtype=np.int32
            ),
        )

    # -- windowing ------------------------------------------------------
    def build_windows(
        self,
        window: int,
        stride: int,
        label_mode: str = "last",
        drop_mixed: bool = False,
        keep_labels: Optional[Sequence[int]] = None,
        max_gap: int = 6,
    ) -> List[Tuple[np.ndarray, int]]:
        """Cut contiguous ``(clip, track)`` runs into labelled windows.

        Args:
            label_mode: ``last`` takes the window's final frame (causal, what
                live inference sees), ``majority`` takes the modal label.
            drop_mixed: discard windows spanning more than one action.
            keep_labels: only emit windows whose label is in this set.
            max_gap: largest frame-number jump that still counts as the same
                run. At 10 fps sampled from 30 fps source the natural step is
                3, so 6 tolerates one missed detection and 12 tolerates three.
        """
        if label_mode not in {"last", "majority"}:
            raise ValueError(f"unknown label_mode {label_mode!r}")

        order = np.lexsort((self.frames, self.track_ids, self.clip_ids))
        windows: List[Tuple[np.ndarray, int]] = []

        group_start = 0
        for pos in range(1, len(order) + 1):
            at_end = pos == len(order)
            if not at_end:
                prev, cur = order[pos - 1], order[pos]
                same_run = (
                    self.clip_ids[prev] == self.clip_ids[cur]
                    and self.track_ids[prev] == self.track_ids[cur]
                    # Allow a small frame gap; the extractor samples at 10 fps
                    # and a missed detection should not split the run.
                    and 0 < self.frames[cur] - self.frames[prev] <= max_gap
                )
                if same_run:
                    continue

            run = order[group_start:pos]
            group_start = pos
            if len(run) < window:
                continue

            run_labels = self.labels[run]
            for s in range(0, len(run) - window + 1, max(1, stride)):
                idx = run[s : s + window]
                seg = run_labels[s : s + window]
                if label_mode == "last":
                    label = int(seg[-1])
                else:
                    label = int(np.bincount(seg).argmax())
                if drop_mixed and len(np.unique(seg)) > 1:
                    continue
                if keep_labels is not None and label not in keep_labels:
                    continue
                windows.append((idx, label))
        return windows

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)[:num_classes]


# ==========================================================================
# Torch dataset
# ==========================================================================
class SkeletonWindowDataset:
    """Windows of skeleton, normalised and laid out as ``(C, T, V)``.

    Implemented as a plain class so importing this module does not require
    torch; it satisfies the ``torch.utils.data.Dataset`` protocol.
    """

    def __init__(
        self,
        store: SkeletonStore,
        cfg: Config,
        windows: Optional[List[Tuple[np.ndarray, int]]] = None,
        stride: Optional[int] = None,
        augment: bool = False,
        label_mode: str = "last",
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.augment = augment
        self.window = int(cfg.sequence.window)
        self.windows = windows if windows is not None else store.build_windows(
            window=self.window,
            stride=int(stride if stride is not None else cfg.sequence.train_stride),
            label_mode=label_mode,
            max_gap=int(getattr(cfg.sequence, "max_gap", 6)),
            keep_labels=tuple(range(cfg.num_classes)),
        )
        self._rng = np.random.default_rng(cfg.train.seed)

    def __len__(self) -> int:
        return len(self.windows)

    # ------------------------------------------------------------------
    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Rotation, scale, jitter and horizontal flip.

        A ceiling camera has no canonical "up", so a customer can approach the
        shelf from any bearing. Random rotation is therefore the single most
        useful augmentation here, and the thesis does not mention any.
        """
        seq = seq.copy()
        xy = seq[..., :2]

        max_rot = np.deg2rad(float(getattr(self.cfg.train, "aug_rotation_deg", 30.0)))
        theta = self._rng.uniform(-max_rot, max_rot)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        xy = xy @ rot.T

        scale_amp = float(getattr(self.cfg.train, "aug_scale", 0.15))
        xy *= self._rng.uniform(1.0 - scale_amp, 1.0 + scale_amp)
        jitter = float(getattr(self.cfg.train, "aug_jitter", 0.01))
        if jitter > 0:
            xy += self._rng.normal(0.0, jitter, size=xy.shape).astype(np.float32)

        if self._rng.random() < float(getattr(self.cfg.train, "aug_flip_prob", 0.5)):
            xy[..., 0] *= -1.0
            # Swapping left/right joints keeps the skeleton anatomically valid.
            if seq.shape[1] == 12:
                pairs = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
            else:
                pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
            for a, b in pairs:
                xy[:, [a, b]] = xy[:, [b, a]]
                seq[:, [a, b], 2] = seq[:, [b, a], 2]

        seq[..., :2] = xy
        return seq

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        import torch

        idx, label = self.windows[index]
        seq = self.store.keypoints[idx]                  # (T, V, 3)
        boxes = self.store.boxes[idx]
        size = self.store.frame_size[idx][-1]

        seq = normalize_sequence(
            seq,
            mode=self.cfg.sequence.normalize,
            boxes=boxes if self.cfg.sequence.normalize == "bbox" else None,
            frame_size=(int(size[0]), int(size[1])) if size.size == 2 else None,
            conf_threshold=self.cfg.sequence.conf_threshold,
        )
        if self.augment:
            seq = self._augment(seq)
        if self.cfg.sequence.use_velocity:
            seq = add_velocity(seq)

        x = torch.from_numpy(to_model_input(seq))
        return x, int(label)

    # ------------------------------------------------------------------
    def label_counts(self) -> np.ndarray:
        counts = np.zeros(self.cfg.num_classes, dtype=np.int64)
        for _, label in self.windows:
            if 0 <= label < len(counts):
                counts[label] += 1
        return counts

    def describe(self) -> str:
        counts = self.label_counts()
        names = self.cfg.label_names
        parts = ", ".join(f"{n}={c}" for n, c in zip(names, counts))
        return f"{len(self)} windows ({parts})"


def keep_primary_track(store: SkeletonStore) -> SkeletonStore:
    """Collapse each clip to one person, chosen per frame by box area.

    MERL Shopping films one shopper at a time, so any second track is a
    detection artefact -- a reflection, a chair, or the same person re-acquired
    under a new id after the tracker lost them. Left alone those artefacts
    shred the sequence: on this dataset the tracker produced 5,258 runs with a
    median length of 5 frames, far short of the 30 a window needs.

    Merging to a single track per clip is only correct when the footage really
    does contain one person. Do not use it on multi-shopper video.
    """
    order = np.lexsort((store.frames, store.clip_ids))
    keep: List[int] = []

    pos = 0
    while pos < len(order):
        end = pos + 1
        while (
            end < len(order)
            and store.clip_ids[order[end]] == store.clip_ids[order[pos]]
            and store.frames[order[end]] == store.frames[order[pos]]
        ):
            end += 1
        candidates = order[pos:end]
        if len(candidates) == 1:
            keep.append(int(candidates[0]))
        else:
            boxes = store.boxes[candidates]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep.append(int(candidates[int(np.argmax(areas))]))
        pos = end

    idx = np.asarray(sorted(keep), dtype=np.int64)
    LOGGER.info(
        "primary-track filter: %d -> %d frames, %d -> 1 track per clip",
        len(store), len(idx), len(np.unique(store.track_ids)),
    )
    return SkeletonStore(
        keypoints=store.keypoints[idx],
        boxes=store.boxes[idx],
        labels=store.labels[idx],
        clip_ids=store.clip_ids[idx],
        track_ids=np.ones(len(idx), dtype=np.int32),
        frames=store.frames[idx],
        clip_names=store.clip_names,
        frame_size=store.frame_size[idx],
    )


def filter_store(
    store: SkeletonStore,
    min_valid: float = 0.3,
    conf_threshold: float = 0.2,
) -> SkeletonStore:
    """Drop frames whose skeleton is too sparse to be informative."""
    ratio = (store.keypoints[..., 2] > conf_threshold).mean(axis=1)
    keep = ratio >= min_valid
    if keep.all():
        return store
    LOGGER.info("dropping %d/%d low-quality frames", int((~keep).sum()), len(store))
    return SkeletonStore(
        keypoints=store.keypoints[keep],
        boxes=store.boxes[keep],
        labels=store.labels[keep],
        clip_ids=store.clip_ids[keep],
        track_ids=store.track_ids[keep],
        frames=store.frames[keep],
        clip_names=store.clip_names,
        frame_size=store.frame_size[keep],
    )
