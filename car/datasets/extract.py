"""Turn video into a labelled skeleton store.

Two entry points:

``extract_merl``  MERL Shopping videos plus their ``.mat`` action ranges, which
                  is the training path the thesis describes.
``extract_video``  any single clip, optionally with a fixed label, used both for
                  quick demos and for labelling your own top-view footage.

Both run the same per-frame loop: YOLOv11n-Pose, head removal, StrongSORT, then
one record per (frame, track).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..pose.estimator import PoseEstimator
from ..pose.tracker import build_tracker
from .merl import BACKGROUND_LABEL, MerlClip, iter_sampled_frames, video_frame_count
from .sequence import SkeletonStore

LOGGER = logging.getLogger(__name__)


def _records_from_clip(
    video_path: Path,
    clip_name: str,
    estimator: PoseEstimator,
    cfg: Config,
    frame_labels: Optional[np.ndarray] = None,
    target_fps: int = 10,
    max_frames: Optional[int] = None,
    progress: bool = True,
) -> List[Dict]:
    """Run pose + tracking over one video and emit per-track records."""
    tracker = build_tracker(cfg.tracker, device=estimator.device)
    records: List[Dict] = []
    seen = 0

    iterator: Iterable = iter_sampled_frames(video_path, target_fps=target_fps)
    if progress:
        try:
            from tqdm import tqdm

            total = None
            count = video_frame_count(video_path)
            if count > 0:
                total = max(1, count // max(1, int(round(30 / max(target_fps, 1)))))
            iterator = tqdm(iterator, total=total, desc=clip_name, unit="f", leave=False)
        except ImportError:
            pass

    for frame_idx, frame in iterator:
        result = estimator.infer(frame)
        tracks = tracker.update(
            result.boxes, result.scores, result.keypoints, frame=frame
        )
        h, w = frame.shape[:2]

        label = BACKGROUND_LABEL
        if frame_labels is not None:
            if frame_idx < len(frame_labels):
                label = int(frame_labels[frame_idx])
            else:
                continue  # past the labelled range

        for track in tracks:
            records.append(
                {
                    "clip": clip_name,
                    "frame": int(frame_idx),
                    "track_id": int(track.track_id),
                    "keypoints": np.asarray(track.keypoints, dtype=np.float32),
                    "box": np.asarray(track.box, dtype=np.float32),
                    "label": int(label),
                    "frame_size": (int(w), int(h)),
                }
            )

        seen += 1
        if max_frames is not None and seen >= max_frames:
            break

    return records


def extract_merl(
    clips: Sequence[MerlClip],
    cfg: Config,
    estimator: Optional[PoseEstimator] = None,
    target_fps: int = 10,
    max_frames_per_clip: Optional[int] = None,
    drop_background: bool = True,
) -> SkeletonStore:
    """Build a skeleton store from MERL clips.

    Args:
        drop_background: discard frames no action range covers. Keep them
            (``False``) together with ``include_background: true`` in the
            config to reproduce the six-column confusion matrices in Fig. 4.2.
    """
    estimator = estimator or PoseEstimator(cfg.pose)
    LOGGER.info("pose backend: %s", estimator.describe())

    all_records: List[Dict] = []
    for i, clip in enumerate(clips, 1):
        total = video_frame_count(clip.video_path)
        labels = clip.frame_labels(total)
        LOGGER.info("[%d/%d] %s (%d frames)", i, len(clips), clip.clip_id, total)
        recs = _records_from_clip(
            clip.video_path,
            clip.clip_id,
            estimator,
            cfg,
            frame_labels=labels,
            target_fps=target_fps,
            max_frames=max_frames_per_clip,
        )
        if drop_background:
            recs = [r for r in recs if r["label"] != BACKGROUND_LABEL]
        all_records.extend(recs)

    if not all_records:
        raise RuntimeError(
            "no skeletons extracted. Check that the videos decode and that "
            "people are actually detected (try lowering pose.conf)."
        )
    return SkeletonStore.from_records(all_records)


def extract_video(
    video_path: str | Path,
    cfg: Config,
    label: Optional[int] = None,
    clip_name: Optional[str] = None,
    estimator: Optional[PoseEstimator] = None,
    target_fps: Optional[int] = None,
    max_frames: Optional[int] = None,
    frame_labels: Optional[np.ndarray] = None,
) -> SkeletonStore:
    """Extract skeletons from one arbitrary clip.

    ``label`` tags every frame with one class, which is how you label a short
    clip you recorded yourself. ``frame_labels`` gives per-frame control.
    """
    video_path = Path(video_path)
    estimator = estimator or PoseEstimator(cfg.pose)
    name = clip_name or video_path.stem

    labels = frame_labels
    if labels is None and label is not None:
        labels = np.full(video_frame_count(video_path) + 1, int(label), dtype=np.int64)

    records = _records_from_clip(
        video_path,
        name,
        estimator,
        cfg,
        frame_labels=labels,
        target_fps=int(target_fps or cfg.sequence.fps),
        max_frames=max_frames,
    )
    if not records:
        raise RuntimeError(f"no person detected in {video_path}")
    return SkeletonStore.from_records(records)


def merge_stores(stores: Sequence[SkeletonStore]) -> SkeletonStore:
    """Concatenate stores, renumbering clip ids so names stay unique."""
    stores = [s for s in stores if len(s)]
    if not stores:
        raise ValueError("nothing to merge")
    if len(stores) == 1:
        return stores[0]

    names: List[str] = []
    name_to_idx: Dict[str, int] = {}
    clip_ids: List[np.ndarray] = []
    for store in stores:
        remap = np.zeros(len(store.clip_names), dtype=np.int32)
        for local, name in enumerate(store.clip_names):
            if name not in name_to_idx:
                name_to_idx[name] = len(names)
                names.append(name)
            remap[local] = name_to_idx[name]
        clip_ids.append(remap[store.clip_ids])

    return SkeletonStore(
        keypoints=np.concatenate([s.keypoints for s in stores]),
        boxes=np.concatenate([s.boxes for s in stores]),
        labels=np.concatenate([s.labels for s in stores]),
        clip_ids=np.concatenate(clip_ids),
        track_ids=np.concatenate([s.track_ids for s in stores]),
        frames=np.concatenate([s.frames for s in stores]),
        clip_names=names,
        frame_size=np.concatenate([s.frame_size for s in stores]),
    )
