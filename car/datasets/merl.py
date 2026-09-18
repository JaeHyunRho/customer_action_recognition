"""MERL Shopping Dataset reader.

Layout produced by the official release::

    data/raw/merl/
        videos/   1_1_crop.mp4, 1_2_crop.mp4, ...
        labels/   1_1_label.mat, 1_2_label.mat, ...

Each ``*_label.mat`` holds a MATLAB variable ``tlabs``: a 5x1 cell array whose
i-th cell is an ``Nx2`` matrix of inclusive ``[start_frame, end_frame]`` ranges
for action ``i``. Frames covered by no range are background ("Standing" in the
thesis' confusion matrices).

Action order is fixed by the dataset and matches ``car.config.ACTION_CLASSES``:
    1 Reach To Shelf, 2 Retract From Shelf, 3 Hand In Shelf,
    4 Inspect Product, 5 Inspect Shelf
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ACTION_CLASSES

LOGGER = logging.getLogger(__name__)

#: Label index used for frames no action range covers.
BACKGROUND_LABEL = len(ACTION_CLASSES)

_NAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<session>\d+)")


@dataclass
class MerlClip:
    """One video plus its per-frame labels."""

    video_path: Path
    label_path: Optional[Path]
    subject: int
    session: int
    #: Per-action inclusive frame ranges, 5 entries of ``(N, 2)``.
    ranges: List[np.ndarray]
    num_frames: Optional[int] = None

    @property
    def clip_id(self) -> str:
        return f"{self.subject}_{self.session}"

    def frame_labels(self, num_frames: Optional[int] = None) -> np.ndarray:
        """Dense per-frame label array, ``BACKGROUND_LABEL`` where nothing applies.

        Overlapping ranges are resolved last-writer-wins in action order, which
        matches how the dataset is normally consumed.
        """
        total = int(num_frames or self.num_frames or 0)
        if total <= 0:
            total = int(max((r[:, 1].max() for r in self.ranges if len(r)), default=0)) + 1
        labels = np.full(total, BACKGROUND_LABEL, dtype=np.int64)
        for action_idx, spans in enumerate(self.ranges):
            for start, end in np.asarray(spans, dtype=np.int64).reshape(-1, 2):
                s = max(int(start) - 1, 0)      # MATLAB is 1-based
                e = min(int(end), total)        # inclusive -> exclusive
                if e > s:
                    labels[s:e] = action_idx
        return labels


def _load_tlabs(path: Path) -> List[np.ndarray]:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    key = "tlabs" if "tlabs" in mat else next(
        (k for k in mat if not k.startswith("__")), None
    )
    if key is None:
        raise ValueError(f"no label variable found in {path}")
    raw = mat[key]

    ranges: List[np.ndarray] = []
    flat = np.asarray(raw).reshape(-1)
    for cell in flat:
        arr = np.asarray(cell)
        while arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.reshape(-1)[0])
        if arr.size == 0:
            ranges.append(np.zeros((0, 2), dtype=np.int64))
        else:
            ranges.append(np.asarray(arr, dtype=np.int64).reshape(-1, 2))

    while len(ranges) < len(ACTION_CLASSES):
        ranges.append(np.zeros((0, 2), dtype=np.int64))
    return ranges[: len(ACTION_CLASSES)]


def discover_clips(
    root: str | Path,
    videos_dir: str = "videos",
    labels_dir: str = "labels",
    require_labels: bool = True,
) -> List[MerlClip]:
    """Pair every video under ``root`` with its label file."""
    root = Path(root)
    vdir = root / videos_dir if (root / videos_dir).is_dir() else root
    ldir = root / labels_dir if (root / labels_dir).is_dir() else root

    clips: List[MerlClip] = []
    for video in sorted(vdir.glob("*.mp4")) + sorted(vdir.glob("*.avi")):
        m = _NAME_RE.match(video.stem)
        if m is None:
            LOGGER.debug("skipping %s: name does not look like <subject>_<session>", video.name)
            continue
        subject = int(m.group("subject"))
        session = int(m.group("session"))

        label = None
        for candidate in (
            ldir / f"{subject}_{session}_label.mat",
            ldir / f"{video.stem}_label.mat",
            ldir / f"{video.stem}.mat",
        ):
            if candidate.exists():
                label = candidate
                break

        if label is None:
            if require_labels:
                LOGGER.warning("no label file for %s; skipping", video.name)
                continue
            ranges = [np.zeros((0, 2), dtype=np.int64) for _ in ACTION_CLASSES]
        else:
            try:
                ranges = _load_tlabs(label)
            except Exception as exc:
                LOGGER.warning("could not read %s (%s); skipping", label.name, exc)
                continue

        clips.append(
            MerlClip(
                video_path=video,
                label_path=label,
                subject=subject,
                session=session,
                ranges=ranges,
            )
        )
    return clips


def split_by_subject(
    clips: Sequence[MerlClip],
    val_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[MerlClip], List[MerlClip]]:
    """Split train/val by subject, never by frame.

    The thesis reports a 70/30 split but does not say at which granularity.
    Splitting frames would put near-identical neighbouring frames on both sides
    and inflate validation accuracy, so we hold out whole subjects.
    """
    subjects = sorted({c.subject for c in clips})
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_subjects = set(shuffled[:n_val])
    train = [c for c in clips if c.subject not in val_subjects]
    val = [c for c in clips if c.subject in val_subjects]
    return train, val


#: The split the dataset's own ReadMe prescribes, by subject id.
OFFICIAL_SPLIT = {
    "train": range(1, 21),   # subjects 1-20, 60 videos
    "val": range(21, 27),    # subjects 21-26, 18 videos
    "test": range(27, 42),   # subjects 27-41, 28 videos
}


def split_official(
    clips: Sequence[MerlClip],
) -> Tuple[List[MerlClip], List[MerlClip], List[MerlClip]]:
    """Split by the dataset's published train/val/test subject ranges.

    The thesis reports a 70/30 split without saying at what granularity. The
    dataset ships its own split, which is what published numbers on MERL
    Shopping are normally measured against, so results stay comparable.
    """
    buckets: Dict[str, List[MerlClip]] = {"train": [], "val": [], "test": []}
    for clip in clips:
        for name, subjects in OFFICIAL_SPLIT.items():
            if clip.subject in subjects:
                buckets[name].append(clip)
                break
    return buckets["train"], buckets["val"], buckets["test"]


def split_by_clip(
    clips: Sequence[MerlClip],
    val_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[MerlClip], List[MerlClip]]:
    """Random clip-level split, closer to a literal reading of the thesis."""
    rng = np.random.default_rng(seed)
    order = list(range(len(clips)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_ratio)))
    val_idx = set(order[:n_val])
    train = [c for i, c in enumerate(clips) if i not in val_idx]
    val = [c for i, c in enumerate(clips) if i in val_idx]
    return train, val


def label_histogram(clips: Sequence[MerlClip], num_frames: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Frame count per class, for sanity-checking against Table 4.3."""
    counts = np.zeros(len(ACTION_CLASSES) + 1, dtype=np.int64)
    for clip in clips:
        total = None if num_frames is None else num_frames.get(clip.clip_id)
        labels = clip.frame_labels(total)
        binc = np.bincount(labels, minlength=len(counts))
        counts += binc[: len(counts)]
    names = list(ACTION_CLASSES) + ["Standing"]
    return {name: int(counts[i]) for i, name in enumerate(names)}


def iter_sampled_frames(
    video_path: str | Path,
    target_fps: int = 10,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(original_frame_index, frame)`` decimated to ``target_fps``.

    The thesis samples the 30 fps source at 10 fps, which is what produces its
    126,350 images from 106 two-minute videos.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(target_fps, 1))))

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_frame_count(video_path: str | Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
