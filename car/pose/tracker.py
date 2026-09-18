"""StrongSORT tracking, implemented in-tree (thesis §2.3, Fig. 2.7).

Written from the paper rather than pulled in as a dependency, because the
pip-installable trackers drag in torchreid/boxmot and a second copy of torch,
which is the wrong trade on a 16 GB Jetson.

What is here, matching Fig. 2.7 and the StrongSORT paper:

* Kalman filter on ``(cx, cy, aspect, height)`` with NSA-weighted measurement
  noise, so a low-confidence detection moves the state less;
* matching cascade over track age, then IoU matching on what is left;
* Tentative -> Confirmed after ``n_init`` consecutive hits, Deleted after
  ``max_age`` frames without an update;
* EMA appearance bank instead of the DeepSORT feature list.

Appearance descriptors are pluggable: an HSV colour histogram by default (no
extra model, works on-device), an optional ReID network, or motion only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackerConfig

_INF = 1e5
_CHI2_95_4DOF = 9.4877  # gating threshold for a 4-dof measurement


# ==========================================================================
# Kalman filter
# ==========================================================================
class KalmanBoxFilter:
    """Constant-velocity filter on ``(cx, cy, a, h)`` and their derivatives."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4, dtype=np.float32)].astype(np.float32)
        h = measurement[3]
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def project(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        confidence: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project to measurement space, NSA-scaling the noise by confidence."""
        h = mean[3]
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        # StrongSORT's NSA Kalman: a confident detection is trusted more.
        std = std * (1.0 - float(np.clip(confidence, 0.0, 0.99)))
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        cov = self._update_mat @ cov @ self._update_mat.T
        return mean, cov + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov, confidence)
        kalman_gain = np.linalg.solve(
            proj_cov.T, (cov @ self._update_mat.T).T
        ).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        proj_mean, proj_cov = self.project(mean, cov)
        if only_position:
            proj_mean, proj_cov = proj_mean[:2], proj_cov[:2, :2]
            measurements = measurements[:, :2]
        try:
            chol = np.linalg.cholesky(proj_cov)
        except np.linalg.LinAlgError:
            return np.full(len(measurements), _INF, dtype=np.float32)
        d = measurements - proj_mean
        z = np.linalg.solve(chol, d.T)
        return np.sum(z * z, axis=0).astype(np.float32)


# ==========================================================================
# Geometry helpers
# ==========================================================================
def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    w = max(float(x2 - x1), 1e-3)
    h = max(float(y2 - y1), 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float32)


def xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = state[:4]
    w = max(float(a * h), 1e-3)
    h = max(float(h), 1e-3)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N,4)`` / ``(M,4)`` xyxy box sets."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


# ==========================================================================
# Appearance descriptors
# ==========================================================================
class HistogramDescriptor:
    """HSV colour histogram of the person crop, L2-normalised.

    Cheap, model-free and good enough to keep two shoppers in different
    clothing apart, which is the realistic failure case in a retail aisle.
    """

    def __init__(self, bins: Tuple[int, int, int] = (8, 8, 4)) -> None:
        self.bins = bins
        self.dim = int(np.prod(bins))

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        import cv2

        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        h, w = frame.shape[:2]
        feats = np.zeros((len(boxes), self.dim), dtype=np.float32)
        for i, box in enumerate(boxes):
            x1 = int(np.clip(box[0], 0, w - 1))
            y1 = int(np.clip(box[1], 0, h - 1))
            x2 = int(np.clip(box[2], x1 + 1, w))
            y2 = int(np.clip(box[3], y1 + 1, h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]
            ).flatten()
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                feats[i] = hist / norm
        return feats


class ReIDDescriptor:  # pragma: no cover - needs an external checkpoint
    """Optional torchreid/OSNet embedding, for crowded scenes."""

    def __init__(self, weights: str, device: str = "cpu", size: Tuple[int, int] = (128, 256)) -> None:
        import torch

        self.device = device
        self.size = size
        self.model = torch.jit.load(weights, map_location=device).eval()
        self.dim = 512

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        h, w = frame.shape[:2]
        crops = []
        for box in boxes:
            x1 = int(np.clip(box[0], 0, w - 1))
            y1 = int(np.clip(box[1], 0, h - 1))
            x2 = int(np.clip(box[2], x1 + 1, w))
            y2 = int(np.clip(box[3], y1 + 1, h))
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, self.size)
            crops.append(crop[:, :, ::-1].astype(np.float32) / 255.0)
        batch = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2)).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return (feats / np.clip(norms, 1e-6, None)).astype(np.float32)


# ==========================================================================
# Track
# ==========================================================================
class TrackState(IntEnum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    n_init: int
    max_age: int
    feature: Optional[np.ndarray] = None
    ema_alpha: float = 0.9
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    keypoints: Optional[np.ndarray] = None
    score: float = 0.0
    history: List[np.ndarray] = field(default_factory=list)

    # -- lifecycle ------------------------------------------------------
    def predict(self, kf: KalmanBoxFilter) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanBoxFilter,
        box: np.ndarray,
        score: float,
        feature: Optional[np.ndarray],
        keypoints: Optional[np.ndarray],
    ) -> None:
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, xyxy_to_xyah(box), confidence=score
        )
        if feature is not None and feature.size:
            if self.feature is None:
                self.feature = feature
            else:
                # StrongSORT EMA: smooth the appearance bank instead of
                # keeping every past feature around.
                blended = self.ema_alpha * self.feature + (1 - self.ema_alpha) * feature
                norm = np.linalg.norm(blended)
                self.feature = blended / norm if norm > 1e-6 else blended
        self.keypoints = keypoints
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    # -- accessors ------------------------------------------------------
    @property
    def tlbr(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


# ==========================================================================
# Tracker
# ==========================================================================
@dataclass
class TrackOutput:
    """One confirmed track at one frame."""

    track_id: int
    box: np.ndarray
    score: float
    keypoints: np.ndarray
    time_since_update: int


class StrongSort:
    """Matching cascade + IoU matching over Kalman-predicted tracks."""

    def __init__(self, cfg: Optional[TrackerConfig] = None, device: str = "cpu") -> None:
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanBoxFilter()
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0

        self.descriptor = None
        if self.cfg.appearance == "histogram":
            self.descriptor = HistogramDescriptor()
        elif self.cfg.appearance == "reid":
            if not self.cfg.reid_weights:
                raise ValueError("appearance='reid' needs tracker.reid_weights")
            self.descriptor = ReIDDescriptor(self.cfg.reid_weights, device=device)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    # ------------------------------------------------------------------
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        keypoints: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackOutput]:
        """Advance the tracker by one frame and return confirmed tracks."""
        self.frame_count += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        keypoints = np.asarray(keypoints, dtype=np.float32)

        features: Optional[np.ndarray] = None
        if self.descriptor is not None and frame is not None and len(boxes):
            features = self.descriptor(frame, boxes)

        for track in self.tracks:
            track.predict(self.kf)

        matches, unmatched_tracks, unmatched_dets = self._match(boxes, scores, features)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(
                self.kf,
                boxes[d_idx],
                float(scores[d_idx]),
                None if features is None else features[d_idx],
                keypoints[d_idx] if len(keypoints) > d_idx else None,
            )
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()
        for d_idx in unmatched_dets:
            self._initiate(
                boxes[d_idx],
                float(scores[d_idx]),
                None if features is None else features[d_idx],
                keypoints[d_idx] if len(keypoints) > d_idx else None,
            )

        self.tracks = [t for t in self.tracks if not t.is_deleted]

        return [
            TrackOutput(
                track_id=t.track_id,
                box=t.tlbr,
                score=t.score,
                keypoints=(
                    t.keypoints
                    if t.keypoints is not None
                    else np.zeros((keypoints.shape[1] if keypoints.ndim == 3 else 12, 3), np.float32)
                ),
                time_since_update=t.time_since_update,
            )
            for t in self.tracks
            if t.is_confirmed and t.time_since_update == 0
        ]

    # ------------------------------------------------------------------
    def _initiate(
        self,
        box: np.ndarray,
        score: float,
        feature: Optional[np.ndarray],
        keypoints: Optional[np.ndarray],
    ) -> None:
        mean, cov = self.kf.initiate(xyxy_to_xyah(box))
        track = Track(
            track_id=self._next_id,
            mean=mean,
            covariance=cov,
            n_init=self.cfg.n_init,
            max_age=self.cfg.max_age,
            feature=feature,
            ema_alpha=self.cfg.ema_alpha,
            keypoints=keypoints,
            score=score,
        )
        self.tracks.append(track)
        self._next_id += 1

    # ------------------------------------------------------------------
    def _match(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed]
        det_indices = list(range(len(boxes)))

        # --- stage 1: matching cascade over appearance + motion ----------
        # Runs even without an appearance descriptor: the cascade is also what
        # lets a track that the detector missed for several frames be picked
        # up again, since IoU matching below only considers fresh tracks.
        matches_a: List[Tuple[int, int]] = []
        remaining_dets = det_indices
        if confirmed and remaining_dets:
            matches_a, _, remaining_dets = self._matching_cascade(
                confirmed, remaining_dets, boxes, scores, features
            )
            unmatched_a = [i for i in confirmed if i not in {t for t, _ in matches_a}]
        else:
            unmatched_a = list(confirmed)

        # --- stage 2: IoU matching on the leftovers ----------------------
        # Only tracks seen very recently join IoU matching; an old track
        # matched purely on overlap is how ids get swapped.
        iou_candidates = unconfirmed + [
            i for i in unmatched_a if self.tracks[i].time_since_update <= 1
        ]
        stale = [i for i in unmatched_a if self.tracks[i].time_since_update > 1]

        matches_b, unmatched_b, unmatched_dets = self._iou_match(
            iou_candidates, remaining_dets, boxes
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(stale + unmatched_b))
        return matches, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    def _matching_cascade(
        self,
        track_indices: Sequence[int],
        det_indices: Sequence[int],
        boxes: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match by track age: freshest tracks get first pick of detections.

        With ``features``, cost is appearance distance gated and blended with
        Kalman motion. Without them, cost is motion alone, which still lets a
        briefly occluded track reconnect.
        """
        matches: List[Tuple[int, int]] = []
        remaining = list(det_indices)
        measurements = np.stack([xyxy_to_xyah(b) for b in boxes]) if len(boxes) else np.zeros((0, 4), np.float32)
        use_appearance = features is not None and len(features) == len(boxes)

        for level in range(self.cfg.max_age):
            if not remaining:
                break
            level_tracks = [
                i for i in track_indices if self.tracks[i].time_since_update == level + 1
            ]
            if not level_tracks:
                continue

            cost = np.zeros((len(level_tracks), len(remaining)), dtype=np.float32)
            for r, t_idx in enumerate(level_tracks):
                track = self.tracks[t_idx]
                gating = self.kf.gating_distance(
                    track.mean, track.covariance, measurements[remaining],
                    only_position=self.cfg.only_position,
                )
                valid = gating <= _CHI2_95_4DOF
                motion = gating / _CHI2_95_4DOF

                if use_appearance and track.feature is not None:
                    appearance = 1.0 - (features[remaining] @ track.feature)
                    appearance[appearance > self.cfg.max_cosine_distance] = _INF
                    blended = (
                        self.cfg.mc_lambda * appearance
                        + (1 - self.cfg.mc_lambda) * motion
                    )
                    cost[r, :] = np.where(appearance >= _INF, _INF, blended)
                else:
                    cost[r, :] = motion
                cost[r, ~valid] = _INF

            rows, cols = linear_sum_assignment(cost)
            matched_dets: List[int] = []
            for r, c in zip(rows, cols):
                if cost[r, c] >= _INF:
                    continue
                matches.append((level_tracks[r], remaining[c]))
                matched_dets.append(remaining[c])
            remaining = [d for d in remaining if d not in matched_dets]

        matched_tracks = {t for t, _ in matches}
        unmatched_tracks = [i for i in track_indices if i not in matched_tracks]
        return matches, unmatched_tracks, remaining

    # ------------------------------------------------------------------
    def _iou_match(
        self,
        track_indices: Sequence[int],
        det_indices: Sequence[int],
        boxes: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_indices = list(track_indices)
        det_indices = list(det_indices)
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        track_boxes = np.stack([self.tracks[i].tlbr for i in track_indices])
        det_boxes = boxes[det_indices]
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        cost[cost > self.cfg.max_iou_distance] = _INF

        rows, cols = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= _INF:
                continue
            matches.append((track_indices[r], det_indices[c]))
            matched_t.add(track_indices[r])
            matched_d.add(det_indices[c])

        unmatched_tracks = [i for i in track_indices if i not in matched_t]
        unmatched_dets = [i for i in det_indices if i not in matched_d]
        return matches, unmatched_tracks, unmatched_dets


# ==========================================================================
# Trivial fallback
# ==========================================================================
class IdentityTracker:
    """Assign an id per detection index. Only useful for ablations."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.frame_count = 0

    def reset(self) -> None:
        self.frame_count = 0

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        keypoints: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackOutput]:
        self.frame_count += 1
        return [
            TrackOutput(
                track_id=i + 1,
                box=np.asarray(boxes[i], dtype=np.float32),
                score=float(scores[i]),
                keypoints=np.asarray(keypoints[i], dtype=np.float32),
                time_since_update=0,
            )
            for i in range(len(boxes))
        ]


def build_tracker(cfg: TrackerConfig, device: str = "cpu"):
    return StrongSort(cfg, device=device) if cfg.enabled else IdentityTracker()
