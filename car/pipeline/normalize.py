"""Skeleton normalisation.

The thesis never says how raw pixel keypoints become model input, and the
choice matters a lot: a customer two metres from the camera produces the same
action with very different pixel coordinates. Three schemes are implemented so
the decision can be measured rather than assumed.

``bbox``   translate by the person box origin, scale by box width/height.
``torso``  translate to the hip midpoint, scale by torso length (shoulder
           midpoint to hip midpoint). Scale- and position-invariant, and it
           survives a person being partially cropped. This is the default.
``none``   divide by frame size only, keeping absolute position in frame.

All functions take and return arrays shaped ``(T, V, 3)`` holding
``(x, y, score)`` and never mutate their input.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# BODY12 indices (see car/keypoints.py)
_L_SHOULDER, _R_SHOULDER = 0, 1
_L_HIP, _R_HIP = 6, 7

# COCO17 fallbacks
_C_L_SHOULDER, _C_R_SHOULDER = 5, 6
_C_L_HIP, _C_R_HIP = 11, 12

_EPS = 1e-6


def _torso_indices(num_joints: int) -> Tuple[int, int, int, int]:
    if num_joints == 12:
        return _L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP
    if num_joints == 17:
        return _C_L_SHOULDER, _C_R_SHOULDER, _C_L_HIP, _C_R_HIP
    raise ValueError(f"unsupported joint count {num_joints}")


def normalize_sequence(
    seq: np.ndarray,
    mode: str = "torso",
    boxes: Optional[np.ndarray] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    conf_threshold: float = 0.2,
) -> np.ndarray:
    """Normalise a ``(T, V, 3)`` skeleton sequence.

    Args:
        seq: keypoints as ``(x_pixels, y_pixels, score)``.
        mode: ``bbox``, ``torso`` or ``none``.
        boxes: ``(T, 4)`` xyxy person boxes, required for ``bbox`` mode.
        frame_size: ``(width, height)``, required for ``none`` mode.
        conf_threshold: joints below this score are zeroed out.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"expected (T, V, 3), got {seq.shape}")

    out = seq.copy()
    xy = out[..., :2]
    score = out[..., 2:3]

    if mode == "bbox":
        if boxes is None:
            raise ValueError("bbox normalisation needs per-frame boxes")
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        origin = boxes[:, None, :2]
        size = (boxes[:, None, 2:4] - boxes[:, None, :2]).clip(min=_EPS)
        xy = (xy - origin) / size
        xy = xy - 0.5  # centre on the box middle

    elif mode == "torso":
        ls, rs, lh, rh = _torso_indices(seq.shape[1])
        shoulder = (xy[:, ls] + xy[:, rs]) * 0.5
        hip = (xy[:, lh] + xy[:, rh]) * 0.5
        # Fall back to the mean of valid joints when hips are missing.
        hip_ok = (score[:, lh, 0] > conf_threshold) & (score[:, rh, 0] > conf_threshold)
        valid = score[..., 0] > conf_threshold
        any_valid = valid.any(axis=1)
        mean_xy = np.zeros_like(hip)
        for t in np.nonzero(any_valid)[0]:
            mean_xy[t] = xy[t][valid[t]].mean(axis=0)
        center = np.where(hip_ok[:, None], hip, mean_xy)

        torso_len = np.linalg.norm(shoulder - hip, axis=-1)
        shoulder_w = np.linalg.norm(xy[:, ls] - xy[:, rs], axis=-1)
        # Under a top-view camera the torso projects short, so take whichever
        # of torso length / shoulder width is larger as the body scale.
        scale = np.maximum(torso_len, shoulder_w)
        good = scale > _EPS
        if good.any():
            # A single scale per clip keeps relative limb motion intact.
            ref = float(np.median(scale[good]))
        else:
            ref = 1.0
        ref = max(ref, _EPS)
        xy = (xy - center[:, None, :]) / ref

    elif mode == "none":
        if frame_size is None:
            raise ValueError("'none' normalisation needs frame_size")
        w, h = frame_size
        xy = xy / np.array([max(w, 1), max(h, 1)], dtype=np.float32)
        xy = xy - 0.5

    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")

    # Zero out low-confidence joints so the model does not chase noise.
    invalid = score[..., 0] <= conf_threshold
    xy[invalid] = 0.0

    out[..., :2] = xy
    return out


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """Append per-joint frame differences, turning ``(T,V,3)`` into ``(T,V,6)``."""
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    vel[..., 2] = seq[..., 2]  # keep the raw score in the velocity block
    return np.concatenate([seq, vel], axis=-1)


def apply_joint_weights(seq: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Scale each joint's coordinates by its weight (thesis' 0.5 arm bonus)."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    w = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1)
    if w.shape[1] != seq.shape[1]:
        raise ValueError(f"expected {seq.shape[1]} weights, got {w.shape[1]}")
    seq[..., :2] *= w
    return seq


def to_model_input(seq: np.ndarray) -> np.ndarray:
    """``(T, V, C)`` -> ``(C, T, V)``, the layout both models expect."""
    return np.ascontiguousarray(np.asarray(seq, dtype=np.float32).transpose(2, 0, 1))


def valid_ratio(seq: np.ndarray, conf_threshold: float = 0.2) -> float:
    """Fraction of joint-frames whose detection score clears the threshold."""
    seq = np.asarray(seq)
    if seq.size == 0:
        return 0.0
    return float((seq[..., 2] > conf_threshold).mean())
