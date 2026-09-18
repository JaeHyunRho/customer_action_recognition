"""Keypoint definitions, head-removal preprocessing and skeleton graphs.

Paper reference: 2.1 / Fig. 2.4 -- the five head landmarks (nose, both eyes,
both ears) are dropped so the model concentrates on torso and arms, which are
what actually distinguish the five shopping actions under a top-view camera.
The paper also gives the shoulder/arm joints an extra weight of 0.5; here that
is realised as a per-keypoint OKS sigma scaling during pose training and as a
feature weight during action training.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# COCO-Pose 17 keypoint layout (what YOLOv11n-Pose ships with)
# --------------------------------------------------------------------------
COCO17_NAMES: Tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

#: Indices of the five facial landmarks removed by the paper's preprocessing.
HEAD_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: The 12 body joints that survive preprocessing, in COCO order.
BODY12_INDICES: Tuple[int, ...] = tuple(
    i for i in range(len(COCO17_NAMES)) if i not in HEAD_INDICES
)
BODY12_NAMES: Tuple[str, ...] = tuple(COCO17_NAMES[i] for i in BODY12_INDICES)

# Official COCO OKS sigmas, one per 17 keypoints.
COCO17_SIGMAS: np.ndarray = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
     0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
    dtype=np.float32,
)
BODY12_SIGMAS: np.ndarray = COCO17_SIGMAS[list(BODY12_INDICES)]

# --------------------------------------------------------------------------
# Skeleton edges
# --------------------------------------------------------------------------
#: Bone list for the 12-joint layout, expressed with BODY12 indices.
BODY12_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # left_shoulder  - right_shoulder
    (0, 2),   # left_shoulder  - left_elbow
    (2, 4),   # left_elbow     - left_wrist
    (1, 3),   # right_shoulder - right_elbow
    (3, 5),   # right_elbow    - right_wrist
    (0, 6),   # left_shoulder  - left_hip
    (1, 7),   # right_shoulder - right_hip
    (6, 7),   # left_hip       - right_hip
    (6, 8),   # left_hip       - left_knee
    (8, 10),  # left_knee      - left_ankle
    (7, 9),   # right_hip      - right_knee
    (9, 11),  # right_knee     - right_ankle
)

#: Bone list for the untouched COCO-17 layout (used when head_removal=False).
COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

#: Joints the paper up-weights (shoulders and the whole arm chain), BODY12 idx.
UPPER_BODY_BODY12: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: Same joints expressed in COCO-17 indices.
UPPER_BODY_COCO17: Tuple[int, ...] = (5, 6, 7, 8, 9, 10)


def layout(head_removal: bool = True) -> Dict[str, object]:
    """Return the active keypoint layout as a plain dict."""
    if head_removal:
        return {
            "num_keypoints": len(BODY12_INDICES),
            "names": BODY12_NAMES,
            "edges": BODY12_EDGES,
            "sigmas": BODY12_SIGMAS,
            "upper_body": UPPER_BODY_BODY12,
            "source_indices": BODY12_INDICES,
        }
    return {
        "num_keypoints": len(COCO17_NAMES),
        "names": COCO17_NAMES,
        "edges": COCO17_EDGES,
        "sigmas": COCO17_SIGMAS,
        "upper_body": UPPER_BODY_COCO17,
        "source_indices": tuple(range(len(COCO17_NAMES))),
    }


def drop_head(keypoints: np.ndarray) -> np.ndarray:
    """Remove the five facial landmarks from a COCO-17 keypoint array.

    Args:
        keypoints: array shaped ``(..., 17, C)`` with ``C`` in ``{2, 3}``.

    Returns:
        Array shaped ``(..., 12, C)`` holding only the body joints.
    """
    keypoints = np.asarray(keypoints)
    if keypoints.shape[-2] == len(BODY12_INDICES):
        return keypoints  # already preprocessed
    if keypoints.shape[-2] != len(COCO17_NAMES):
        raise ValueError(
            f"expected 17 or 12 keypoints, got {keypoints.shape[-2]}"
        )
    return keypoints[..., list(BODY12_INDICES), :]


def keypoint_weights(
    head_removal: bool = True,
    upper_body_bonus: float = 0.5,
) -> np.ndarray:
    """Per-keypoint feature weights.

    Every joint starts at 1.0; the shoulder/elbow/wrist chain receives
    ``upper_body_bonus`` on top, matching the paper's "가중치 0.5를 추가로 부여".
    """
    lay = layout(head_removal)
    weights = np.ones(int(lay["num_keypoints"]), dtype=np.float32)
    for idx in lay["upper_body"]:  # type: ignore[union-attr]
        weights[idx] += float(upper_body_bonus)
    return weights


def oks_sigmas(
    head_removal: bool = True,
    upper_body_bonus: float = 0.5,
) -> np.ndarray:
    """OKS sigmas rescaled so up-weighted joints are penalised harder.

    A smaller sigma means the OKS term decays faster with distance, so the
    keypoint loss effectively cares more about that joint. Dividing by
    ``1 + bonus`` is the natural way to turn a feature weight into a sigma.
    """
    lay = layout(head_removal)
    sigmas = np.array(lay["sigmas"], dtype=np.float32).copy()
    for idx in lay["upper_body"]:  # type: ignore[union-attr]
        sigmas[idx] = sigmas[idx] / (1.0 + float(upper_body_bonus))
    return sigmas


def edge_index(head_removal: bool = True, self_loops: bool = True) -> np.ndarray:
    """Return the skeleton edges as an ``(2, E)`` index array."""
    lay = layout(head_removal)
    edges: List[Tuple[int, int]] = list(lay["edges"])  # type: ignore[arg-type]
    pairs: List[Tuple[int, int]] = []
    for a, b in edges:
        pairs.append((a, b))
        pairs.append((b, a))
    if self_loops:
        pairs.extend((i, i) for i in range(int(lay["num_keypoints"])))
    return np.asarray(pairs, dtype=np.int64).T


def skeleton_for_plot(head_removal: bool = True) -> Sequence[Tuple[int, int]]:
    """Edges in the form the drawing helpers expect."""
    return layout(head_removal)["edges"]  # type: ignore[return-value]
