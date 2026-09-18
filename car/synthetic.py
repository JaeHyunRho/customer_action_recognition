"""A synthetic top-view shopper, for testing the pipeline without a camera.

Why this exists: the action half of the system cannot be exercised at all
without labelled skeleton sequences, and until the MERL Shopping dataset is
downloaded or a camera is connected there are none. This module generates
12-joint skeletons whose *motion* matches how each of the five actions actually
looks from a ceiling camera, so the normalisation, windowing, training and
inference code can be run and measured today.

It is a test fixture, not a dataset. Numbers obtained from it say the plumbing
works; they say nothing about accuracy on real shoppers.

The five actions, as seen from above:

============================  ===================================================
Reach To Shelf                one wrist travels outward, away from the torso
Hand In Shelf                 the wrist stays extended, small jitter
Retract From Shelf            the wrist travels back toward the torso
Inspect Product               both wrists held close in front, slow sway
Inspect Shelf                 arms down at the sides, torso drifts along the aisle
============================  ===================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ACTION_CLASSES

#: Order the demo cycles through, matching ``ACTION_CLASSES`` indices.
ACTION_SEQUENCE: Tuple[str, ...] = (
    "Inspect Shelf",
    "Reach To Shelf",
    "Hand In Shelf",
    "Retract From Shelf",
    "Inspect Product",
)

# BODY12 joint order (see car/keypoints.py):
# 0 l_shoulder 1 r_shoulder 2 l_elbow 3 r_elbow 4 l_wrist 5 r_wrist
# 6 l_hip      7 r_hip      8 l_knee  9 r_knee  10 l_ankle 11 r_ankle
_NUM_JOINTS = 12


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    return points @ rot.T


@dataclass
class SyntheticShopper:
    """Animates one shopper through the five actions in a loop."""

    origin: Tuple[float, float] = (480.0, 300.0)
    scale: float = 90.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    #: Frames spent on each action before moving to the next.
    hold: int = 40
    noise: float = 0.012

    def __post_init__(self) -> None:
        self.t = 0
        self.action_idx = int(self.rng.integers(0, len(ACTION_SEQUENCE)))
        self.phase = int(self.rng.integers(0, self.hold))
        self.heading = float(self.rng.uniform(-0.4, 0.4))
        self.drift = np.zeros(2, dtype=np.float32)
        self.reach_side = int(self.rng.integers(0, 2))  # 0 left arm, 1 right

    # ------------------------------------------------------------------
    @property
    def action(self) -> str:
        return ACTION_SEQUENCE[self.action_idx]

    @property
    def label(self) -> int:
        return ACTION_CLASSES.index(self.action)

    # ------------------------------------------------------------------
    def _body_frame(self, progress: float) -> np.ndarray:
        """Skeleton in body-local units, before rotation and translation.

        Seen from above, the torso is a short quadrilateral and the arms are
        what move; the legs barely change, which is exactly why the thesis
        drops the head and keeps the arm chain.
        """
        action = self.action
        s = 1.0

        # Static torso and legs. y grows "forward", toward the shelf.
        l_sh = np.array([-0.30, 0.18], dtype=np.float32) * s
        r_sh = np.array([+0.30, 0.18], dtype=np.float32) * s
        l_hip = np.array([-0.22, -0.20], dtype=np.float32) * s
        r_hip = np.array([+0.22, -0.20], dtype=np.float32) * s
        l_knee = np.array([-0.20, -0.42], dtype=np.float32) * s
        r_knee = np.array([+0.20, -0.42], dtype=np.float32) * s
        l_ank = np.array([-0.19, -0.60], dtype=np.float32) * s
        r_ank = np.array([+0.19, -0.60], dtype=np.float32) * s

        # Arm extension in [0, 1]: 0 = tucked at the side, 1 = fully reaching.
        if action == "Reach To Shelf":
            extend = progress
        elif action == "Hand In Shelf":
            extend = 0.92 + 0.06 * np.sin(progress * np.pi * 4)
        elif action == "Retract From Shelf":
            extend = 1.0 - progress
        elif action == "Inspect Product":
            extend = 0.34 + 0.05 * np.sin(progress * np.pi * 3)
        else:  # Inspect Shelf
            extend = 0.10

        both_arms = action == "Inspect Product"

        def arm(shoulder: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
            active = both_arms or side == self.reach_side
            e = extend if active else 0.10
            sign = -1.0 if side == 0 else 1.0
            # Elbow swings outward then forward as the arm extends.
            elbow = shoulder + np.array(
                [sign * (0.22 - 0.10 * e), 0.10 + 0.26 * e], dtype=np.float32
            ) * s
            if both_arms:
                # Hands meet in front of the chest to hold a product.
                wrist = shoulder + np.array(
                    [sign * (0.22 - 0.19 * e), 0.16 + 0.36 * e], dtype=np.float32
                ) * s
            else:
                wrist = elbow + np.array(
                    [sign * (0.06 - 0.04 * e), 0.10 + 0.40 * e], dtype=np.float32
                ) * s
            return elbow, wrist

        l_el, l_wr = arm(l_sh, 0)
        r_el, r_wr = arm(r_sh, 1)

        return np.stack(
            [l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self) -> Tuple[np.ndarray, str]:
        """Advance one frame and return ``((12, 3) keypoints, action_name)``."""
        progress = (self.phase % self.hold) / float(self.hold)
        action = self.action

        body = self._body_frame(progress)

        # Torso motion: "Inspect Shelf" walks the aisle, the rest stay put.
        if action == "Inspect Shelf":
            self.drift[0] += float(np.sin(self.t * 0.06)) * 0.012 * self.scale
            self.heading += 0.004 * float(np.sin(self.t * 0.05))
        else:
            self.drift *= 0.97
            self.heading += 0.002 * float(np.sin(self.t * 0.09))

        body = _rotate(body, self.heading)
        body = body * self.scale + np.asarray(self.origin, dtype=np.float32) + self.drift
        body += self.rng.normal(0.0, self.noise * self.scale, body.shape).astype(np.float32)

        scores = np.full((_NUM_JOINTS, 1), 0.9, dtype=np.float32)
        # Occasional missed joints, as a real detector would produce.
        drop = self.rng.random(_NUM_JOINTS) < 0.03
        scores[drop] = 0.05

        self.phase += 1
        self.t += 1
        if self.phase % self.hold == 0:
            self.action_idx = (self.action_idx + 1) % len(ACTION_SEQUENCE)
            if self.rng.random() < 0.3:
                self.reach_side = int(self.rng.integers(0, 2))

        return np.concatenate([body, scores], axis=1), action

    # ------------------------------------------------------------------
    def box(self, keypoints: np.ndarray, margin: float = 0.18) -> np.ndarray:
        xy = keypoints[:, :2]
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        pad = (hi - lo) * margin + 6.0
        return np.concatenate([lo - pad, hi + pad]).astype(np.float32)

    def draw(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        from .utils.viz import draw_skeleton

        return draw_skeleton(frame, keypoints, conf_threshold=0.2, radius=4, thickness=3)


def draw_scene(width: int, height: int) -> np.ndarray:
    """Background that looks vaguely like an aisle floor with shelving."""
    import cv2

    frame = np.full((height, width, 3), 196, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width, int(height * 0.22)), (168, 176, 188), -1)
    for x in range(0, width, max(60, width // 12)):
        cv2.rectangle(
            frame, (x + 6, int(height * 0.04)), (x + 48, int(height * 0.19)),
            (140, 150, 170), -1,
        )
    cv2.line(frame, (0, int(height * 0.22)), (width, int(height * 0.22)), (110, 118, 130), 3)
    for y in range(int(height * 0.3), height, 70):
        cv2.line(frame, (0, y), (width, y), (186, 186, 186), 1)
    return frame


# ==========================================================================
# Skeleton store generation
# ==========================================================================
def generate_store(
    num_clips: int = 24,
    frames_per_clip: int = 300,
    people_per_clip: int = 1,
    seed: int = 0,
    hold: int = 40,
):
    """Build a labelled :class:`~car.datasets.sequence.SkeletonStore`.

    Each clip is an independent shopper, so a clip-level train/val split has
    genuinely unseen sequences on the validation side.
    """
    from .datasets.sequence import SkeletonStore

    rng = np.random.default_rng(seed)
    records: List[Dict] = []

    for clip in range(num_clips):
        shoppers = [
            SyntheticShopper(
                origin=(
                    float(rng.uniform(220, 740)),
                    float(rng.uniform(220, 380)),
                ),
                scale=float(rng.uniform(70, 110)),
                rng=np.random.default_rng(seed * 1000 + clip * 10 + p),
                hold=hold,
            )
            for p in range(people_per_clip)
        ]
        for frame_idx in range(frames_per_clip):
            for track_id, shopper in enumerate(shoppers, start=1):
                kpts, action = shopper.step()
                records.append(
                    {
                        "clip": f"synth_{clip:03d}",
                        "frame": frame_idx,
                        "track_id": track_id,
                        "keypoints": kpts,
                        "box": shopper.box(kpts),
                        "label": ACTION_CLASSES.index(action),
                        "frame_size": (960, 540),
                    }
                )
    return SkeletonStore.from_records(records)
