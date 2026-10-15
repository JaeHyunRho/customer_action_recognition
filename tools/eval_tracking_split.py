#!/usr/bin/env python3
"""Measure what StrongSORT adds to pose estimation (thesis Table 2.2).

The thesis reports accuracy, recall, F1, fps and latency for "Pose estimation"
against "Pose estimation with StrongSORT", and claims about +6 points on each
score. It does not say what those scores are computed against -- MERL Shopping
ships action-interval labels, not tracking ground truth, so there is no
per-frame "is this the right person" annotation to score against.

What is measurable, and what this script reports, is whether the pipeline has a
usable skeleton for the shopper at each frame. On MERL exactly one person is in
shot, so for every sampled frame the ground truth is "one person is present".

  true positive   a skeleton is produced with enough confident joints
  false negative  no skeleton, or too few confident joints
  false positive  more than one person reported (only one is really there)

Tracking cannot add detections the pose model missed, but it can carry a track
across a missed frame through the Kalman prediction, and it suppresses spurious
extra detections by requiring ``n_init`` consecutive hits. Both show up here.

This copy differs from ``tools/eval_tracking.py`` in three ways only: it scores
all 28 test videos of the subject split (subjects 27-41) instead of the first
``--clips`` of them, it runs pose estimation once per frame and applies both
conditions to that same result, and it also reports the frames with a second
usable skeleton and a per-video breakdown. The scoring rules are unchanged.

Usage::

    .venv/bin/python tools/eval_tracking_split.py --merl data/raw/merl \\
        --set pose.weights=weights/custom_yolo11n_pose_s1_20.pt \\
        --out runs/subject_split/analysis/tracking_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import load_config, parse_overrides  # noqa: E402
from car.datasets.merl import discover_clips, iter_sampled_frames, split_official  # noqa: E402
from car.pose.estimator import PoseEstimator  # noqa: E402
from car.pose.tracker import build_tracker  # noqa: E402

LOGGER = logging.getLogger("eval-track")


def score(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    # "Accuracy" here is the MOTA-style detection accuracy: the fraction of
    # frames handled without a miss or a phantom person.
    accuracy = tp / max(tp + fp + fn, 1)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def usable(kpts, conf_threshold: float, min_joints: int) -> list:
    """Skeletons with at least ``min_joints`` joints above ``conf_threshold``."""
    return [
        k for k in kpts
        if k is not None and int((np.asarray(k)[:, 2] > conf_threshold).sum()) >= min_joints
    ]


class Tally:
    """TP/FP/FN of one condition, scored as in tools/eval_tracking.py."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.frames = self.second_person = 0

    def add(self, good: list) -> None:
        self.frames += 1
        if not good:
            self.fn += 1
        else:
            self.tp += 1
            self.fp += len(good) - 1  # only one shopper is really present
        if len(good) >= 2:
            self.second_person += 1

    def result(self) -> dict:
        out = score(self.tp, self.fp, self.fn)
        out.update({"frames": self.frames, "frames_with_second_person": self.second_person})
        return out


def run(clips, cfg, estimator: PoseEstimator, target_fps: int, min_joints: int) -> dict:
    """One pose pass per frame; both conditions are scored on the same result."""
    thr = cfg.sequence.conf_threshold
    total = {"pose_only": Tally(), "pose_with_tracker": Tally()}
    id_switches_total = 0
    per_video = {}
    started = time.time()

    for i, clip in enumerate(clips, 1):
        tracker = build_tracker(cfg.tracker, device=estimator.device)
        clip_tally = {"pose_only": Tally(), "pose_with_tracker": Tally()}
        previous_id = None
        id_switches = 0

        for _, frame in iter_sampled_frames(clip.video_path, target_fps=target_fps):
            result = estimator.infer(frame)

            # (a) pose only
            good = usable(list(result.keypoints), thr, min_joints)
            clip_tally["pose_only"].add(good)
            total["pose_only"].add(good)

            # (b) pose + tracker, on the very same detections
            tracks = tracker.update(result.boxes, result.scores, result.keypoints, frame=frame)
            good = usable([t.keypoints for t in tracks], thr, min_joints)
            clip_tally["pose_with_tracker"].add(good)
            total["pose_with_tracker"].add(good)

            ids = [t.track_id for t in tracks]
            if ids:
                current = ids[0]
                if previous_id is not None and current != previous_id:
                    id_switches += 1
                previous_id = current

        id_switches_total += id_switches
        per_video[clip.clip_id] = {
            "pose_only": clip_tally["pose_only"].result(),
            "pose_with_tracker": {**clip_tally["pose_with_tracker"].result(), "id_switches": id_switches},
        }
        LOGGER.info("[%d/%d] %s: %d frames, usable %d / %d (pose / tracker), id switches %d",
                    i, len(clips), clip.clip_id, clip_tally["pose_only"].frames,
                    clip_tally["pose_only"].tp, clip_tally["pose_with_tracker"].tp, id_switches)

    elapsed = time.time() - started
    return {
        "pose_only": total["pose_only"].result(),
        "pose_with_tracker": {**total["pose_with_tracker"].result(), "id_switches": id_switches_total},
        "per_video": per_video,
        "elapsed_s": round(elapsed, 1),
        "ms_per_frame_pose_and_tracker": round(elapsed * 1000 / max(total["pose_only"].frames, 1), 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pose estimation with and without StrongSORT (thesis Table 2.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--merl", default="data/raw/merl")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="official subject split to score (default: test, subjects 27-41)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-joints", type=int, default=6,
                        help="confident joints needed to count as a usable skeleton")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/subject_split/analysis/tracking_eval.json")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(
        args.config if Path(args.config).exists() else None, parse_overrides(args.set)
    )

    clips = discover_clips(args.merl)
    if not clips:
        LOGGER.error("no clips under %s", args.merl)
        return 1
    tr, va, te = split_official(clips)
    clips = {"train": tr, "val": va, "test": te}[args.split]
    LOGGER.info("scoring %d clips (%s split, subjects %d-%d)", len(clips), args.split,
                min(c.subject for c in clips), max(c.subject for c in clips))

    estimator = PoseEstimator(cfg.pose)
    LOGGER.info("pose: %s", estimator.describe())

    results = run(clips, cfg, estimator, args.fps, args.min_joints)
    a, b = results["pose_only"], results["pose_with_tracker"]
    results["delta"] = {
        k: round(b[k] - a[k], 4) for k in ("accuracy", "precision", "recall", "f1")
    }
    import dataclasses

    results["config"] = {
        "split": args.split,
        "videos": [c.clip_id for c in clips],
        "fps_sampling": args.fps,
        "min_joints": args.min_joints,
        "conf_threshold": cfg.sequence.conf_threshold,
        "pose_weights": cfg.pose.weights,
        "pose": dataclasses.asdict(cfg.pose),
        "tracker": dataclasses.asdict(cfg.tracker),
        "config_file": args.config,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    rows = [("Accuracy", "accuracy", "{:.4f}"), ("Precision", "precision", "{:.4f}"),
            ("Recall", "recall", "{:.4f}"), ("F1", "f1", "{:.4f}"),
            ("Frames with a second person", "frames_with_second_person", "{:,}")]
    md = ["| | Pose only | Pose and tracker |", "|---|---|---|"]
    md += [f"| {label} | {fmt.format(a[k])} | {fmt.format(b[k])} |" for label, k, fmt in rows]
    md += [f"| Identity switches | -- | {b['id_switches']:,} |",
           f"| Frames | {a['frames']:,} | {b['frames']:,} |"]
    out.with_suffix(".md").write_text(
        f"Test videos: {len(clips)} (subjects 27-41), {args.fps} fps sampling, "
        f"usable = at least {args.min_joints} joints with score > {cfg.sequence.conf_threshold}, "
        f"pose weights {cfg.pose.weights}\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
