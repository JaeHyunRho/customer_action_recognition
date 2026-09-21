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

Usage::

    .venv/bin/python tools/eval_tracking.py --merl data/raw/merl \\
        --clips 20 --out runs/eval_tracking.json
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
from car.datasets.merl import discover_clips, iter_sampled_frames  # noqa: E402
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


def run(
    clips,
    cfg,
    estimator: PoseEstimator,
    use_tracker: bool,
    target_fps: int,
    min_joints: int,
) -> dict:
    tracker_cfg = cfg.tracker
    tp = fp = fn = 0
    frames = 0
    id_switches = 0
    started = time.time()

    for clip in clips:
        tracker = build_tracker(tracker_cfg, device=estimator.device) if use_tracker else None
        previous_id = None

        for _, frame in iter_sampled_frames(clip.video_path, target_fps=target_fps):
            frames += 1
            result = estimator.infer(frame)

            if use_tracker:
                tracks = tracker.update(
                    result.boxes, result.scores, result.keypoints, frame=frame
                )
                kpts = [t.keypoints for t in tracks]
                ids = [t.track_id for t in tracks]
            else:
                kpts = list(result.keypoints)
                ids = []

            good = [
                k for k in kpts
                if k is not None
                and int((np.asarray(k)[:, 2] > cfg.sequence.conf_threshold).sum()) >= min_joints
            ]

            if not good:
                fn += 1
            else:
                tp += 1
                fp += len(good) - 1  # only one shopper is really present

            if use_tracker and ids:
                current = ids[0]
                if previous_id is not None and current != previous_id:
                    id_switches += 1
                previous_id = current

    elapsed = time.time() - started
    out = score(tp, fp, fn)
    out.update({
        "frames": frames,
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "latency_ms": round(elapsed * 1000 / max(frames, 1), 2),
    })
    if use_tracker:
        out["id_switches"] = id_switches
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pose estimation with and without StrongSORT (thesis Table 2.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--merl", default="data/raw/merl")
    parser.add_argument("--clips", type=int, default=20, help="how many clips to score")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-joints", type=int, default=6,
                        help="confident joints needed to count as a usable skeleton")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/eval_tracking.json")
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
    # Score the official test subjects, which the action models never saw.
    clips = [c for c in clips if c.subject >= 27][: args.clips]
    LOGGER.info("scoring %d clips (test subjects)", len(clips))

    estimator = PoseEstimator(cfg.pose)
    LOGGER.info("pose: %s", estimator.describe())

    results = {}
    for tag, use_tracker in (("pose_only", False), ("pose_with_strongsort", True)):
        LOGGER.info("evaluating %s ...", tag)
        results[tag] = run(clips, cfg, estimator, use_tracker, args.fps, args.min_joints)
        r = results[tag]
        LOGGER.info(
            "  acc %.3f  recall %.3f  f1 %.3f  %.1f fps  %.1f ms",
            r["accuracy"], r["recall"], r["f1"], r["fps"], r["latency_ms"],
        )

    a, b = results["pose_only"], results["pose_with_strongsort"]
    results["delta"] = {
        k: round(b[k] - a[k], 4) for k in ("accuracy", "precision", "recall", "f1")
    }
    results["config"] = {
        "clips": len(clips),
        "fps_sampling": args.fps,
        "min_joints": args.min_joints,
        "pose_weights": cfg.pose.weights,
        "tracker": cfg.tracker.appearance,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print()
    print(f"{'Metric':<14}{'Pose estimation':>18}{'with StrongSORT':>18}{'delta':>9}")
    print("-" * 59)
    for key, label in (("accuracy", "Accuracy"), ("recall", "Recall"), ("f1", "F1-score")):
        print(f"{label:<14}{a[key]:>18.3f}{b[key]:>18.3f}{results['delta'][key]:>+9.3f}")
    print(f"{'Fps':<14}{a['fps']:>18.1f}{b['fps']:>18.1f}")
    print(f"{'Latency':<14}{a['latency_ms']:>16.1f}ms{b['latency_ms']:>16.1f}ms")
    if "id_switches" in b:
        print(f"{'ID switches':<14}{'':>18}{b['id_switches']:>18}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
