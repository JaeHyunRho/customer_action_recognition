#!/usr/bin/env python3
"""Build a 12-keypoint top-view pose dataset by pseudo-labelling with a teacher.

The thesis says its top-view frames were labelled "in YOLO format" but not how.
126,350 frames is far past hand-labelling, so the practical reading is that a
model produced the labels. This script makes that explicit: a larger pose model
(``yolo11x-pose`` by default) labels the frames, the five facial keypoints are
dropped, weak detections are filtered out, and the result is written as an
ultralytics pose dataset that ``car/train/train_pose.py`` fine-tunes the nano
model on.

The teacher runs at a high confidence threshold and only its single best
detection per frame is kept, because a wrong pseudo-label is worse than a
missing one.

Usage::

    python3 tools/autolabel_pose.py --videos data/raw/merl/videos \
        --out data/processed/pose_dataset --teacher yolo11x-pose.pt --fps 2
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import load_config, parse_overrides  # noqa: E402
from car.datasets.merl import iter_sampled_frames  # noqa: E402
from car.keypoints import drop_head  # noqa: E402
from car.pose.estimator import resolve_device  # noqa: E402
from car.train.train_pose import write_dataset_yaml  # noqa: E402

LOGGER = logging.getLogger("autolabel")

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def to_yolo_pose_line(
    box: np.ndarray,
    keypoints: np.ndarray,
    width: int,
    height: int,
    conf_threshold: float = 0.3,
) -> str:
    """One YOLO pose label row: ``cls cx cy w h (x y v) * V``, all normalised."""
    x1, y1, x2, y2 = box[:4]
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]

    for x, y, score in keypoints:
        # visibility: 2 = labelled and visible, 0 = not labelled
        v = 2 if score >= conf_threshold else 0
        nx = float(np.clip(x / width, 0.0, 1.0)) if v else 0.0
        ny = float(np.clip(y / height, 0.0, 1.0)) if v else 0.0
        parts.append(f"{nx:.6f} {ny:.6f} {v}")
    return " ".join(parts)


def main(argv=None) -> int:
    import cv2
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(
        description="Pseudo-label top-view frames into a YOLO pose dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--videos", required=True, help="directory of source videos")
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--teacher", default="yolo11x-pose.pt",
                        help="teacher weights; downloads on first use")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="frames sampled per second (default 2, keeps the set diverse)")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="teacher confidence floor; deliberately strict")
    parser.add_argument("--kpt-conf", type=float, default=0.3,
                        help="per-keypoint score below which a joint is marked unlabelled")
    parser.add_argument("--min-keypoints", type=int, default=8,
                        help="skip a frame unless this many joints are confident")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=None, help="global cap")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--keep-head", action="store_true",
                        help="keep all 17 keypoints instead of the thesis' 12")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(
        args.config if Path(args.config).exists() else None, parse_overrides(args.set)
    )
    head_removal = not args.keep_head

    videos = sorted(
        p for p in Path(args.videos).rglob("*") if p.suffix.lower() in _VIDEO_EXT
    )
    if not videos:
        LOGGER.error("no videos under %s", args.videos)
        return 1
    LOGGER.info("found %d videos", len(videos))

    root = Path(args.out)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.pose.device)
    LOGGER.info("teacher: %s on %s", args.teacher, device)
    if device == "cpu":
        LOGGER.warning(
            "teacher runs on CPU; yolo11x-pose is heavy. Consider --teacher "
            "yolo11m-pose.pt or a low --fps."
        )
    teacher = YOLO(args.teacher, task="pose")

    rng = random.Random(args.seed)
    kept = skipped = 0

    for vi, video in enumerate(videos, 1):
        LOGGER.info("[%d/%d] %s", vi, len(videos), video.name)
        for frame_idx, frame in iter_sampled_frames(video, target_fps=max(1, int(round(args.fps)))):
            if args.max_frames and kept >= args.max_frames:
                break
            h, w = frame.shape[:2]
            result = teacher.predict(
                frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False
            )[0]

            if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
                skipped += 1
                continue

            kpts = result.keypoints.data.cpu().numpy().astype(np.float32)
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)

            # Keep only the most confident person; a top-view frame with two
            # partially visible shoppers produces unreliable pseudo-labels.
            best = int(np.argmax(scores))
            kp = kpts[best]
            if head_removal:
                kp = drop_head(kp[None])[0]

            if int((kp[:, 2] >= args.kpt_conf).sum()) < args.min_keypoints:
                skipped += 1
                continue

            split = "val" if rng.random() < args.val_ratio else "train"
            stem = f"{video.stem}_{frame_idx:06d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), frame)
            (root / "labels" / split / f"{stem}.txt").write_text(
                to_yolo_pose_line(boxes[best], kp, w, h, args.kpt_conf) + "\n"
            )
            kept += 1

            if kept % 500 == 0:
                LOGGER.info("  %d labelled, %d skipped", kept, skipped)

        if args.max_frames and kept >= args.max_frames:
            LOGGER.info("reached --max-frames")
            break

    if kept == 0:
        LOGGER.error(
            "nothing labelled. The teacher found no confident person -- try a "
            "lower --conf or check that the videos are top-view footage of people."
        )
        return 1

    yaml_path = write_dataset_yaml(root, root / "dataset.yaml", head_removal=head_removal)
    LOGGER.info("labelled %d frames (%d skipped)", kept, skipped)
    LOGGER.info("dataset spec: %s", yaml_path)
    LOGGER.info(
        "next: python3 -m car.train.train_pose --data %s --weights %s",
        yaml_path, cfg.pose.base_weights,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
