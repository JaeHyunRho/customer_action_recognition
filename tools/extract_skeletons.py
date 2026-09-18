#!/usr/bin/env python3
"""Extract skeleton sequences from video into a training-ready ``.npz`` store.

Three modes:

``--merl ROOT``       MERL Shopping videos plus their ``.mat`` label files.
``--video PATH``      one clip, with ``--label`` naming its single action.
``--video-dir DIR``   a directory whose subfolders are class names::

                          clips/
                            Reach To Shelf/   a.mp4  b.mp4
                            Hand In Shelf/    c.mp4

The last mode is how you label your own top-view footage: record one short clip
per action and drop it in the matching folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import load_config, parse_overrides  # noqa: E402
from car.datasets.extract import extract_merl, extract_video, merge_stores  # noqa: E402
from car.datasets.merl import discover_clips, label_histogram, split_official  # noqa: E402
from car.datasets.sequence import SkeletonStore, filter_store  # noqa: E402
from car.pose.estimator import PoseEstimator  # noqa: E402

LOGGER = logging.getLogger("extract")


def _report(store: SkeletonStore, cfg, out_path: Path) -> None:
    names = list(cfg.label_names)
    if len(names) < int(store.labels.max(initial=0)) + 1:
        names = names + ["Standing"]
    counts = np.bincount(store.labels, minlength=len(names))[: len(names)]

    LOGGER.info("saved %d frames to %s", len(store), out_path)
    for name, count in zip(names, counts):
        LOGGER.info("  %-22s %7d frames", name, int(count))

    windows = store.build_windows(cfg.sequence.window, cfg.sequence.train_stride)
    LOGGER.info(
        "  -> %d windows of %d frames at stride %d",
        len(windows), cfg.sequence.window, cfg.sequence.train_stride,
    )
    if not windows:
        LOGGER.warning(
            "no windows could be cut. Tracks are too short or too broken; try "
            "a smaller sequence.window or a lower pose.conf."
        )

    stats = {
        "frames": len(store),
        "clips": len(store.clip_names),
        "tracks": int(len(np.unique(store.track_ids))),
        "windows": len(windows),
        "per_class": {n: int(c) for n, c in zip(names, counts)},
    }
    out_path.with_suffix(".json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract skeletons from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--merl", metavar="ROOT", help="MERL Shopping dataset root")
    group.add_argument("--video", metavar="PATH", help="single video file")
    group.add_argument("--video-dir", metavar="DIR", help="directory of per-class folders")

    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--label", default=None,
                        help="class name or index for --video")
    parser.add_argument("--fps", type=int, default=None,
                        help="sampling rate (default: sequence.fps, thesis uses 10)")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--split", default=None,
                        choices=["train", "val", "test", "trainval"],
                        help="MERL only: extract just this official subject split "
                             "(train=1-20, val=21-26, test=27-41)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap frames per clip, useful for a smoke test")
    parser.add_argument("--keep-background", action="store_true",
                        help="keep frames outside any labelled action range")
    parser.add_argument("--min-valid", type=float, default=None,
                        help="drop frames whose detected joint ratio is below this")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    config_path = args.config if Path(args.config).exists() else None
    cfg = load_config(config_path, parse_overrides(args.set))
    fps = int(args.fps or cfg.sequence.fps)
    names = list(cfg.label_names)

    estimator = PoseEstimator(cfg.pose)
    LOGGER.info("pose: %s", estimator.describe())

    # ---------------------------------------------------------------- MERL
    if args.merl:
        clips = discover_clips(args.merl)
        if not clips:
            LOGGER.error(
                "no clips found under %s. Expected videos/*.mp4 and labels/*_label.mat",
                args.merl,
            )
            return 1
        if args.split:
            tr, va, te = split_official(clips)
            chosen = {"train": tr, "val": va, "test": te, "trainval": tr + va}[args.split]
            LOGGER.info(
                "official split %s: %d of %d clips (subjects %s)",
                args.split, len(chosen), len(clips),
                sorted({c.subject for c in chosen})[:5],
            )
            clips = chosen
        if args.limit_clips:
            clips = clips[: args.limit_clips]
        LOGGER.info("found %d clips", len(clips))
        LOGGER.info("label frame counts (ground truth): %s", label_histogram(clips))

        store = extract_merl(
            clips, cfg, estimator=estimator, target_fps=fps,
            max_frames_per_clip=args.max_frames,
            drop_background=not args.keep_background,
        )

    # -------------------------------------------------------- single video
    elif args.video:
        label_idx = None
        if args.label is not None:
            label_idx = (
                int(args.label) if str(args.label).isdigit()
                else names.index(args.label)
            )
        store = extract_video(
            args.video, cfg, label=label_idx, estimator=estimator,
            target_fps=fps, max_frames=args.max_frames,
        )

    # ------------------------------------------------- directory of classes
    else:
        root = Path(args.video_dir)
        stores = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name not in names:
                LOGGER.warning("skipping %s: not one of %s", class_dir.name, names)
                continue
            label_idx = names.index(class_dir.name)
            videos = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            )
            LOGGER.info("%s -> label %d (%d videos)", class_dir.name, label_idx, len(videos))
            for video in videos:
                try:
                    stores.append(
                        extract_video(
                            video, cfg, label=label_idx,
                            clip_name=f"{class_dir.name}/{video.stem}",
                            estimator=estimator, target_fps=fps,
                            max_frames=args.max_frames,
                        )
                    )
                except RuntimeError as exc:
                    LOGGER.warning("  %s: %s", video.name, exc)
        if not stores:
            LOGGER.error("nothing extracted from %s", root)
            return 1
        store = merge_stores(stores)

    if args.min_valid is not None:
        store = filter_store(store, args.min_valid, cfg.sequence.conf_threshold)

    out_path = Path(args.out)
    store.save(out_path)
    _report(store, cfg, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
