"""Run customer action recognition on a video file, webcam, CSI camera or RTSP.

Examples::

    # recorded top-view clip, write an annotated video and a per-frame CSV
    python3 tools/run_recognition.py --source data/videos/shop.mp4 \
        --weights runs/action/stgcn/best.pt \
        --save-video runs/demo/shop_annotated.mp4 \
        --save-csv runs/demo/shop_actions.csv --no-display

    # USB webcam on a machine with a display
    python3 tools/run_recognition.py --source 0

    # Jetson CSI camera
    python3 tools/run_recognition.py --source 0 --csi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from car.config import load_config, parse_overrides  # noqa: E402
from car.pipeline.realtime import run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Skeleton-based customer action recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", default=None,
                        help="video path, camera index, or rtsp:// URL (default: config)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None, help="action model checkpoint")
    parser.add_argument("--pose-weights", default=None, help="override YOLO pose weights")
    parser.add_argument("--csi", action="store_true", help="source is a Jetson CSI camera")
    parser.add_argument("--no-display", action="store_true", help="headless; do not open a window")
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--save-summary", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--loop", action="store_true", help="replay a file forever")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. pose.conf=0.3 runtime.smooth_window=7")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    overrides = parse_overrides(args.set)
    if args.pose_weights:
        overrides["pose.weights"] = args.pose_weights

    config_path = args.config if args.config and Path(args.config).exists() else None
    cfg = load_config(config_path, overrides)

    summary = run(
        cfg,
        source=args.source,
        action_weights=args.weights,
        display=not args.no_display,
        save_video=args.save_video,
        save_csv=args.save_csv,
        save_summary=args.save_summary,
        max_frames=args.max_frames,
        loop=args.loop,
        csi=args.csi or None,
    )
    return 0 if summary.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
