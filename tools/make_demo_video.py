"""Render a synthetic top-view shopping clip for smoke-testing the pipeline.

This is a stand-in, not data: it animates articulated stick figures viewed from
above so the tracker, buffers and action models have something with real
temporal structure to chew on before a camera or the MERL dataset is available.
YOLO will not reliably detect these figures -- for a pose-estimation test use
real footage. What it *does* exercise end to end is
``skeleton -> normalise -> window -> train -> classify``, via
``tools/make_demo_skeletons.py``, which shares this motion model.

Usage::

    python3 tools/make_demo_video.py --out data/videos/demo_topview.mp4 \
        --seconds 40 --people 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.synthetic import ACTION_SEQUENCE, SyntheticShopper, draw_scene  # noqa: E402


def main(argv=None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description="Render a synthetic top-view clip")
    parser.add_argument("--out", default="data/videos/demo_topview.mp4")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--people", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shoppers = [
        SyntheticShopper(
            origin=(
                args.width * (0.3 + 0.4 * i / max(args.people - 1, 1)),
                args.height * 0.62,
            ),
            scale=args.height / 6.0,
            rng=np.random.default_rng(args.seed + i),
        )
        for i in range(args.people)
    ]

    total = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        print(f"could not open {out} for writing", file=sys.stderr)
        return 1

    for t in range(total):
        frame = draw_scene(args.width, args.height)
        for shopper in shoppers:
            kpts, action = shopper.step()
            shopper.draw(frame, kpts)
        writer.write(frame)

    writer.release()
    print(f"wrote {out} ({total} frames, {args.fps} fps, {args.people} people)")
    print("actions cycled:", ", ".join(ACTION_SEQUENCE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
