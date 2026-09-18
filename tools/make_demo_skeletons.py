#!/usr/bin/env python3
"""Generate a synthetic labelled skeleton store so the pipeline can be run today.

This produces a test fixture, not training data. Use it to verify that
extraction, windowing, training, export and live inference all work before the
MERL Shopping dataset or a camera is available. Accuracy measured on it
reflects the synthetic motion model, not real shoppers.

Usage::

    python3 tools/make_demo_skeletons.py --out data/processed/demo_skeletons.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import ACTION_CLASSES, load_config  # noqa: E402
from car.synthetic import generate_store  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic skeleton data")
    parser.add_argument("--out", default="data/processed/demo_skeletons.npz")
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--people", type=int, default=1)
    parser.add_argument("--hold", type=int, default=40,
                        help="frames spent per action before switching")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    store = generate_store(
        num_clips=args.clips,
        frames_per_clip=args.frames,
        people_per_clip=args.people,
        seed=args.seed,
        hold=args.hold,
    )
    out = Path(args.out)
    store.save(out)

    counts = np.bincount(store.labels, minlength=len(ACTION_CLASSES))
    print(f"wrote {out}: {len(store)} frames, {len(store.clip_names)} clips")
    for name, count in zip(ACTION_CLASSES, counts):
        print(f"  {name:<22} {int(count):>7}")

    cfg = load_config(args.config if Path(args.config).exists() else None)
    windows = store.build_windows(cfg.sequence.window, cfg.sequence.train_stride)
    print(f"  -> {len(windows)} windows of {cfg.sequence.window} at stride {cfg.sequence.train_stride}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
