#!/usr/bin/env python3
"""Generate paper-ready figures from a processed video.

Produces the visual evidence a write-up needs:

* per-action montage   one representative frame per recognised action
* timeline             predicted action over time against ground truth
* pose comparison      stock vs custom YOLOv11n-Pose on the same frame
* confusion matrices   already written by training, copied here for convenience

Usage::

    .venv/bin/python tools/make_figures.py \\
        --video data/raw/merl/videos/1_2_crop.mp4 \\
        --csv figures/1_2_actions.csv \\
        --out figures
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import ACTION_CLASSES  # noqa: E402
from car.datasets.merl import discover_clips  # noqa: E402


def load_predictions(csv_path: Path) -> list:
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("action"):
                rows.append(
                    {
                        "frame": int(row["frame"]),
                        "track": int(row["track_id"]),
                        "action": row["action"],
                        "score": float(row["score"]),
                        "box": [float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                    }
                )
    return rows


def montage(video: Path, rows: list, out: Path, per_action: int = 1) -> Path:
    """One representative frame per action, tiled into a single image."""
    import cv2

    from car.utils.viz import draw_track

    # Pick the highest-confidence frame for each action.
    best = {}
    for r in rows:
        cur = best.get(r["action"])
        if cur is None or r["score"] > cur["score"]:
            best[r["action"]] = r

    order = [a for a in ACTION_CLASSES if a in best]
    if not order:
        raise RuntimeError("no actions were predicted; nothing to montage")

    cap = cv2.VideoCapture(str(video))
    tiles = []
    for action in order:
        r = best[action]
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        draw_track(frame, np.array(r["box"]), r["track"], label=action, score=r["score"])
        cv2.putText(
            frame, f"frame {r['frame']}", (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(cv2.resize(frame, (460, 340)))
    cap.release()

    cols = min(3, len(tiles))
    rows_n = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows_n * 340, cols * 460, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * 340 : (r + 1) * 340, c * 460 : (c + 1) * 460] = tile

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def timeline(rows: list, out: Path, ground_truth=None, fps: float = 30.0) -> Path:
    """Predicted action over time, optionally against the label file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ACTION_CLASSES)
    idx = {n: i for i, n in enumerate(names)}

    frames = np.array([r["frame"] for r in rows])
    preds = np.array([idx.get(r["action"], -1) for r in rows])
    scores = np.array([r["score"] for r in rows])
    seconds = frames / max(fps, 1)

    have_gt = ground_truth is not None
    fig, axes = plt.subplots(
        3 if have_gt else 2, 1,
        figsize=(11, 6.0 if have_gt else 4.4),
        dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1] if have_gt else [2, 1]},
    )

    ax = axes[0]
    ax.scatter(seconds, preds, c=preds, cmap="tab10", s=8, vmin=0, vmax=len(names) - 1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.3)
    ax.set_title("Recognised customer action over time")

    if have_gt:
        gt_frames = np.arange(len(ground_truth))
        valid = ground_truth < len(names)
        ax = axes[1]
        ax.scatter(
            gt_frames[valid] / max(fps, 1), ground_truth[valid],
            c=ground_truth[valid], cmap="tab10", s=2, vmin=0, vmax=len(names) - 1,
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylabel("Ground truth")
        ax.grid(alpha=0.3)

    ax = axes[-1]
    ax.plot(seconds, scores, lw=0.8, color="#3d7dca")
    ax.axhline(0.4, ls="--", lw=0.8, color="#cc4444", label="min confidence")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def pose_comparison(video: Path, frame_idx: int, out: Path) -> Path:
    """Stock vs custom pose model on the same frame, side by side."""
    import cv2

    from car.config import load_config
    from car.pose.estimator import PoseEstimator
    from car.utils.viz import draw_skeleton

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {video}")

    panels = []
    for weights, title in (
        ("weights/yolo11n-pose.pt", "YOLOv11n-Pose (COCO)"),
        ("weights/custom_yolo11n_pose.pt", "Custom YOLOv11n-Pose (top-view)"),
    ):
        if not Path(weights).exists():
            continue
        cfg = load_config("configs/default.yaml", {"pose.weights": weights})
        est = PoseEstimator(cfg.pose)
        result = est.infer(frame)
        panel = frame.copy()
        conf = 0.0
        for kpts, box in zip(result.keypoints, result.boxes):
            draw_skeleton(panel, kpts, conf_threshold=0.2, radius=4, thickness=3)
            cv2.rectangle(
                panel, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 220, 0), 2,
            )
            conf = max(conf, float(kpts[:, 2].mean()))
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (30, 30, 30), -1)
        cv2.putText(
            panel, f"{title}   mean joint conf {conf:.3f}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        panels.append(panel)
        del est

    if not panels:
        raise RuntimeError("no pose weights available for comparison")

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(panels) if len(panels) > 1 else panels[0])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True, help="per-frame predictions from run_recognition")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--merl", default="data/raw/merl", help="for ground-truth labels")
    parser.add_argument("--pose-frame", type=int, default=600)
    args = parser.parse_args(argv)

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_predictions(Path(args.csv))
    print(f"loaded {len(rows)} labelled predictions")

    # Ground truth, when the clip is part of MERL.
    gt = None
    try:
        stem = video.stem.replace("_crop", "")
        for clip in discover_clips(args.merl):
            if clip.clip_id == stem:
                import cv2

                cap = cv2.VideoCapture(str(video))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                gt = clip.frame_labels(total)
                print(f"ground truth loaded for clip {stem}")
                break
    except Exception as exc:
        print(f"no ground truth ({exc})")

    produced = []
    produced.append(montage(video, rows, out / "fig_actions_montage.png"))
    produced.append(timeline(rows, out / "fig_timeline.png", ground_truth=gt))
    try:
        produced.append(pose_comparison(video, args.pose_frame, out / "fig_pose_comparison.png"))
    except Exception as exc:
        print(f"pose comparison skipped: {exc}")

    for name, src in (
        ("fig_confusion_stgcn.png", "runs/action/stgcn_custom/confusion_matrix_norm.png"),
        ("fig_confusion_lstm.png", "runs/action/lstm_custom/confusion_matrix_norm.png"),
        ("fig_history_stgcn.png", "runs/action/stgcn_custom/history.png"),
        ("fig_history_lstm.png", "runs/action/lstm_custom/history.png"),
    ):
        if Path(src).exists():
            shutil.copy2(src, out / name)
            produced.append(out / name)

    print("\nwrote:")
    for p in produced:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
