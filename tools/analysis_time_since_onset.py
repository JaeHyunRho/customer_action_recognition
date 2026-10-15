#!/usr/bin/env python3
"""Accuracy by time since the labelled action began (subject-split E1 checkpoints).

Reads the per-window predictions written by ``tools/subject_split.py predict``
for the six E1 checkpoints, measures for every test window how long the action
of its last detection had been running, and bins the windows::

    .venv/bin/python tools/analysis_time_since_onset.py
    -> runs/subject_split/analysis/time_since_onset.json, fig_time_since_onset.{pdf,png}

Elapsed time is ``(f_end - f_start) / fps`` on the source frames: ``f_end`` is
the frame of the window's last detection, ``f_start`` the first frame of the
uninterrupted run of the same ``frame_labels`` label that ends at ``f_end``,
and ``fps`` is read from the video. Accuracy is the per-window argmax only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.datasets.merl import discover_clips, video_frame_count  # noqa: E402
from car.datasets.sequence import SkeletonStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/subject_split/analysis"
STORE = ROOT / "data/processed/subject_split/custom_test.npz"
MODELS = (("stgcn", "ST-GCN"), ("lstm", "LSTM"))
SEEDS = (42, 43, 44)
EDGES = [0.0, 0.25, 0.55, 1.05, 2.05, 2.95, float("inf")]
LABELS = ["0–0.25", "0.25–0.55", "0.55–1.05", "1.05–2.05", "2.05–2.95", "≥2.95"]


def run_starts(labels: np.ndarray) -> np.ndarray:
    """First frame of the uninterrupted run each frame belongs to."""
    change = np.r_[True, labels[1:] != labels[:-1]]
    return np.maximum.accumulate(np.where(change, np.arange(len(labels)), 0))


def main() -> int:
    import cv2

    store = SkeletonStore.load(STORE)
    preds = {(m, s): np.load(OUT / f"predictions/e1_custom_{m}_s{s}.npz", allow_pickle=True)
             for m, _ in MODELS for s in SEEDS}
    ref = preds[("stgcn", 42)]
    for key, z in preds.items():
        if not (np.array_equal(z["rows"], ref["rows"]) and np.array_equal(z["y_true"], ref["y_true"])):
            raise SystemExit(f"STOP: window set of {key} differs from stgcn/42")

    clips = {c.clip_id: c for c in discover_clips(ROOT / "data/raw/merl")}
    starts, fps = {}, {}
    for name in np.unique(ref["clip"]):
        c = clips[str(name)]
        lab = c.frame_labels(video_frame_count(c.video_path))
        starts[str(name)] = (lab, run_starts(lab))
        cap = cv2.VideoCapture(str(c.video_path))
        fps[str(name)] = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    elapsed, mismatch = [], 0
    for name, f_end, y in zip(ref["clip"], ref["end_frame"], ref["y_true"]):
        lab, st = starts[str(name)]
        mismatch += int(lab[int(f_end)] != int(y))
        elapsed.append((int(f_end) - int(st[int(f_end)])) / fps[str(name)])
    elapsed = np.array(elapsed)
    if mismatch:
        raise SystemExit(f"STOP: {mismatch} windows whose label differs from frame_labels at f_end")
    rows = ref["rows"]
    single = (store.labels[rows] == store.labels[rows][:, :1]).all(axis=1)
    bucket = np.digitize(elapsed, EDGES[1:-1], right=False)   # [lo, hi) bins

    result = {
        "bins_seconds": [[lo, hi if np.isfinite(hi) else None] for lo, hi in zip(EDGES[:-1], EDGES[1:])],
        "bin_labels": LABELS,
        "windows": int(len(elapsed)),
        "fps_read_from_videos": sorted(set(fps.values())),
        "elapsed_rule": "(f_end - f_start) / fps; f_start = first frame of the uninterrupted frame_labels run ending at f_end",
        "accuracy": "per-window argmax; no majority vote, no confidence threshold",
        "checkpoints": {m: [f"runs/subject_split/action/e1_custom_{m}_s{s}/best.pt" for s in SEEDS] for m, _ in MODELS},
        "store": str(STORE.relative_to(ROOT)),
        "bins": [],
    }
    for b, name in enumerate(LABELS):
        sel = bucket == b
        entry = {"bin": name, "n": int(sel.sum()),
                 "single_action_share": round(float(single[sel].mean()), 4) if sel.any() else None}
        for m, _ in MODELS:
            accs = [float((preds[(m, s)]["y_pred"][sel] == preds[(m, s)]["y_true"][sel]).mean()) for s in SEEDS]
            entry[m] = {"seeds": dict(zip(map(str, SEEDS), [round(a, 4) for a in accs])),
                        "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs, ddof=1)), 4)}
        result["bins"].append(entry)
    result["std"] = "sample standard deviation over the 3 seeds (ddof=1)"
    result["overall"] = {m: [round(float((preds[(m, s)]["y_pred"] == preds[(m, s)]["y_true"]).mean()), 4) for s in SEEDS]
                         for m, _ in MODELS}
    (OUT / "time_since_onset.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    figure(result)
    for e in result["bins"]:
        print(f"{e['bin']:>10}  n={e['n']:5d}  ST-GCN {e['stgcn']['mean']:.4f}±{e['stgcn']['std']:.4f}  "
              f"LSTM {e['lstm']['mean']:.4f}±{e['lstm']['std']:.4f}  single {e['single_action_share']}")
    return 0


def figure(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"  # text inks, axis, grid
    colors = {"stgcn": "#2a78d6", "lstm": "#eb6834"}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": muted, "axes.linewidth": 0.6, "axes.labelcolor": ink2,
        "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": grid, "grid.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42,
    })
    bins = result["bins"]
    x = np.arange(len(bins))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.subplots_adjust(left=0.12, right=0.995, top=0.88, bottom=0.25)
    for j, (m, name) in enumerate(MODELS):
        means = [b[m]["mean"] for b in bins]
        stds = [b[m]["std"] for b in bins]
        pos = x + (j - 0.5) * (width + 0.03)          # small surface gap between the two bars
        ax.bar(pos, means, width, color=colors[m], label=name, zorder=2)
        ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=ink, elinewidth=0.7, capsize=1.6, zorder=3)
    ax.set_xticks(x, [f"{b['bin']}\nn = {b['n']:,}" for b in bins])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Time since the action began (s)", labelpad=4)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.17), handlelength=1.0, columnspacing=1.2)
    fig.savefig(OUT / "fig_time_since_onset.pdf")
    fig.savefig(OUT / "fig_time_since_onset.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
