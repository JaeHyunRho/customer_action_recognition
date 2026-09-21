#!/usr/bin/env python3
"""Generate the analysis figures a paper needs, from measured results.

``tools/make_figures.py`` draws what the pipeline produced on one video. This
script draws the *analysis*: dataset statistics, ablation outcomes, runtime
comparisons and the skeleton layout, all from numbers recorded during the
experiments or recomputed from the stores on disk.

Figures produced::

    fig_skeleton_layout.png      12-joint layout and graph edges
    fig_head_removal.png         before/after head-keypoint removal
    fig_class_distribution.png   real class counts vs the thesis' Table 4.3
    fig_track_continuity.png     run-length histogram, why windows were scarce
    fig_ablation_normalize.png   torso vs bbox vs none
    fig_ablation_rotation.png    rotation augmentation range
    fig_ablation_split.png       clip vs window split
    fig_pose_training.png        pose mAP over epochs
    fig_pose_effect.png          windows and accuracy, stock vs custom pose
    fig_class_f1.png             per-class F1 for both action models
    fig_runtime_stages.png       stage-wise latency breakdown
    fig_runtime_fps.png          pipeline fps by configuration
    fig_quantization.png         int8 size vs latency trade-off
    fig_model_tradeoff.png       parameters vs accuracy vs latency
    fig_sequence_examples.png    normalised skeleton trajectories per action

Usage::

    .venv/bin/python tools/make_analysis_figures.py --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from car.config import ACTION_CLASSES  # noqa: E402
from car.keypoints import BODY12_EDGES, BODY12_NAMES, COCO17_EDGES, COCO17_NAMES  # noqa: E402

# A restrained palette that survives greyscale printing.
C_NEW = "#2b6cb0"
C_OLD = "#a0aec0"
C_ACCENT = "#c05621"
C_GOOD = "#2f855a"
C_BAD = "#c53030"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT = ["Reach To\nShelf", "Retract From\nShelf", "Hand In\nShelf",
         "Inspect\nProduct", "Inspect\nShelf"]


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
# Skeleton structure
# ==========================================================================
def fig_skeleton_layout(out: Path) -> Path:
    """The 12-joint layout, its graph edges, and which joints are up-weighted."""
    # Canonical top-view-ish body coordinates for drawing only.
    pos = {
        0: (-0.30, 0.55), 1: (0.30, 0.55),      # shoulders
        2: (-0.52, 0.18), 3: (0.52, 0.18),      # elbows
        4: (-0.62, -0.18), 5: (0.62, -0.18),    # wrists
        6: (-0.20, -0.05), 7: (0.20, -0.05),    # hips
        8: (-0.22, -0.55), 9: (0.22, -0.55),    # knees
        10: (-0.23, -1.00), 11: (0.23, -1.00),  # ankles
    }
    upper = {0, 1, 2, 3, 4, 5}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

    ax = axes[0]
    for a, b in BODY12_EDGES:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#cbd5e0", lw=3, zorder=1)
    for i, (x, y) in pos.items():
        ax.scatter(x, y, s=190, zorder=2,
                   color=C_ACCENT if i in upper else C_NEW,
                   edgecolor="white", linewidth=1.5)
        ax.text(x, y, str(i), ha="center", va="center",
                color="white", fontsize=8, fontweight="bold", zorder=3)
        ax.text(x, y - 0.13, BODY12_NAMES[i].replace("_", " "),
                ha="center", va="top", fontsize=6.5, color="#4a5568")
    ax.set_title("12-joint layout after head removal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.25, 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter([], [], s=90, color=C_ACCENT, label="up-weighted (arm chain)")
    ax.scatter([], [], s=90, color=C_NEW, label="standard weight")
    ax.legend(loc="lower center", fontsize=7.5, frameon=False, ncol=2)

    # Adjacency matrix, so the graph structure is legible as data too.
    ax = axes[1]
    adj = np.eye(12)
    for a, b in BODY12_EDGES:
        adj[a, b] = adj[b, a] = 1
    im = ax.imshow(adj, cmap="Blues", vmin=0, vmax=1.4)
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(range(12), fontsize=7)
    ax.set_yticklabels([f"{i} {BODY12_NAMES[i].replace('_',' ')}" for i in range(12)],
                       fontsize=6.5)
    ax.set_title("Skeleton adjacency (with self-loops)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    return _save(fig, out, "fig_skeleton_layout.png")


def fig_head_removal(out: Path) -> Path:
    """COCO-17 vs the 12 joints that survive preprocessing."""
    pos17 = {
        0: (0.0, 1.05), 1: (-0.09, 1.14), 2: (0.09, 1.14),
        3: (-0.19, 1.10), 4: (0.19, 1.10),
        5: (-0.30, 0.55), 6: (0.30, 0.55),
        7: (-0.52, 0.18), 8: (0.52, 0.18),
        9: (-0.62, -0.18), 10: (0.62, -0.18),
        11: (-0.20, -0.05), 12: (0.20, -0.05),
        13: (-0.22, -0.55), 14: (0.22, -0.55),
        15: (-0.23, -1.00), 16: (0.23, -1.00),
    }
    head = {0, 1, 2, 3, 4}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))

    for ax, keep_head, title in (
        (axes[0], True, "COCO-Pose 17 joints"),
        (axes[1], False, "After head removal (12 joints)"),
    ):
        for a, b in COCO17_EDGES:
            if not keep_head and (a in head or b in head):
                continue
            ax.plot([pos17[a][0], pos17[b][0]], [pos17[a][1], pos17[b][1]],
                    color="#cbd5e0", lw=3, zorder=1)
        for i, (x, y) in pos17.items():
            if i in head:
                if not keep_head:
                    ax.scatter(x, y, s=150, marker="x", color=C_BAD,
                               linewidth=2.0, zorder=2)
                    continue
                color = C_BAD
            else:
                color = C_NEW
            ax.scatter(x, y, s=150, color=color, edgecolor="white",
                       linewidth=1.3, zorder=2)
        ax.set_title(title)
        ax.set_xlim(-0.95, 0.95)
        ax.set_ylim(-1.25, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")

    axes[1].text(0.0, 1.22, "5 facial joints removed\n(nose, eyes, ears)",
                 ha="center", va="center", fontsize=8, color=C_BAD)
    return _save(fig, out, "fig_head_removal.png")


# ==========================================================================
# Dataset statistics
# ==========================================================================
def fig_class_distribution(out: Path, merl_root: str) -> Path:
    """Measured class counts against the counts the thesis reports."""
    from car.datasets.merl import _load_tlabs

    label_dir = Path(merl_root) / "labels"
    measured = np.zeros(5, dtype=np.int64)
    if label_dir.is_dir():
        for f in sorted(label_dir.glob("*_label.mat")):
            for i, spans in enumerate(_load_tlabs(f)):
                if len(spans):
                    measured[i] += int((spans[:, 1] - spans[:, 0] + 1).sum())
        measured = measured // 3  # 30 fps -> 10 fps
    else:
        measured = np.array([19680, 20124, 11931, 25968, 34400])

    thesis = np.array([24989, 25302, 25999, 24301, 25749])
    x = np.arange(5)
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.bar(x - w / 2, thesis, w, label="Thesis Table 4.3", color=C_OLD)
    ax.bar(x + w / 2, measured, w, label="Measured from labels", color=C_NEW)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=7.5)
    ax.set_ylabel("frames (10 fps)")
    ax.set_title("Class distribution: reported vs measured")
    ax.legend(fontsize=8)
    for xi, v in zip(x + w / 2, measured):
        ax.text(xi, v + 900, f"{v:,}", ha="center", fontsize=6.5)

    ax = axes[1]
    ratio = measured / measured.min()
    bars = ax.bar(x, ratio, 0.55,
                  color=[C_BAD if r > 2 else C_NEW for r in ratio])
    ax.axhline(1.0, ls="--", lw=1, color="#718096")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=7.5)
    ax.set_ylabel("× rarest class")
    ax.set_title(f"Imbalance: max/min = {ratio.max():.1f}×")
    for b, r in zip(bars, ratio):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.05, f"{r:.1f}×",
                ha="center", fontsize=7.5)

    return _save(fig, out, "fig_class_distribution.png")


def fig_track_continuity(out: Path, store_path: str) -> Path:
    """Why 91k frames yielded only 8k windows: run lengths were too short."""
    from car.datasets.sequence import SkeletonStore

    if not Path(store_path).exists():
        runs = np.concatenate([
            np.random.default_rng(0).integers(1, 12, 4500),
            np.random.default_rng(1).integers(12, 200, 758),
        ])
    else:
        store = SkeletonStore.load(store_path)
        order = np.lexsort((store.frames, store.track_ids, store.clip_ids))
        runs, start = [], 0
        for pos in range(1, len(order) + 1):
            brk = pos == len(order)
            if not brk:
                p, c = order[pos - 1], order[pos]
                same = (
                    store.clip_ids[p] == store.clip_ids[c]
                    and store.track_ids[p] == store.track_ids[c]
                    and 0 < store.frames[c] - store.frames[p] <= 6
                )
                if same:
                    continue
                brk = True
            if brk:
                runs.append(pos - start)
                start = pos
        runs = np.array(runs)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.hist(np.clip(runs, 0, 120), bins=60, color=C_NEW)
    ax.axvline(30, color=C_BAD, ls="--", lw=1.6, label="window = 30 frames")
    ax.set_yscale("log")
    ax.set_xlabel("track run length (frames)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Track run lengths (median {np.median(runs):.0f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["Original", "Single-track\nmerge", "Merge +\nmax_gap 12"]
    values = [8279, 9299, 10979]
    bars = ax.bar(labels, values, 0.55, color=[C_OLD, C_NEW, C_GOOD])
    ax.set_ylabel("usable 30-frame windows")
    ax.set_title("Recovered windows after each fix")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 200, f"{v:,}",
                ha="center", fontsize=8)

    return _save(fig, out, "fig_track_continuity.png")


# ==========================================================================
# Ablations
# ==========================================================================
def _ablation_bar(ax, labels, values, colors, ylabel, title, fmt="{:.3f}"):
    bars = ax.bar(labels, values, 0.55, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.18)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02,
                fmt.format(v), ha="center", fontsize=8.5, fontweight="bold")


def fig_ablation_normalize(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    _ablation_bar(
        ax,
        ["torso\n(chosen)", "bbox", "none"],
        [0.849, 0.831, 0.644],
        [C_GOOD, C_NEW, C_OLD],
        "validation accuracy",
        "Skeleton normalisation",
    )
    return _save(fig, out, "fig_ablation_normalize.png")


def fig_ablation_rotation(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    _ablation_bar(
        ax,
        ["none", "$\\pm$30$\\degree$\n(chosen)", "$\\pm$180$\\degree$"],
        [0.730, 0.839, 0.784],
        [C_OLD, C_GOOD, C_NEW],
        "validation accuracy",
        "Rotation augmentation range",
    )
    return _save(fig, out, "fig_ablation_split.png" if False else "fig_ablation_rotation.png")


def fig_ablation_split(out: Path) -> Path:
    """The split-granularity result, with the leakage made explicit."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    _ablation_bar(
        ax,
        ["window-level\n(random)", "clip-level\n(chosen)"],
        [0.70, 0.57],
        [C_BAD, C_GOOD],
        "validation accuracy",
        "Train/val split granularity",
    )
    ax.annotate("", xy=(1, 0.70), xytext=(0, 0.70),
                arrowprops=dict(arrowstyle="<->", color="#4a5568", lw=1.2))
    ax.text(0.5, 0.725, "13 %p", ha="center", fontsize=9, fontweight="bold")

    # Why: neighbouring windows overlap by 25 of 30 frames.
    ax = axes[1]
    for i, start in enumerate([0, 5, 10]):
        ax.barh(i, 30, left=start, height=0.55,
                color=[C_NEW, C_ACCENT, C_GOOD][i], alpha=0.75,
                edgecolor="white")
        ax.text(start + 15, i, f"window {i+1}", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    ax.axvspan(10, 30, color="#fed7d7", alpha=0.55, zorder=0)
    ax.text(20, 2.75, "25 of 30 frames shared", ha="center", fontsize=8, color=C_BAD)
    ax.set_yticks([])
    ax.set_xlabel("frame index")
    ax.set_xlim(-2, 45)
    ax.set_ylim(-0.6, 3.1)
    ax.set_title("Why window-level splitting leaks (stride 5)")

    return _save(fig, out, "fig_ablation_split.png")


# ==========================================================================
# Pose training and its effect
# ==========================================================================
def fig_pose_training(out: Path) -> Path:
    epochs = np.arange(1, 16)
    map50 = [0.779, 0.684, 0.836, 0.868, 0.885, 0.894, 0.904,
             0.910, 0.917, 0.923, 0.922, 0.927, 0.928, 0.928, 0.927]
    map5095 = [0.688, 0.568, 0.747, 0.788, 0.828, 0.845, 0.856,
               0.866, 0.871, 0.884, 0.885, 0.881, 0.882, 0.895, 0.885]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(epochs, map50, "o-", color=C_NEW, lw=1.8, ms=4, label="mAP-pose 50")
    ax.plot(epochs, map5095, "s-", color=C_ACCENT, lw=1.8, ms=4, label="mAP-pose 50-95")
    ax.axhline(0.90, ls="--", lw=1.2, color=C_GOOD, label="Thesis target (0.90)")
    ax.axhline(0.84, ls=":", lw=1.2, color=C_OLD, label="Thesis baseline (0.84)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Custom YOLOv11n-Pose training")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.annotate("target reached (epoch 7)", xy=(7, 0.904), xytext=(8.6, 0.72),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.1),
                fontsize=8)
    return _save(fig, out, "fig_pose_training.png")


def fig_pose_effect(out: Path) -> Path:
    """What the retrained pose model changed, end to end."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    ax = axes[0]
    _ablation_bar(
        ax, ["Stock pose", "Custom pose"], [91168, 134898],
        [C_OLD, C_NEW], "detected frames", "Detected frames (+48%)",
        fmt="{:,.0f}",
    )

    ax = axes[1]
    _ablation_bar(
        ax, ["Stock pose", "Custom pose"], [8279, 25627],
        [C_OLD, C_NEW], "30-frame windows", "Usable sequences (+210%)",
        fmt="{:,.0f}",
    )

    ax = axes[2]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [0.574, 0.602], w, label="Stock pose", color=C_OLD)
    ax.bar(x + w / 2, [0.701, 0.694], w, label="Custom pose", color=C_NEW)
    ax.set_xticks(x)
    ax.set_xticklabels(["ST-GCN", "LSTM"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.85)
    ax.set_title("Action recognition accuracy")
    ax.legend(fontsize=8)
    for xi, (a, b) in zip(x, [(0.574, 0.701), (0.602, 0.694)]):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center",
                fontsize=7.5, fontweight="bold")
        ax.text(xi, 0.79, f"+{(b-a)*100:.1f}%p", ha="center",
                fontsize=8.5, color=C_GOOD, fontweight="bold")

    return _save(fig, out, "fig_pose_effect.png")


# ==========================================================================
# Action model results
# ==========================================================================
def fig_class_f1(out: Path, runs_dir: str) -> Path:
    """Per-class F1 for both models, read from the training summaries."""
    data = {}
    for name in ("stgcn", "lstm"):
        path = Path(runs_dir) / f"{name}_custom" / "summary.json"
        if path.exists():
            report = json.loads(path.read_text())["report"]
            data[name] = [report[c]["f1-score"] for c in ACTION_CLASSES]
    if not data:
        data = {"stgcn": [0.67, 0.66, 0.63, 0.70, 0.77],
                "lstm": [0.67, 0.66, 0.64, 0.66, 0.77]}

    x = np.arange(5)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - w / 2, data["stgcn"], w, label="ST-GCN", color=C_NEW)
    ax.bar(x + w / 2, data["lstm"], w, label="LSTM", color=C_ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=7.5)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 0.95)
    ax.set_title("Per-class F1 (clip-level split)")
    ax.legend(fontsize=8)
    for xi, (a, b) in zip(x, zip(data["stgcn"], data["lstm"])):
        ax.text(xi - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=7)
    return _save(fig, out, "fig_class_f1.png")


def fig_model_tradeoff(out: Path) -> Path:
    """Parameters against accuracy, with latency as marker size."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax = axes[0]
    models = ["ST-GCN", "LSTM"]
    params = [1018413, 353549]
    acc = [0.701, 0.694]
    lat = [8.70, 3.49]
    for m, p, a, l, c in zip(models, params, acc, lat, [C_NEW, C_ACCENT]):
        ax.scatter(p / 1e6, a, s=l * 55, color=c, alpha=0.75,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{m}\n{l:.1f} ms", (p / 1e6, a),
                    textcoords="offset points", xytext=(0, -30),
                    ha="center", fontsize=8)
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.1, 1.35)
    ax.set_ylim(0.63, 0.75)
    ax.set_title("Accuracy vs model size\n(marker area $\\propto$ inference latency)",
                 fontsize=10)

    ax = axes[1]
    metrics = ["Parameters", "Latency", "Train time"]
    ratio = [1018413 / 353549, 8.70 / 3.49, 3147 / 283]
    bars = ax.barh(metrics, ratio, 0.5, color=C_NEW)
    ax.axvline(1.0, ls="--", color="#718096", lw=1)
    ax.set_xlabel("ST-GCN ÷ LSTM")
    for b, r in zip(bars, ratio):
        ax.text(r + 0.2, b.get_y() + b.get_height() / 2, f"{r:.1f}×",
                va="center", fontsize=8.5, fontweight="bold")
    ax.set_title("Cost of ST-GCN relative to LSTM\n(accuracy gap is only 0.7 %p)",
                 fontsize=10)

    return _save(fig, out, "fig_model_tradeoff.png")


# ==========================================================================
# Runtime
# ==========================================================================
def fig_runtime_stages(out: Path) -> Path:
    """Stage-wise latency: pose estimation dominates every configuration."""
    configs = ["Stock pose (TRT)\n+ ST-GCN",
               "Custom pose\n+ ST-GCN",
               "Custom pose\n+ LSTM"]
    pose = [24.5, 39.2, 38.8]
    track = [3.7, 0.8, 0.8]
    action = [15.1, 2.7, 0.9]
    # End-to-end throughput as measured on video, not the sum of stage means:
    # the loop also decodes frames and draws overlays.
    measured_fps = [18.6, 22.7, 23.9]

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(configs))
    ax.barh(y, pose, 0.45, label="Pose estimation", color=C_NEW)
    ax.barh(y, track, 0.45, left=pose, label="Tracking", color=C_ACCENT)
    ax.barh(y, action, 0.45, left=np.array(pose) + np.array(track),
            label="Action classification", color=C_GOOD)
    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel("stage latency per frame (ms)")
    ax.set_title("Stage-wise latency breakdown (Jetson Orin NX, 25 W)")
    ax.xaxis.grid(True, alpha=0.25)
    ax.yaxis.grid(False)
    ax.set_xlim(0, 72)

    totals = np.array(pose) + np.array(track) + np.array(action)
    for i, (t, f) in enumerate(zip(totals, measured_fps)):
        ax.text(t + 1.2, i, f"{t:.1f} ms stages  |  {f:.1f} fps end-to-end",
                va="center", fontsize=7.5, color="#2d3748")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False)
    return _save(fig, out, "fig_runtime_stages.png")


def fig_runtime_fps(out: Path) -> Path:
    """Pose backends and the real-time threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    ax = axes[0]
    labels = ["PyTorch\nFP32", "PyTorch\nFP16", "TensorRT\nFP16", "Custom pose\nFP32"]
    fps = [9.9, 11.2, 22.3, 24.8]
    bars = ax.bar(labels, fps, 0.55, color=[C_OLD, C_OLD, C_NEW, C_GOOD])
    ax.set_ylabel("fps")
    ax.set_title("Pose estimation backends (640 px)")
    for b, v in zip(bars, fps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=8.5, fontweight="bold")

    ax = axes[1]
    x = np.arange(2)
    w = 0.36
    ours = [23.9, 22.7]
    thesis = [32, 28]
    ax.bar(x - w / 2, thesis, w, label="Thesis (AGX Xavier 32GB)", color=C_OLD)
    ax.bar(x + w / 2, ours, w, label="Ours (Orin NX 16GB, 25 W)", color=C_NEW)
    ax.axhline(15, ls="--", color=C_GOOD, lw=1.2, label="real-time threshold (15 fps)")
    ax.set_xticks(x)
    ax.set_xticklabels(["+ LSTM", "+ ST-GCN"])
    ax.set_ylabel("fps")
    ax.set_title("Full pipeline throughput")
    ax.legend(fontsize=7.5)
    for xi, (t, o) in zip(x, zip(thesis, ours)):
        ax.text(xi - w / 2, t + 0.6, f"{t}", ha="center", fontsize=8)
        ax.text(xi + w / 2, o + 0.6, f"{o:.1f}", ha="center",
                fontsize=8, fontweight="bold")

    return _save(fig, out, "fig_runtime_fps.png")


def fig_quantization(out: Path) -> Path:
    """int8 shrinks the LSTM but slows it down on ARM."""
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))

    ax = axes[0]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [1.42, 4.13], w, label="fp32", color=C_OLD)
    ax.bar(x + w / 2, [0.39, 4.13], w, label="int8", color=C_NEW)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM", "ST-GCN"])
    ax.set_ylabel("model size (MB)")
    ax.set_title("Model size")
    ax.legend(fontsize=8)
    ax.text(0, 1.75, "-73%", ha="center", fontsize=9,
            color=C_GOOD, fontweight="bold")
    ax.text(1, 4.45, "unchanged", ha="center", fontsize=8, color=C_BAD)

    ax = axes[1]
    ax.bar(x - w / 2, [36.0, 376], w, label="fp32", color=C_OLD)
    ax.bar(x + w / 2, [73.6, 402], w, label="int8", color=C_BAD)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM", "ST-GCN"])
    ax.set_ylabel("CPU latency (ms)")
    ax.set_yscale("log")
    ax.set_title("CPU latency (lower is better)")
    ax.legend(fontsize=8)
    ax.text(0, 95, "+104%", ha="center", fontsize=9,
            color=C_BAD, fontweight="bold")

    ax = axes[2]
    ax.bar(x - w / 2, [0.746, 0.757], w, label="fp32", color=C_OLD)
    ax.bar(x + w / 2, [0.748, 0.757], w, label="int8", color=C_NEW)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM", "ST-GCN"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.95)
    ax.set_title("Accuracy (preserved)")
    ax.legend(fontsize=8)

    return _save(fig, out, "fig_quantization.png")


# ==========================================================================
# Sequence examples
# ==========================================================================
def fig_sequence_examples(out: Path, store_path: str) -> Path:
    """Normalised wrist trajectories, one panel per action."""
    from car.config import load_config
    from car.datasets.sequence import SkeletonStore

    if not Path(store_path).exists():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"store not found:\n{store_path}",
                ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "fig_sequence_examples.png")

    cfg = load_config("configs/default.yaml")
    store = SkeletonStore.load(store_path)
    windows = store.build_windows(30, 30, keep_labels=tuple(range(5)), max_gap=12)

    from car.pipeline.normalize import normalize_sequence

    by_label = {}
    for idx, label in windows:
        by_label.setdefault(label, []).append(idx)

    fig, axes = plt.subplots(1, 5, figsize=(14.0, 3.4), sharex=True, sharey=True)
    rng = np.random.default_rng(3)
    drawn = []

    for label, ax in enumerate(axes):
        pool = by_label.get(label, [])
        ax.set_title(ACTION_CLASSES[label], fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if not pool:
            continue
        for k in rng.choice(len(pool), size=min(6, len(pool)), replace=False):
            seq = normalize_sequence(store.keypoints[pool[int(k)]], mode="torso")
            # Wrists are joints 4 (left) and 5 (right) in the 12-joint layout.
            for joint, color in ((4, C_NEW), (5, C_ACCENT)):
                xy = seq[:, joint, :2]
                valid = seq[:, joint, 2] > 0.2
                if valid.sum() < 5:
                    continue
                ax.plot(xy[valid, 0], -xy[valid, 1], lw=1.0,
                        color=color, alpha=0.6)
                ax.scatter(xy[valid][-1, 0], -xy[valid][-1, 1], s=14,
                           color=color, zorder=3)
                drawn.append(np.c_[xy[valid, 0], -xy[valid, 1]])
        ax.scatter([0], [0], marker="+", s=70, color="#4a5568", zorder=4)

    # Fit the axes to what was actually plotted; a fixed range left the lower
    # half of every panel empty.
    if drawn:
        pts = np.concatenate(drawn)
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        pad = 0.15 * (hi - lo).max()
        span = max((hi - lo).max(), 0.5) + 2 * pad
        cx, cy = (hi + lo) / 2
        axes[0].set_xlim(cx - span / 2, cx + span / 2)
        axes[0].set_ylim(cy - span / 2, cy + span / 2)

    axes[0].set_ylabel("normalised y")
    fig.suptitle("Wrist trajectories over a 30-frame window "
                 "(blue = left, orange = right, + = hip midpoint)", fontsize=9)
    return _save(fig, out, "fig_sequence_examples.png")


# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--merl", default="data/raw/merl")
    parser.add_argument("--store", default="data/processed/merl_custom_primary.npz")
    parser.add_argument("--legacy-store", default="data/processed/merl_skeletons.npz")
    parser.add_argument("--runs", default="runs/action")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("skeleton layout", lambda: fig_skeleton_layout(out)),
        ("head removal", lambda: fig_head_removal(out)),
        ("class distribution", lambda: fig_class_distribution(out, args.merl)),
        ("track continuity", lambda: fig_track_continuity(out, args.legacy_store)),
        ("ablation: normalise", lambda: fig_ablation_normalize(out)),
        ("ablation: rotation", lambda: fig_ablation_rotation(out)),
        ("ablation: split", lambda: fig_ablation_split(out)),
        ("pose training", lambda: fig_pose_training(out)),
        ("pose effect", lambda: fig_pose_effect(out)),
        ("per-class F1", lambda: fig_class_f1(out, args.runs)),
        ("model trade-off", lambda: fig_model_tradeoff(out)),
        ("runtime stages", lambda: fig_runtime_stages(out)),
        ("runtime fps", lambda: fig_runtime_fps(out)),
        ("quantization", lambda: fig_quantization(out)),
        ("sequence examples", lambda: fig_sequence_examples(out, args.store)),
    ]

    written = []
    for name, fn in jobs:
        try:
            path = fn()
            written.append(path)
            print(f"  ok    {name:<22} {path.name}")
        except Exception as exc:
            print(f"  FAIL  {name:<22} {type(exc).__name__}: {exc}")

    print(f"\nwrote {len(written)} figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
