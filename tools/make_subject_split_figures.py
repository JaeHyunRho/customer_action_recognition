#!/usr/bin/env python3
"""Figures for the subject-split protocol, drawn from the run artefacts.

    .venv/bin/python tools/subject_split.py report            # refresh eval/report.json first
    .venv/bin/python tools/make_subject_split_figures.py      # -> figures/subject_split/*.png

fig_ss_pose_comparison   coverage, window accuracy and end-to-end accuracy of the three pose models
fig_ss_intersection      the three pose models on the frames all of them detected
fig_ss_e1_confusion      test confusion matrices of ST-GCN and LSTM on Custom skeletons
fig_ss_subject_accuracy  accuracy per test subject, the unit the confidence intervals resample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/subject_split"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # fixed order: never more than three series
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POSES = [("custom", "Custom nano", BLUE), ("teacher_x", "Teacher (yolo11x)", ORANGE), ("coco_n", "COCO nano", AQUA)]
SHORT = ["Reach", "Retract", "Hand in\nshelf", "Inspect\nproduct", "Inspect\nshelf"]

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 200, "figure.facecolor": "white",
})


def vbar(ax, x, top, width, color, r_pt=2.5):
    """Column with a rounded data end and a square foot on the baseline."""
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((1, 1))
    sx, sy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
    r = r_pt * ax.figure.dpi / 72
    rx, ry = min(r / sx, width / 2), min(r / sy, abs(top))
    x0, x1 = x - width / 2, x + width / 2
    verts = [(x0, 0), (x0, top - ry), (x0, top), (x0 + rx, top), (x1 - rx, top), (x1, top),
             (x1, top - ry), (x1, 0), (x0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
             MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=2))


def fig_pose_comparison(report, out):
    full = {"custom": "e1_custom", "teacher_x": "e2full_teacher_x", "coco_n": "e2full_coco_n"}
    panels = [
        ("Ground-truth end frames\nthat can be scored", lambda g: g["end2end"]["coverage"]),
        ("Window accuracy\n(own windows)", lambda g: g["test"]["accuracy_mean"]),
        ("End-to-end accuracy\n(missed detections count as errors)", lambda g: g["end2end"]["accuracy_mean"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.08, wspace=0.08)
    for ax, (title, value) in zip(axes, panels):
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y")
        ax.set_xticks([])
        for i, (key, _, color) in enumerate(POSES):
            g = report["groups"].get(f"{full[key]}_stgcn")
            if g is None or "end2end" not in g:
                continue
            v = value(g)
            vbar(ax, i, v, 0.5, color)
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left", pad=6, fontsize=9.5)
    axes[0].set_ylabel("Share / accuracy (ST-GCN, mean of 3 seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for _, _, c in POSES]
    fig.legend(handles, [n for _, n, _ in POSES], loc="upper left", ncol=3, bbox_to_anchor=(0.06, 1.0),
               handlelength=1.1, handleheight=0.9, columnspacing=1.6)
    fig.savefig(out / "fig_ss_pose_comparison.png")
    plt.close(fig)


def fig_intersection(report, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.12)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.45, 0.72)
    ax.grid(axis="y")
    ax.set_xticks(range(3), [n for _, n, _ in POSES])
    ax.tick_params(axis="x", length=0)
    for j, (model, label, color) in enumerate((("stgcn", "ST-GCN", BLUE), ("lstm", "LSTM", ORANGE))):
        for i, (key, _, _) in enumerate(POSES):
            g = report["groups"].get(f"e2isect_{key}_{model}")
            if g is None:
                continue
            x = i + (j - 0.5) * 0.36
            seeds = [h["accuracy"] for h in g["test"]["seeds"]]
            ax.plot([x] * len(seeds), seeds, "o", ms=4.5, mfc="white", mec=color, mew=1.2, zorder=2)
            ax.plot(x, np.mean(seeds), "o", ms=8.5, mfc=color, mec="white", mew=1.4, zorder=3,
                    label=label if i == 0 else None)
            ax.text(x + (0.09 if j else -0.09), np.mean(seeds), f"{np.mean(seeds):.3f}", va="center",
                    ha="left" if j else "right", fontsize=8.5, color=INK)
    ax.set_ylabel("Test accuracy on shared frames")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.17), handletextpad=0.3, columnspacing=1.4)
    ax.set_title("Same frames for every pose model (2,569 test windows)", loc="left", pad=8, fontsize=9.5)
    fig.savefig(out / "fig_ss_intersection.png")
    plt.close(fig)


def fig_confusion(out):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.17, wspace=0.3)
    for ax, (title, model) in zip(axes, (("ST-GCN", "stgcn"), ("LSTM", "lstm"))):
        cms, accs = [], []
        for seed in (42, 43, 44):
            s = json.loads((RUNS / f"action/e1_custom_{model}_s{seed}/summary.json").read_text())
            cms.append(s["test_confusion_matrix"])
            accs.append(s["test_accuracy"])
        cm = np.sum(cms, 0).astype(float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.45 else INK)
        ax.set_xticks(range(5), SHORT)
        ax.set_yticks(range(5), SHORT)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 5), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{title}: test accuracy {np.mean(accs):.3f} (3 seeds)", loc="left", pad=6)
        ax.set_xlabel("Predicted action")
        ax.set_ylabel("True action")
    fig.savefig(out / "fig_ss_e1_confusion.png")
    plt.close(fig)


def fig_subject_accuracy(report, out):
    a, b = report["groups"]["e1_custom_stgcn"]["test"], report["groups"]["e1_custom_lstm"]["test"]
    subjects = a["subjects"]
    sa, sb = np.array(a["subject_accuracy_seed_mean"]), np.array(b["subject_accuracy_seed_mean"])
    order = np.argsort(-sa)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.17)
    xs = np.arange(len(order))
    dodge = 0.13   # side by side, so a near-tie does not hide one of the two marks
    for k, i in enumerate(order):
        ax.plot([k - dodge, k + dodge], [sa[i], sb[i]], color=MUTED, lw=1.0, zorder=1)
    ax.plot(xs - dodge, sa[order], "o", ms=7.5, mfc=BLUE, mec="white", mew=1.3, zorder=3, label="ST-GCN")
    ax.plot(xs + dodge, sb[order], "o", ms=7.5, mfc=ORANGE, mec="white", mew=1.3, zorder=2, label="LSTM")
    ax.axhline(sa.mean(), color=INK2, lw=0.8, zorder=0)
    ax.text(len(order) - 0.45, sa.mean() + 0.006, f"ST-GCN subject mean {sa.mean():.3f}", ha="right",
            va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)), [subjects[i] for i in order])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.55, 0.85)
    ax.grid(axis="y")
    ax.set_xlabel("Test subject (sorted by ST-GCN accuracy)")
    ax.set_ylabel("Accuracy (mean of 3 seeds)")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.17), handletextpad=0.3, columnspacing=1.4)
    fig.savefig(out / "fig_ss_subject_accuracy.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "figures/subject_split"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUNS / "eval/report.json").read_text())
    fig_pose_comparison(report, out)
    fig_intersection(report, out)
    fig_confusion(out)
    fig_subject_accuracy(report, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
