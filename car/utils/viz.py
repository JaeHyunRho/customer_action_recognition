"""Drawing helpers for skeletons, tracks and action labels."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..keypoints import skeleton_for_plot

# A stable, colour-blind-safe palette; index by track id.
_TRACK_COLORS = (
    (255, 176, 0), (0, 178, 238), (135, 206, 84), (255, 105, 97),
    (186, 133, 255), (255, 214, 0), (0, 210, 190), (244, 130, 200),
)
_LIMB_COLOR = (0, 255, 255)
_JOINT_COLOR = (0, 0, 255)


def track_color(track_id: int) -> Tuple[int, int, int]:
    return _TRACK_COLORS[int(track_id) % len(_TRACK_COLORS)]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.2,
    color: Optional[Tuple[int, int, int]] = None,
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """Draw one skeleton in place. ``keypoints`` is ``(V, 3)`` in pixels."""
    import cv2

    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[0] not in (12, 17):
        return frame

    edges = skeleton_for_plot(head_removal=kpts.shape[0] == 12)
    limb = color or _LIMB_COLOR
    for a, b in edges:
        if kpts[a, 2] < conf_threshold or kpts[b, 2] < conf_threshold:
            continue
        pa = (int(kpts[a, 0]), int(kpts[a, 1]))
        pb = (int(kpts[b, 0]), int(kpts[b, 1]))
        cv2.line(frame, pa, pb, limb, thickness, cv2.LINE_AA)

    for x, y, s in kpts:
        if s < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, _JOINT_COLOR, -1, cv2.LINE_AA)
    return frame


def draw_track(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    label: Optional[str] = None,
    score: Optional[float] = None,
    keypoints: Optional[np.ndarray] = None,
    conf_threshold: float = 0.2,
) -> np.ndarray:
    """Box, id and action caption for one tracked person."""
    import cv2

    color = track_color(track_id)
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if keypoints is not None:
        draw_skeleton(frame, keypoints, conf_threshold, color=color)

    caption = f"ID {track_id}"
    if label:
        caption += f" | {label}"
        if score is not None:
            caption += f" {score:.2f}"

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), base = cv2.getTextSize(caption, font, scale, thick)
    ty = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(frame, caption, (x1 + 3, ty - 2), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (10, 24),
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Translucent stats panel in the top-left corner."""
    import cv2

    if not lines:
        return frame
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    widths = [cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines]
    box_w = max(widths) + 16
    box_h = 22 * len(lines) + 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (origin[0] - 8, origin[1] - 18),
        (origin[0] - 8 + box_w, origin[1] - 18 + box_h), (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (origin[0], origin[1] + 22 * i),
            font, scale, color, thick, cv2.LINE_AA,
        )
    return frame


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: str,
    title: str = "Confusion Matrix for Action Recognition",
    normalize: bool = False,
) -> str:
    """Save a confusion matrix figure in the style of the thesis' Fig. 4.2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.asarray(matrix, dtype=np.float64)
    display = mat.copy()
    if normalize:
        row = display.sum(axis=1, keepdims=True)
        display = np.divide(display, np.clip(row, 1e-9, None))

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    im = ax.imshow(display, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = display.max() / 2.0 if display.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(mat[i, j])}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8,
                color="white" if display[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_history(history: dict, path: str, title: str = "") -> str:
    """Accuracy and loss curves, mirroring the thesis' Fig. 4.1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

    axes[0].plot(history.get("train_acc", []), label="Train Accuracy")
    axes[0].plot(history.get("val_acc", []), label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.get("train_loss", []), label="Train Loss")
    axes[1].plot(history.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
