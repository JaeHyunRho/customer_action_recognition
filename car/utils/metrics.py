"""Classification metrics and on-device benchmarking helpers.

Mirrors the numbers the thesis reports: accuracy, macro precision/recall,
mean F1 (Table 4.4), plus fps, latency and memory (Table 4.5).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are the true class, columns the prediction."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    np.add.at(mat, (y_true[valid], y_pred[valid]), 1)
    return mat


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict:
    """Per-class precision/recall/F1 plus macro and weighted averages."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = np.divide(tp, np.clip(predicted, 1, None), dtype=np.float64)
    recall = np.divide(tp, np.clip(support, 1, None), dtype=np.float64)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, np.clip(denom, 1e-12, None))
    f1[denom == 0] = 0.0

    total = max(int(support.sum()), 1)
    report: Dict = {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1-score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    present = support > 0
    report["accuracy"] = round(float(tp.sum() / total), 4)
    report["macro avg"] = {
        "precision": round(float(precision[present].mean() if present.any() else 0.0), 4),
        "recall": round(float(recall[present].mean() if present.any() else 0.0), 4),
        "f1-score": round(float(f1[present].mean() if present.any() else 0.0), 4),
        "support": total,
    }
    weights = support / total
    report["weighted avg"] = {
        "precision": round(float((precision * weights).sum()), 4),
        "recall": round(float((recall * weights).sum()), 4),
        "f1-score": round(float((f1 * weights).sum()), 4),
        "support": total,
    }
    return report


def format_report(report: Dict, class_names: Sequence[str]) -> str:
    """Render a report as a fixed-width table."""
    width = max(len(n) for n in list(class_names) + ["weighted avg"]) + 2
    lines = [f"{'':<{width}}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}"]
    for name in class_names:
        row = report[name]
        lines.append(
            f"{name:<{width}}{row['precision']:>8.3f}{row['recall']:>8.3f}"
            f"{row['f1-score']:>8.3f}{row['support']:>9d}"
        )
    lines.append("")
    lines.append(f"{'accuracy':<{width}}{'':>8}{'':>8}{report['accuracy']:>8.3f}"
                 f"{report['macro avg']['support']:>9d}")
    for avg in ("macro avg", "weighted avg"):
        row = report[avg]
        lines.append(
            f"{avg:<{width}}{row['precision']:>8.3f}{row['recall']:>8.3f}"
            f"{row['f1-score']:>8.3f}{row['support']:>9d}"
        )
    return "\n".join(lines)


# ==========================================================================
# Runtime measurement
# ==========================================================================
class LatencyMeter:
    """Rolling mean/percentile latency, in milliseconds."""

    def __init__(self, window: int = 120) -> None:
        self.window = int(window)
        self._samples: List[float] = []
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        dt = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        self._samples.append(dt)
        if len(self._samples) > self.window:
            self._samples.pop(0)
        return dt

    def __enter__(self) -> "LatencyMeter":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def mean(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._samples, 95)) if self._samples else 0.0

    @property
    def fps(self) -> float:
        m = self.mean
        return 1000.0 / m if m > 0 else 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._t0 = None


def memory_snapshot() -> Dict[str, float]:
    """Process RSS, system RAM and GPU memory, all in megabytes.

    On a Jetson the GPU shares system memory, so ``gpu_used_mb`` comes from
    torch's allocator rather than from a discrete-GPU query, which is the same
    quantity the thesis reports in Table 4.5.
    """
    out: Dict[str, float] = {}
    try:
        import psutil

        proc = psutil.Process()
        out["process_rss_mb"] = proc.memory_info().rss / 1e6
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = (vm.total - vm.available) / 1e6
        out["ram_total_mb"] = vm.total / 1e6
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1e6
            out["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
            free, total = torch.cuda.mem_get_info()
            out["gpu_used_mb"] = (total - free) / 1e6
            out["gpu_total_mb"] = total / 1e6
    except Exception:
        pass
    return out


def summarize_runtime(
    frames: int,
    elapsed_s: float,
    stages: Optional[Dict[str, LatencyMeter]] = None,
) -> Dict[str, float]:
    """Assemble the Table 4.5-style runtime block."""
    summary: Dict[str, float] = {
        "frames": int(frames),
        "elapsed_s": round(float(elapsed_s), 2),
        "fps": round(frames / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "latency_ms": round(elapsed_s * 1000.0 / max(frames, 1), 2),
    }
    if stages:
        for name, meter in stages.items():
            summary[f"{name}_ms"] = round(meter.mean, 2)
            summary[f"{name}_p95_ms"] = round(meter.p95, 2)
    summary.update({k: round(v, 1) for k, v in memory_snapshot().items()})
    return summary
