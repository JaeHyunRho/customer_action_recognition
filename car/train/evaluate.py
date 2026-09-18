"""Evaluate a trained action model and compare models side by side.

Produces the thesis' Table 4.4 block -- accuracy, macro precision/recall, mean
F1 and parameter count -- plus a confusion matrix in the style of Fig. 4.2.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config, load_config, parse_overrides
from ..datasets.sequence import SkeletonStore, SkeletonWindowDataset
from ..models import load_model
from ..utils.metrics import classification_report, confusion_matrix, format_report
from ..utils.viz import plot_confusion_matrix

LOGGER = logging.getLogger(__name__)


def evaluate(
    checkpoint: str,
    store_path: str,
    cfg: Optional[Config] = None,
    stride: Optional[int] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 128,
) -> Dict:
    """Score a checkpoint on every window of a store."""
    import torch
    from torch.utils.data import DataLoader

    model, ckpt = load_model(checkpoint, map_location="cpu")
    class_names = list(ckpt.get("classes", []))

    if cfg is None:
        # The checkpoint carries the config it was trained with, so evaluation
        # cannot silently use a different window length or normalisation.
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(ckpt["config"], fh, sort_keys=False, allow_unicode=True)
            cfg = load_config(fh.name)
    if not class_names:
        class_names = list(cfg.label_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    store = SkeletonStore.load(store_path)
    dataset = SkeletonWindowDataset(
        store, cfg, stride=stride or cfg.sequence.infer_stride, augment=False
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no windows of length {cfg.sequence.window} in {store_path}")
    LOGGER.info("evaluating on %s", dataset.describe())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    confidences: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
            confidences.append(conf.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    conf = np.concatenate(confidences)

    cm = confusion_matrix(y_true, y_pred, len(class_names))
    report = classification_report(y_true, y_pred, class_names)

    result = {
        "checkpoint": checkpoint,
        "store": store_path,
        "model": ckpt.get("model_name"),
        "parameters": ckpt.get("num_parameters", model.num_parameters()),
        "windows": int(len(y_true)),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "mean_confidence": round(float(conf.mean()), 4),
        "classes": class_names,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }

    print(format_report(report, class_names))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        plot_confusion_matrix(
            cm, class_names, str(out / "confusion_matrix.png"),
            title=f"Confusion Matrix ({ckpt.get('model_name')})",
        )
        plot_confusion_matrix(
            cm, class_names, str(out / "confusion_matrix_norm.png"),
            title=f"Confusion Matrix ({ckpt.get('model_name')}, normalised)", normalize=True,
        )
        LOGGER.info("wrote results to %s", out)
    return result


def compare(results: Sequence[Dict]) -> str:
    """Render the thesis' Table 4.4 comparison across models."""
    if not results:
        return "(nothing to compare)"
    header = f"{'Metric':<26}" + "".join(f"{r['model'] or '?':>22}" for r in results)
    rows = [
        ("Accuracy", "accuracy"),
        ("Macro average precision", "macro_precision"),
        ("Macro average recall", "macro_recall"),
        ("Mean f1-Score", "macro_f1"),
    ]
    lines = [header, "-" * len(header)]
    for label, key in rows:
        lines.append(f"{label:<26}" + "".join(f"{r[key]:>22.2f}" for r in results))
    lines.append(
        f"{'Parameters':<26}" + "".join(f"{r['parameters']:>22,}" for r in results)
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate action model checkpoints")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="one or more best.pt files")
    parser.add_argument("--store", required=True, help="skeleton .npz to evaluate on")
    parser.add_argument("--out", default=None, help="directory for plots and json")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--config", default=None,
                        help="override the config stored in the checkpoint")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = None
    if args.config:
        cfg = load_config(args.config, parse_overrides(args.set))

    results = []
    for ckpt in args.checkpoint:
        name = Path(ckpt).parent.name
        print(f"\n=== {name} ===")
        out_dir = str(Path(args.out) / name) if args.out else None
        results.append(evaluate(ckpt, args.store, cfg, args.stride, out_dir))

    if len(results) > 1:
        print("\n" + compare(results))
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "comparison.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
