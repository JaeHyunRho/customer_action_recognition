"""Fine-tune YOLOv11n-Pose on top-view data with head keypoints removed.

This is thesis §2.2. Two things make it different from a stock ultralytics pose
run:

1. the dataset has 12 keypoints, not 17, because the five facial landmarks are
   dropped (§2.1 / Fig. 2.4);
2. the shoulder and arm joints carry the thesis' extra weight of 0.5, applied
   here by shrinking their OKS sigmas so the keypoint loss penalises them more.

The thesis labels its top-view frames "using the YOLO format" without saying by
hand or by model. Hand-labelling 126,350 frames is not plausible, so
``tools/autolabel_pose.py`` pseudo-labels with a larger pose model and this
script fine-tunes the nano model on the result -- teacher-student distillation
in everything but name.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

from ..config import Config, load_config, parse_overrides
from ..keypoints import BODY12_NAMES, oks_sigmas

LOGGER = logging.getLogger(__name__)


def write_dataset_yaml(
    root: str | Path,
    output: str | Path,
    head_removal: bool = True,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
) -> Path:
    """Write the ultralytics dataset descriptor for a 12- or 17-keypoint set.

    ``flip_idx`` has to list the left/right partner of every joint, otherwise a
    horizontal-flip augmentation silently swaps limbs and teaches the model
    mirrored anatomy.
    """
    root = Path(root).resolve()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if head_removal:
        names = list(BODY12_NAMES)
        # 0 l_sh 1 r_sh 2 l_el 3 r_el 4 l_wr 5 r_wr 6 l_hip 7 r_hip
        # 8 l_knee 9 r_knee 10 l_ank 11 r_ank
        flip_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    else:
        names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    spec = {
        "path": str(root),
        "train": train_dir,
        "val": val_dir,
        "kpt_shape": [len(names), 3],
        "flip_idx": flip_idx,
        "names": {0: "person"},
        "keypoint_names": names,
    }
    with output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    LOGGER.info("wrote dataset spec %s (%d keypoints)", output, len(names))
    return output


def apply_keypoint_weights(head_removal: bool = True, bonus: float = 0.5) -> np.ndarray:
    """Patch ultralytics' OKS sigmas so arm joints are weighted harder.

    ultralytics reads the sigma table off the model at loss-construction time,
    so this has to run before ``model.train`` is called.
    """
    sigmas = oks_sigmas(head_removal=head_removal, upper_body_bonus=bonus)
    try:
        from ultralytics.utils import metrics as ul_metrics

        ul_metrics.OKS_SIGMA = sigmas.astype(np.float32)
        LOGGER.info(
            "patched OKS sigmas for %d keypoints; arm joints tightened by /%.1f",
            len(sigmas), 1.0 + bonus,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("could not patch OKS sigmas (%s); training with defaults", exc)
    return sigmas


def train_pose(
    data_yaml: str | Path,
    cfg: Config,
    weights: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: str = "custom_yolo11n_pose",
    resume: bool = False,
) -> Dict:
    """Fine-tune YOLOv11n-Pose and report the thesis' Table 2.1 metrics."""
    from ultralytics import YOLO

    from ..pose.estimator import resolve_device

    apply_keypoint_weights(cfg.pose.head_removal, cfg.pose.upper_body_bonus)

    start = weights or cfg.pose.base_weights
    model = YOLO(str(start), task="pose")
    device = resolve_device(cfg.pose.device)
    if device == "cpu":
        LOGGER.warning(
            "training on CPU. On a Jetson this will take days -- install a "
            "CUDA-enabled torch build first (see README, Jetson section)."
        )

    out_root = Path(output_dir or f"{cfg.paths.runs}/pose")
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.pose_train.epochs,
        batch=cfg.pose_train.batch_size,
        imgsz=cfg.pose_train.imgsz,
        lr0=cfg.pose_train.lr0,
        patience=cfg.pose_train.patience,
        pose=cfg.pose_train.pose_weight,
        kobj=cfg.pose_train.kobj_weight,
        freeze=cfg.pose_train.freeze,
        device=device,
        project=str(out_root),
        name=name,
        resume=resume,
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else out_root / name
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        target = Path(cfg.paths.weights) / "custom_yolo11n_pose.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        LOGGER.info("copied best weights to %s", target)

    summary: Dict = {"save_dir": str(save_dir), "weights": str(best)}
    metrics = getattr(results, "results_dict", None) or {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            summary[key.replace("metrics/", "")] = round(float(value), 4)
    (save_dir / "thesis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def evaluate_pose(
    weights: str | Path,
    data_yaml: str | Path,
    cfg: Config,
    imgsz: Optional[int] = None,
) -> Dict:
    """Validate a pose model and pull out mAP50 / mAP50-95 (thesis Table 2.1)."""
    from ultralytics import YOLO

    from ..pose.estimator import resolve_device

    apply_keypoint_weights(cfg.pose.head_removal, cfg.pose.upper_body_bonus)
    model = YOLO(str(weights), task="pose")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=int(imgsz or cfg.pose.imgsz),
        device=resolve_device(cfg.pose.device),
        verbose=False,
    )
    pose = getattr(metrics, "pose", None)
    box = getattr(metrics, "box", None)
    return {
        "mAP_pose_50": round(float(pose.map50), 4) if pose is not None else None,
        "mAP_pose_50_95": round(float(pose.map), 4) if pose is not None else None,
        "mAP_box_50": round(float(box.map50), 4) if box is not None else None,
        "mAP_box_50_95": round(float(box.map), 4) if box is not None else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11n-Pose for top-view")
    parser.add_argument("--data", required=True, help="dataset yaml (see write_dataset_yaml)")
    parser.add_argument("--weights", default=None, help="starting weights")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--name", default="custom_yolo11n_pose")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(
        args.config if Path(args.config).exists() else None, parse_overrides(args.set)
    )

    if args.eval_only:
        weights = args.weights or cfg.pose.weights
        print(json.dumps(evaluate_pose(weights, args.data, cfg), indent=2))
        return 0

    summary = train_pose(
        args.data, cfg, weights=args.weights, output_dir=args.out,
        name=args.name, resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
