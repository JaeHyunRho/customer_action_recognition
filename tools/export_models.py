#!/usr/bin/env python3
"""Export trained models for deployment: ONNX, TensorRT, int8 or TFLite.

Examples::

    # action model -> ONNX, then verify it matches PyTorch
    python3 tools/export_models.py --checkpoint runs/action/stgcn/best.pt --format onnx

    # action model -> TensorRT FP16 engine (Jetson)
    python3 tools/export_models.py --checkpoint runs/action/stgcn/best.pt \
        --format tensorrt --precision fp16

    # action model -> int8 TFLite, calibrated on real windows (the thesis' route)
    python3 tools/export_models.py --checkpoint runs/action/lstm/best.pt \
        --format tflite --calibration data/processed/merl_skeletons.npz

    # pose model -> TensorRT engine
    python3 tools/export_models.py --pose weights/yolo11n-pose.pt --format engine --half
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from car.config import load_config  # noqa: E402
from car.export.quantize import (  # noqa: E402
    calibration_from_store,
    export_onnx,
    export_pose,
    export_tensorrt,
    export_tflite,
    quantize_torch_int8,
    verify_onnx,
)
from car.models import load_model  # noqa: E402

LOGGER = logging.getLogger("export")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="action model best.pt")
    group.add_argument("--pose", help="YOLO pose weights")

    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "tensorrt", "tflite", "torch-int8", "engine",
                                 "openvino", "torchscript"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--calibration", default=None,
                        help="skeleton .npz providing int8 calibration windows")
    parser.add_argument("--half", action="store_true", help="pose export in FP16")
    parser.add_argument("--int8", action="store_true", help="pose export in INT8")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # ------------------------------------------------------------ pose
    if args.pose:
        fmt = "engine" if args.format in ("tensorrt", "engine") else args.format
        path = export_pose(
            args.pose, format=fmt, imgsz=args.imgsz,
            half=args.half or args.precision == "fp16",
            int8=args.int8 or args.precision == "int8",
        )
        print(json.dumps({"format": fmt, "output": path}, indent=2))
        return 0

    # ---------------------------------------------------------- action
    model, ckpt = load_model(args.checkpoint, map_location="cpu")
    saved_cfg = ckpt.get("config", {})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(saved_cfg, fh, sort_keys=False, allow_unicode=True)
        cfg = load_config(fh.name)

    window = cfg.sequence.window
    kpts = cfg.num_keypoints
    channels = cfg.in_channels
    stem = Path(args.checkpoint).parent
    name = ckpt.get("model_name", "action")
    LOGGER.info(
        "%s: %s params, window=%d keypoints=%d channels=%d",
        name, f"{model.num_parameters():,}", window, kpts, channels,
    )

    result = {"model": name, "parameters": model.num_parameters()}

    if args.format == "torch-int8":
        out = Path(args.out or stem / f"{name}_int8.pt")
        _, path = quantize_torch_int8(model, out)
        result["output"] = str(path)
        print(json.dumps(result, indent=2))
        return 0

    onnx_path = Path(args.out or stem / f"{name}.onnx")
    if args.format != "onnx":
        onnx_path = stem / f"{name}.onnx"
    export_onnx(model, onnx_path, window=window, num_keypoints=kpts, in_channels=channels)
    result["onnx"] = str(onnx_path)

    if not args.no_verify:
        check = verify_onnx(onnx_path, model, window, kpts, channels)
        result["verification"] = check
        if check["within_tolerance"]:
            LOGGER.info("ONNX matches PyTorch (max diff %.2e)", check["max_abs_diff"])
        else:
            LOGGER.warning(
                "ONNX output differs by %.2e; argmax agrees=%s",
                check["max_abs_diff"], check["argmax_agrees"],
            )

    calibration = None
    if args.calibration:
        calibration = calibration_from_store(args.calibration, cfg)
        LOGGER.info("calibration set: %s", calibration.shape)

    if args.format in ("tensorrt", "engine"):
        out = Path(args.out or stem / f"{name}_{args.precision}.engine")
        export_tensorrt(
            onnx_path, out, precision=args.precision,
            max_batch=args.max_batch, calibration_data=calibration,
        )
        result["engine"] = str(out)

    elif args.format == "tflite":
        out = Path(args.out or stem / f"{name}_int8.tflite")
        export_tflite(onnx_path, out, calibration_data=calibration, int8=True)
        result["tflite"] = str(out)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
