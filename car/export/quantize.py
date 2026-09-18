"""Model export and quantisation for on-device deployment.

The thesis converts the float32 PyTorch models to int8 TFLite (§3, §5). TFLite
is a reasonable target on a generic edge board, but on a Jetson it leaves the
GPU unused, so this module supports both routes and the README explains when to
pick which:

``onnx``      portable intermediate, also the input to TensorRT
``tensorrt``  Jetson's native accelerator, FP16 or INT8
``torch-int8``  dynamic quantisation, CPU only, no extra toolchain
``tflite``    the thesis' route, via onnx2tf/ai-edge-litert when installed

Action models here are small (0.35M-1.0M parameters); the pose model dominates
runtime, so quantising *it* is what actually moves frames per second.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ==========================================================================
# ONNX
# ==========================================================================
def export_onnx(
    model,
    output: str | Path,
    window: int = 30,
    num_keypoints: int = 12,
    in_channels: int = 3,
    opset: int = 13,
    dynamic_batch: bool = True,
) -> Path:
    """Export an action model to ONNX with a dynamic batch axis.

    A dynamic batch matters at inference: the number of tracked shoppers
    changes every frame, and a fixed batch would force one forward pass per
    person.
    """
    import torch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy = torch.zeros(1, in_channels, window, num_keypoints, dtype=torch.float32)

    dynamic_axes = {"skeleton": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["skeleton"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    LOGGER.info("wrote %s (%.2f MB)", output, output.stat().st_size / 1e6)
    return output


def verify_onnx(
    onnx_path: str | Path,
    model,
    window: int = 30,
    num_keypoints: int = 12,
    in_channels: int = 3,
    tolerance: float = 1e-3,
) -> Dict:
    """Compare ONNX Runtime output against PyTorch on random input."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, in_channels, window, num_keypoints).astype(np.float32)

    with torch.no_grad():
        torch_out = model.eval().cpu()(torch.from_numpy(x)).numpy()
    onnx_out = session.run(None, {"skeleton": x})[0]

    diff = float(np.abs(torch_out - onnx_out).max())
    agree = bool(np.array_equal(torch_out.argmax(1), onnx_out.argmax(1)))
    return {
        "max_abs_diff": diff,
        "within_tolerance": diff <= tolerance,
        "argmax_agrees": agree,
    }


# ==========================================================================
# TensorRT
# ==========================================================================
def export_tensorrt(
    onnx_path: str | Path,
    output: str | Path,
    precision: str = "fp16",
    workspace_gb: int = 2,
    max_batch: int = 8,
    calibration_data: Optional[np.ndarray] = None,
) -> Path:
    """Build a TensorRT engine from ONNX.

    On a Jetson this is the deployment format that actually uses the GPU.
    ``trtexec`` from the command line does the same thing; this keeps it in
    Python so the calibration set can come from real skeletons.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as fh:
        if not parser.parse(fh.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolFlag.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            LOGGER.warning("platform reports no fast FP16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_data is None:
            LOGGER.warning(
                "INT8 without calibration data falls back to dynamic ranges and "
                "will lose accuracy; pass real skeleton windows"
            )

    inp = network.get_input(0)
    shape = list(inp.shape)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        min=tuple([1] + shape[1:]),
        opt=tuple([2] + shape[1:]),
        max=tuple([max_batch] + shape[1:]),
    )
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    output.write_bytes(engine)
    LOGGER.info("wrote %s (%.2f MB, %s)", output, output.stat().st_size / 1e6, precision)
    return output


# ==========================================================================
# PyTorch dynamic int8
# ==========================================================================
def quantize_torch_int8(model, output: str | Path) -> Tuple[object, Path]:
    """Dynamic int8 quantisation of the Linear/LSTM layers, CPU only.

    This is the one quantisation path with no extra toolchain, which makes it
    the sensible default when a board has no working GPU build. Convolutions
    are untouched, so it helps the LSTM far more than the ST-GCN.
    """
    import torch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )
    torch.save({"state_dict": quantized.state_dict(), "quantized": True}, output)

    fp32 = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    LOGGER.info("wrote %s (fp32 weights were %.2f MB)", output, fp32)
    return quantized, output


# ==========================================================================
# TFLite (the thesis' route)
# ==========================================================================
def export_tflite(
    onnx_path: str | Path,
    output: str | Path,
    calibration_data: Optional[np.ndarray] = None,
    int8: bool = True,
) -> Path:
    """Convert ONNX to int8 TFLite, reproducing the thesis' §3 quantisation.

    Requires ``onnx2tf`` and ``tensorflow``, neither of which installs cleanly
    on every Jetson image, so this raises a clear error rather than failing
    deep inside a converter.
    """
    try:
        import onnx2tf  # noqa: F401
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "TFLite export needs onnx2tf and tensorflow:\n"
            "    pip install onnx2tf tensorflow onnx_graphsurgeon sng4onnx\n"
            "On a Jetson, prefer the TensorRT path instead (see docs/DEPLOYMENT.md)."
        ) from exc

    onnx_path = Path(onnx_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_model = output.parent / f"{output.stem}_saved_model"

    import onnx2tf as converter

    converter.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model),
        non_verbose=True,
    )

    tf_converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    tf_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8 and calibration_data is not None:
        def representative_dataset():
            for sample in calibration_data[:200]:
                yield [sample[None].astype(np.float32)]

        tf_converter.representative_dataset = representative_dataset
        tf_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tf_converter.inference_input_type = tf.int8
        tf_converter.inference_output_type = tf.int8

    output.write_bytes(tf_converter.convert())
    LOGGER.info("wrote %s (%.2f MB)", output, output.stat().st_size / 1e6)
    return output


# ==========================================================================
# Pose model export
# ==========================================================================
def export_pose(
    weights: str | Path,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    device: Optional[str] = None,
) -> str:
    """Export YOLOv11n-Pose through ultralytics.

    The pose network is the pipeline's bottleneck, so this is where an
    accelerated format pays off most.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights), task="pose")
    kwargs: Dict = {"format": format, "imgsz": imgsz}
    if half:
        kwargs["half"] = True
    if int8:
        kwargs["int8"] = True
    if device:
        kwargs["device"] = device
    path = model.export(**kwargs)
    LOGGER.info("exported pose model to %s", path)
    return str(path)


# ==========================================================================
def calibration_from_store(
    store_path: str | Path,
    cfg,
    limit: int = 200,
) -> np.ndarray:
    """Draw real normalised windows from a store, for int8 calibration."""
    from ..datasets.sequence import SkeletonStore, SkeletonWindowDataset

    store = SkeletonStore.load(store_path)
    dataset = SkeletonWindowDataset(store, cfg, augment=False)
    if len(dataset) == 0:
        raise RuntimeError(f"no windows in {store_path}")

    rng = np.random.default_rng(cfg.train.seed)
    picks = rng.choice(len(dataset), size=min(limit, len(dataset)), replace=False)
    return np.stack([dataset[int(i)][0].numpy() for i in picks])
