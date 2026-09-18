#!/usr/bin/env python3
"""Measure the numbers the thesis reports in Tables 2.2, 4.4 and 4.5.

Covers pose-only latency, pose + tracking, and the full pipeline with each
action model, plus parameter counts and memory. Run it on the edge device and
again on a desktop to reproduce the thesis' desktop-versus-on-device comparison
(Fig. 4.3).

Usage::

    python3 tools/benchmark.py --source data/videos/demo_topview.mp4 \
        --frames 300 --out runs/bench/orin_nx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from car.config import load_config, parse_overrides  # noqa: E402
from car.models import build_model  # noqa: E402
from car.pipeline.buffer import BufferBank  # noqa: E402
from car.pose.estimator import PoseEstimator, resolve_device  # noqa: E402
from car.pose.tracker import build_tracker  # noqa: E402
from car.utils.metrics import LatencyMeter, memory_snapshot  # noqa: E402
from car.utils.video import FrameSource  # noqa: E402

LOGGER = logging.getLogger("bench")


def device_info() -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    # Jetson boards identify themselves here; on a desktop this is absent.
    model = Path("/proc/device-tree/model")
    if model.exists():
        info["board"] = model.read_text(errors="ignore").strip("\x00").strip()
    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count()
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


def bench_action_models(cfg, repeats: int = 100, batch: int = 1) -> dict:
    """Isolated forward-pass cost for each action model."""
    import torch

    device = torch.device(resolve_device(cfg.pose.device))
    out = {}
    for name in ("stgcn", "lstm"):
        model = build_model(name, cfg).to(device).eval()
        x = torch.zeros(
            batch, cfg.in_channels, cfg.sequence.window, cfg.num_keypoints, device=device
        )
        with torch.no_grad():
            for _ in range(10):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        ms = elapsed * 1000.0 / repeats
        out[name] = {
            "parameters": model.num_parameters(),
            "latency_ms": round(ms, 3),
            "throughput_hz": round(1000.0 / ms, 1) if ms > 0 else 0.0,
            "batch": batch,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_pipeline(cfg, source: str, frames: int, action_model: str | None) -> dict:
    """Stage-by-stage latency over real frames."""
    import torch

    device = torch.device(resolve_device(cfg.pose.device))
    estimator = PoseEstimator(cfg.pose)
    tracker = build_tracker(cfg.tracker, device=str(device))
    bank = BufferBank(
        window=cfg.sequence.window,
        normalize=cfg.sequence.normalize,
        conf_threshold=cfg.sequence.conf_threshold,
        min_valid_ratio=cfg.sequence.min_valid_ratio,
        use_velocity=cfg.sequence.use_velocity,
    )
    model = None
    if action_model:
        model = build_model(action_model, cfg).to(device).eval()

    meters = {k: LatencyMeter(window=frames) for k in ("pose", "track", "action", "total")}
    src = FrameSource(source, width=cfg.runtime.camera_width, height=cfg.runtime.camera_height)
    estimator.warmup()

    processed = 0
    detections = 0
    started = time.time()
    for idx, frame in src:
        meters["total"].start()

        meters["pose"].start()
        result = estimator.infer(frame)
        meters["pose"].stop()
        detections += len(result)

        meters["track"].start()
        tracks = tracker.update(result.boxes, result.scores, result.keypoints, frame=frame)
        meters["track"].stop()

        for track in tracks:
            bank.update(track.track_id, track.keypoints, track.box, idx)

        if model is not None:
            meters["action"].start()
            ids, batch = bank.batch_inputs(frame_size=(frame.shape[1], frame.shape[0]))
            if batch is not None:
                with torch.no_grad():
                    model(torch.from_numpy(batch).to(device))
            meters["action"].stop()

        meters["total"].stop()
        processed += 1
        if processed >= frames:
            break
    src.release()
    elapsed = time.time() - started

    return {
        "action_model": action_model or "none",
        "frames": processed,
        "elapsed_s": round(elapsed, 2),
        "fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
        "mean_detections": round(detections / max(processed, 1), 2),
        "pose_ms": round(meters["pose"].mean, 2),
        "pose_p95_ms": round(meters["pose"].p95, 2),
        "track_ms": round(meters["track"].mean, 2),
        "action_ms": round(meters["action"].mean, 2),
        "total_ms": round(meters["total"].mean, 2),
        "memory": {k: round(v, 1) for k, v in memory_snapshot().items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the recognition pipeline")
    parser.add_argument("--source", default=None, help="video or camera index; omit for model-only")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/bench/benchmark.json")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(
        args.config if Path(args.config).exists() else None, parse_overrides(args.set)
    )

    report = {"device": device_info(), "config": {
        "window": cfg.sequence.window,
        "imgsz": cfg.pose.imgsz,
        "keypoints": cfg.num_keypoints,
        "normalize": cfg.sequence.normalize,
        "tracker": cfg.tracker.appearance if cfg.tracker.enabled else "disabled",
    }}

    LOGGER.info("device: %s", report["device"])
    LOGGER.info("benchmarking action models ...")
    report["action_models"] = bench_action_models(cfg)
    for name, stats in report["action_models"].items():
        LOGGER.info(
            "  %-6s %9s params  %6.2f ms  %.0f Hz",
            name, f"{stats['parameters']:,}", stats["latency_ms"], stats["throughput_hz"],
        )

    if args.source and not args.models_only:
        for action_model in (None, "lstm", "stgcn"):
            tag = action_model or "pose+track only"
            LOGGER.info("benchmarking pipeline (%s) ...", tag)
            key = f"pipeline_{action_model or 'none'}"
            report[key] = bench_pipeline(cfg, args.source, args.frames, action_model)
            r = report[key]
            LOGGER.info(
                "  %-16s %5.1f fps  pose %5.1f ms  track %4.1f ms  action %4.1f ms",
                tag, r["fps"], r["pose_ms"], r["track_ms"], r["action_ms"],
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
