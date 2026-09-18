"""End-to-end customer action recognition (thesis Fig. 2.1, right column).

Per frame: YOLOv11n-Pose -> head removal -> StrongSORT -> per-track 30-frame
buffer -> ST-GCN or LSTM -> smoothed label. The source can equally be a
recorded top-view clip, a USB webcam, a Jetson CSI camera or an RTSP stream,
which is the whole point of routing everything through ``FrameSource``.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..models import load_model
from ..pose.estimator import PoseEstimator, resolve_device
from ..pose.tracker import build_tracker
from ..utils.metrics import LatencyMeter, summarize_runtime
from ..utils.video import FrameSource, VideoWriter
from ..utils.viz import draw_hud, draw_track
from .buffer import BufferBank

LOGGER = logging.getLogger(__name__)


@dataclass
class FramePrediction:
    """One track's state at one frame."""

    frame: int
    track_id: int
    box: np.ndarray
    label: Optional[str]
    score: float
    buffered: int

    def as_row(self) -> Dict:
        return {
            "frame": self.frame,
            "track_id": self.track_id,
            "x1": round(float(self.box[0]), 1),
            "y1": round(float(self.box[1]), 1),
            "x2": round(float(self.box[2]), 1),
            "y2": round(float(self.box[3]), 1),
            "action": self.label or "",
            "score": round(float(self.score), 4),
            "buffered": self.buffered,
        }


@dataclass
class RunSummary:
    frames: int = 0
    tracks_seen: int = 0
    predictions: int = 0
    runtime: Dict = field(default_factory=dict)
    label_counts: Dict[str, int] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "tracks_seen": self.tracks_seen,
            "predictions": self.predictions,
            "label_counts": self.label_counts,
            "runtime": self.runtime,
        }


class ActionRecognizer:
    """Pose + tracking + action classification over a frame stream."""

    def __init__(
        self,
        cfg: Config,
        action_weights: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import torch

        self.cfg = cfg
        self.device = torch.device(resolve_device(cfg.pose.device))

        self.estimator = PoseEstimator(cfg.pose)
        self.tracker = build_tracker(cfg.tracker, device=str(self.device))

        self.model = None
        self.class_names: List[str] = list(cfg.label_names)
        weights = action_weights or cfg.runtime.action_weights
        if weights and Path(weights).exists():
            self.model, ckpt = load_model(weights, map_location=str(self.device))
            self.model.to(self.device).eval()
            self.class_names = list(ckpt.get("classes", self.class_names))
            LOGGER.info(
                "action model: %s (%s params, val acc %.3f)",
                ckpt.get("model_name", model_name or "?"),
                f"{ckpt.get('num_parameters', 0):,}",
                float(ckpt.get("val_acc", 0.0)),
            )
        else:
            LOGGER.warning(
                "no action checkpoint at %s -- running pose + tracking only. "
                "Train one with car/train/train_action.py.",
                weights,
            )

        self.bank = BufferBank(
            window=cfg.sequence.window,
            normalize=cfg.sequence.normalize,
            conf_threshold=cfg.sequence.conf_threshold,
            min_valid_ratio=cfg.sequence.min_valid_ratio,
            use_velocity=cfg.sequence.use_velocity,
            max_idle=max(cfg.tracker.max_age, cfg.sequence.window),
            smooth_window=cfg.runtime.smooth_window,
        )

        self.meters = {
            "pose": LatencyMeter(),
            "track": LatencyMeter(),
            "action": LatencyMeter(),
            "total": LatencyMeter(),
        }
        self._seen_tracks: set = set()

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        self.estimator.warmup()
        if self.model is not None:
            import torch

            dummy = torch.zeros(
                1, self.cfg.in_channels, self.cfg.sequence.window, self.cfg.num_keypoints,
                device=self.device,
            )
            with torch.no_grad():
                self.model(dummy)

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[FramePrediction]:
        """Run one frame through the whole pipeline."""
        self.meters["total"].start()

        self.meters["pose"].start()
        result = self.estimator.infer(frame)
        self.meters["pose"].stop()

        self.meters["track"].start()
        tracks = self.tracker.update(
            result.boxes, result.scores, result.keypoints, frame=frame
        )
        self.meters["track"].stop()

        h, w = frame.shape[:2]
        for track in tracks:
            self._seen_tracks.add(int(track.track_id))
            self.bank.update(track.track_id, track.keypoints, track.box, frame_idx)
        self.bank.prune(frame_idx, [t.track_id for t in tracks])

        predictions: Dict[int, Tuple[Optional[str], float]] = {}
        run_action = (
            self.model is not None
            and frame_idx % max(1, self.cfg.runtime.action_every) == 0
        )
        if run_action:
            self.meters["action"].start()
            predictions = self._classify(frame_size=(w, h))
            self.meters["action"].stop()

        out: List[FramePrediction] = []
        for track in tracks:
            tid = int(track.track_id)
            smoothed = self.bank.smoothed(tid)
            label, score = (None, 0.0)
            if smoothed is not None:
                idx, score = smoothed
                if score >= self.cfg.runtime.min_action_conf and 0 <= idx < len(self.class_names):
                    label = self.class_names[idx]
            buf = self.bank.buffers.get(tid)
            out.append(
                FramePrediction(
                    frame=frame_idx,
                    track_id=tid,
                    box=track.box,
                    label=label,
                    score=float(score),
                    buffered=buf.filled if buf else 0,
                )
            )

        self.meters["total"].stop()
        return out

    # ------------------------------------------------------------------
    def _classify(self, frame_size: Tuple[int, int]) -> Dict[int, Tuple[Optional[str], float]]:
        """One batched forward pass over every track whose buffer is full."""
        import torch

        ids, batch = self.bank.batch_inputs(frame_size=frame_size)
        if batch is None:
            return {}

        x = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        out: Dict[int, Tuple[Optional[str], float]] = {}
        for tid, prob in zip(ids, probs):
            idx = int(np.argmax(prob))
            score = float(prob[idx])
            self.bank.record_prediction(tid, idx, score)
            out[tid] = (self.class_names[idx] if idx < len(self.class_names) else None, score)
        return out

    # ------------------------------------------------------------------
    def annotate(
        self,
        frame: np.ndarray,
        predictions: List[FramePrediction],
        extra_hud: Optional[List[str]] = None,
    ) -> np.ndarray:
        for pred in predictions:
            buf = self.bank.buffers.get(pred.track_id)
            kpts = buf.keypoints[-1] if buf and buf.keypoints else None
            caption = pred.label
            if caption is None:
                filled = pred.buffered
                caption = (
                    f"buffering {filled}/{self.cfg.sequence.window}"
                    if self.model is not None and filled < self.cfg.sequence.window
                    else None
                )
            draw_track(
                frame,
                pred.box,
                pred.track_id,
                label=caption,
                score=pred.score if pred.label else None,
                keypoints=kpts,
                conf_threshold=self.cfg.sequence.conf_threshold,
            )

        hud = [
            f"pose {self.meters['pose'].mean:5.1f} ms   "
            f"track {self.meters['track'].mean:4.1f} ms   "
            f"action {self.meters['action'].mean:4.1f} ms",
            f"pipeline {self.meters['total'].mean:5.1f} ms  "
            f"({self.meters['total'].fps:4.1f} fps)   tracks {len(predictions)}",
        ]
        if extra_hud:
            hud.extend(extra_hud)
        return draw_hud(frame, hud)


# ==========================================================================
def run(
    cfg: Config,
    source: Optional[str] = None,
    action_weights: Optional[str] = None,
    display: Optional[bool] = None,
    save_video: Optional[str] = None,
    save_csv: Optional[str] = None,
    save_summary: Optional[str] = None,
    max_frames: Optional[int] = None,
    loop: bool = False,
    csi: Optional[bool] = None,
) -> RunSummary:
    """Drive the recogniser over a source and collect results."""
    import cv2

    src_spec = source if source is not None else cfg.runtime.source
    use_csi = cfg.runtime.csi_camera if csi is None else bool(csi)
    show = cfg.runtime.display if display is None else bool(display)

    recognizer = ActionRecognizer(cfg, action_weights=action_weights)
    LOGGER.info("pose: %s", recognizer.estimator.describe())

    frames = FrameSource(
        src_spec,
        width=cfg.runtime.camera_width,
        height=cfg.runtime.camera_height,
        fps=cfg.runtime.camera_fps,
        csi=use_csi,
        target_fps=cfg.runtime.max_fps,
        loop=loop,
    )
    LOGGER.info("source: %s", frames.info)
    recognizer.warmup()

    writer = VideoWriter(save_video, fps=frames.info.fps) if save_video else None
    csv_file = None
    csv_writer = None
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(save_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2", "action", "score", "buffered"],
        )
        csv_writer.writeheader()

    summary = RunSummary(source=str(frames.info))
    label_counts: Dict[str, int] = {}
    started = time.time()
    processed = 0

    try:
        for frame_idx, frame in frames:
            predictions = recognizer.process_frame(frame, frame_idx)
            processed += 1

            for pred in predictions:
                if pred.label:
                    label_counts[pred.label] = label_counts.get(pred.label, 0) + 1
                    summary.predictions += 1
                if csv_writer is not None:
                    csv_writer.writerow(pred.as_row())

            if show or writer is not None:
                annotated = recognizer.annotate(frame.copy(), predictions)
                if writer is not None:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Customer Action Recognition", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        LOGGER.info("stopped by user")
                        break

            if max_frames is not None and processed >= max_frames:
                break
    except KeyboardInterrupt:
        LOGGER.info("interrupted")
    finally:
        frames.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    elapsed = time.time() - started
    summary.frames = processed
    summary.tracks_seen = len(recognizer._seen_tracks)
    summary.label_counts = dict(sorted(label_counts.items(), key=lambda kv: -kv[1]))
    summary.runtime = summarize_runtime(processed, elapsed, recognizer.meters)

    LOGGER.info(
        "processed %d frames in %.1fs (%.1f fps, %.1f ms/frame), %d tracks",
        processed, elapsed, summary.runtime["fps"], summary.runtime["latency_ms"],
        summary.tracks_seen,
    )
    if label_counts:
        LOGGER.info("actions: %s", summary.label_counts)

    if save_summary:
        Path(save_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(save_summary).write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False)
        )
    return summary
