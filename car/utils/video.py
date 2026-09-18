"""Frame sources: video files, V4L2 webcams, Jetson CSI cameras and RTSP.

Keeping every source behind one iterator is what lets the same runtime script
serve "run it on my recorded top-view clip" today and "run it on the ceiling
camera" once one is plugged in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def csi_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """GStreamer pipeline for a Jetson CSI (MIPI) camera.

    Needed because OpenCV's V4L2 path does not drive ``nvarguscamerasrc``;
    a plain ``VideoCapture(0)`` will fail on a CSI module.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=2"
    )


@dataclass
class SourceInfo:
    kind: str           # video | camera | csi | stream
    name: str
    width: int
    height: int
    fps: float
    frame_count: int    # 0 when unknown (live sources)

    @property
    def is_live(self) -> bool:
        return self.kind in {"camera", "csi", "stream"}

    def __str__(self) -> str:
        total = f" frames={self.frame_count}" if self.frame_count else ""
        return f"{self.kind}:{self.name} {self.width}x{self.height} @{self.fps:.1f}fps{total}"


class FrameSource:
    """Iterate ``(frame_index, frame)`` from any supported source."""

    def __init__(
        self,
        source: str | int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        csi: bool = False,
        target_fps: Optional[float] = None,
        loop: bool = False,
    ) -> None:
        import cv2

        self.loop = bool(loop)
        self.target_fps = target_fps
        self._cv2 = cv2
        self._spec = source

        if csi:
            sensor = int(source) if str(source).isdigit() else 0
            pipeline = csi_gstreamer_pipeline(
                sensor_id=sensor, display_width=width, display_height=height, framerate=fps
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            kind, name = "csi", f"sensor{sensor}"
        elif isinstance(source, int) or str(source).isdigit():
            index = int(source)
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            # A one-frame buffer keeps live latency down.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            kind, name = "camera", f"/dev/video{index}"
        elif str(source).startswith(("rtsp://", "http://", "https://", "udp://")):
            self.cap = cv2.VideoCapture(str(source))
            kind, name = "stream", str(source)
        else:
            path = Path(str(source))
            if not path.exists():
                raise FileNotFoundError(f"video not found: {path}")
            self.cap = cv2.VideoCapture(str(path))
            kind, name = "video", path.name

        if not self.cap.isOpened():
            raise IOError(
                f"could not open source {source!r}. "
                "For a USB camera check /dev/video*; for a CSI camera pass --csi."
            )

        self.info = SourceInfo(
            kind=kind,
            name=name,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width,
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height,
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or float(fps),
            frame_count=max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        )
        self._step = 1
        if target_fps and self.info.fps > 0 and not self.info.is_live:
            self._step = max(1, int(round(self.info.fps / target_fps)))

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        emitted = 0
        min_dt = 1.0 / self.target_fps if (self.target_fps and self.info.is_live) else 0.0
        last = 0.0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and not self.info.is_live:
                    self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    continue
                break

            if idx % self._step == 0:
                if min_dt:
                    now = time.monotonic()
                    wait = min_dt - (now - last)
                    if wait > 0:
                        time.sleep(wait)
                    last = time.monotonic()
                yield idx, frame
                emitted += 1
            idx += 1

    # ------------------------------------------------------------------
    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """Lazy MP4 writer that adopts the first frame's dimensions."""

    def __init__(self, path: str, fps: float = 30.0, fourcc: str = "mp4v") -> None:
        self.path = str(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.fourcc = fourcc
        self.writer = None

    def write(self, frame: np.ndarray) -> None:
        import cv2

        if self.writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
            )
            if not self.writer.isOpened():
                LOGGER.warning("could not open %s for writing", self.path)
                self.writer = None
                return
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_cameras(max_index: int = 8) -> list:
    """Probe ``/dev/video*`` and report which indices actually deliver frames."""
    import cv2

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append(
                    {
                        "index": i,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                    }
                )
        cap.release()
    return found
