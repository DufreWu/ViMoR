"""Threaded camera source shared by any number of vision apps.

A background thread reads frames as fast as the camera delivers them and keeps only the
most recent one ("latest frame wins"), which is how real-time perception consumes a
camera: a slow consumer sees fresh frames instead of an ever-growing backlog. Each
captured frame ticks the ``(name, "capture")`` stream of an optional ``FpsMonitor``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from power_monitor.monitors.fps_monitor import FpsMonitor, FpsStream


@dataclass
class Frame:
    frame_id: int
    capture_ts: float  # time.monotonic() right after the frame was read
    image: Any  # numpy.ndarray (H, W, 3) BGR


class CameraSource:
    def __init__(
        self,
        device: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        backend: str = "v4l2",
        stereo_crop: Optional[str] = None,
        name: str = "camera0",
        fps_monitor: Optional[FpsMonitor] = None,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.stereo_crop = stereo_crop  # None | "left" | "right" for side-by-side stereo (e.g. ZED)
        self.name = name
        self._capture_stream: Optional[FpsStream] = (
            fps_monitor.register(name, "capture", target_fps=fps) if fps_monitor is not None else None
        )
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._next_id = 0
        self.read_failures = 0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], fps_monitor: Optional[FpsMonitor] = None) -> "CameraSource":
        return cls(
            device=cfg.get("device", 0),
            width=cfg.get("width"),
            height=cfg.get("height"),
            fps=cfg.get("fps"),
            backend=cfg.get("backend", "v4l2"),
            stereo_crop=cfg.get("stereo_crop"),
            name=cfg.get("name", "camera0"),
            fps_monitor=fps_monitor,
        )

    # ----- lifecycle ------------------------------------------------------
    def open(self) -> "CameraSource":
        import cv2  # type: ignore

        api = {"v4l2": cv2.CAP_V4L2, "gstreamer": cv2.CAP_GSTREAMER}.get(self.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.device, api)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device!r} with backend {self.backend}")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-capture", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def properties(self) -> Dict[str, Any]:
        """Negotiated camera properties (for run metadata)."""
        if self._cap is None:
            return {}
        import cv2  # type: ignore

        return {
            "device": self.device,
            "backend": self.backend,
            "width": self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "stereo_crop": self.stereo_crop,
        }

    # ----- capture loop ---------------------------------------------------
    def _crop(self, image: Any) -> Any:
        if self.stereo_crop not in ("left", "right"):
            return image
        half = image.shape[1] // 2
        return image[:, :half] if self.stereo_crop == "left" else image[:, half:]

    def _loop(self) -> None:
        while self._running.is_set():
            ok, image = self._cap.read()
            stamp = time.monotonic()
            if not ok:
                self.read_failures += 1
                time.sleep(0.005)
                continue
            frame = Frame(frame_id=self._next_id, capture_ts=stamp, image=self._crop(image))
            self._next_id += 1
            if self._capture_stream is not None:
                self._capture_stream.tick(ts=stamp, frame_id=frame.frame_id)
            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    # ----- consumer API ---------------------------------------------------
    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._latest

    def wait_newer(self, after_id: int = -1, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame with ``frame_id > after_id`` is available (or timeout)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest.frame_id <= after_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._running.is_set():
                    return self._latest if (self._latest and self._latest.frame_id > after_id) else None
                self._cond.wait(remaining)
            return self._latest


class SyntheticCamera(CameraSource):
    """Dry-run stand-in: emits black frames at ``fps`` with the same consumer API."""

    def open(self) -> "SyntheticCamera":
        import numpy as np  # type: ignore

        width = int(self.width or 1344)
        height = int(self.height or 376)
        self._image = np.zeros((height, width, 3), dtype=np.uint8)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-synthetic", daemon=True)
        self._thread.start()
        return self

    def properties(self) -> Dict[str, Any]:
        return {"device": "synthetic", "width": self.width, "height": self.height, "fps": self.fps, "stereo_crop": self.stereo_crop}

    def _loop(self) -> None:
        period = 1.0 / float(self.fps or 30)
        next_t = time.monotonic()
        while self._running.is_set():
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            stamp = time.monotonic()
            frame = Frame(frame_id=self._next_id, capture_ts=stamp, image=self._crop(self._image))
            self._next_id += 1
            if self._capture_stream is not None:
                self._capture_stream.tick(ts=stamp, frame_id=frame.frame_id)
            with self._cond:
                self._latest = frame
                self._cond.notify_all()
