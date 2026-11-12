"""Periodic camera-driven vision application.

Any per-frame model can be wrapped as an app: pass an ``infer_fn(image) -> Any``.
Several apps can share one ``CameraSource`` and one ``FpsMonitor``; each reports to
its own ``(name, "output")`` FPS stream, so per-app throughput is tracked separately::

    fps = FpsMonitor()
    with CameraSource(0, stereo_crop="left", fps_monitor=fps) as cam:
        yolo = PeriodicVisionApp("yolo", yolo_fn, cam, target_fps=10, deadline_ms=100, fps_monitor=fps)
        seg  = PeriodicVisionApp("segformer", seg_fn, cam, target_fps=5, deadline_ms=200, fps_monitor=fps)
        fps.start()
        threads = [yolo.start_thread(60), seg.start_thread(60)]
        for t in threads: t.join()
        fps.stop()
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from power_monitor.analysis.metrics import percentile
from power_monitor.monitors.fps_monitor import FpsMonitor
from power_monitor.sensors.camera import CameraSource

InferFn = Callable[[Any], Any]


class PeriodicVisionApp:
    def __init__(
        self,
        name: str,
        infer_fn: InferFn,
        camera: CameraSource,
        target_fps: float,
        deadline_ms: float,
        fps_monitor: Optional[FpsMonitor] = None,
    ):
        self.name = name
        self.infer_fn = infer_fn
        self.camera = camera
        self.target_fps = float(target_fps)
        self.deadline_ms = float(deadline_ms)
        self.fps_stream = fps_monitor.register(name, "output", target_fps=target_fps) if fps_monitor else None
        self.frames: List[Dict[str, Any]] = []
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()

    def warmup(self, seconds: float = 0.0, min_iters: int = 3) -> None:
        """Run untimed inferences so CUDA/cuDNN initialisation does not pollute results."""
        end = time.monotonic() + max(0.0, seconds)
        done = 0
        while done < min_iters or time.monotonic() < end:
            frame = self.camera.wait_newer(timeout=2.0)
            if frame is None:
                raise RuntimeError(f"[{self.name}] camera delivered no frame during warmup")
            self.infer_fn(frame.image)
            done += 1

    def run(self, duration_s: float, t0: Optional[float] = None) -> List[Dict[str, Any]]:
        """Release one job every ``1/target_fps`` s on the freshest camera frame.

        If inference overruns so badly that whole periods have already elapsed, those
        releases are recorded as ``skipped`` (deadline missed, counted as FPS drops)
        instead of building an unbounded backlog.
        """
        period = 1.0 / self.target_fps
        start = time.monotonic() if t0 is None else t0
        end = start + duration_s
        release_idx = 0
        last_cam_id = -1
        while not self._stop.is_set():
            release = start + release_idx * period
            if release >= end:
                break
            now = time.monotonic()
            if now < release:
                time.sleep(release - now)
            elif now >= release + period:
                skipped = int((now - release) // period)
                for k in range(skipped):
                    self.frames.append(self._skipped_record(release_idx + k, start + (release_idx + k) * period))
                if self.fps_stream:
                    for k in range(skipped):
                        self.fps_stream.drop(ts=start + (release_idx + k) * period)
                release_idx += skipped
                continue

            submit = time.monotonic()
            frame = self.camera.latest()
            if frame is None:
                frame = self.camera.wait_newer(timeout=period)
            if frame is None:
                self.frames.append(self._skipped_record(release_idx, release))
                if self.fps_stream:
                    self.fps_stream.drop(ts=release)
                release_idx += 1
                continue

            infer_start = time.monotonic()
            self.infer_fn(frame.image)
            infer_end = time.monotonic()
            if self.fps_stream:
                self.fps_stream.tick(ts=infer_end, frame_id=release_idx)

            response_ms = (infer_end - submit) * 1000.0
            self.frames.append(
                {
                    "app": self.name,
                    "frame_id": release_idx,
                    "camera_frame_id": frame.frame_id,
                    "stale_frame": frame.frame_id == last_cam_id,
                    "capture_ts": frame.capture_ts,
                    "scheduled_release_ts": release,
                    "actual_submit_ts": submit,
                    "inference_start_ts": infer_start,
                    "inference_end_ts": infer_end,
                    "response_ms": response_ms,
                    "e2e_ms": (infer_end - frame.capture_ts) * 1000.0,
                    "deadline_ms": self.deadline_ms,
                    "deadline_met": (infer_end - release) * 1000.0 <= self.deadline_ms,
                    "skipped": False,
                }
            )
            last_cam_id = frame.frame_id
            release_idx += 1
        return self.frames

    def summary(self, start: float, end: float, satisfaction_window_s: float = 1.0) -> Dict[str, Any]:
        """Latency / deadline / FPS-target metrics for releases inside ``[start, end)``.

        * ``inference_*_ms``: pure model latency (``inference_end - inference_start``).
        * ``response_*_ms``: release-to-result latency (includes waiting for a frame).
        * ``fps_q``: Q_i^FPS = processed frames / measurement time / target FPS.
        * ``fps_satisfaction_rate``: share of consecutive, non-overlapping windows of
          ``satisfaction_window_s`` (aligned to ``start``) whose FPS_i >= target. A processed
          frame is attributed to the window of its *scheduled release*, so release-time
          jitter cannot move a frame across a window boundary and fake a miss.
        """
        frames = [f for f in self.frames if start <= f["scheduled_release_ts"] < end]
        done = [f for f in frames if not f.get("skipped")]
        resp = [f["response_ms"] for f in done]
        infer = [(f["inference_end_ts"] - f["inference_start_ts"]) * 1000.0 for f in done]
        e2e = [f["e2e_ms"] for f in done]
        p = self.name
        duration = end - start
        n_windows = int(duration / satisfaction_window_s + 1e-9) if satisfaction_window_s > 0 else 0
        counts = [0] * n_windows
        for f in done:
            k = int((f["scheduled_release_ts"] - start) / satisfaction_window_s + 1e-9)
            if 0 <= k < n_windows:
                counts[k] += 1
        window_fps = [c / satisfaction_window_s for c in counts]
        satisfied = [fps + 1e-9 >= self.target_fps for fps in window_fps]
        return {
            f"{p}_fps_target": self.target_fps,
            f"{p}_deadline_ms": self.deadline_ms,
            f"{p}_releases": len(frames),
            f"{p}_processed_frames": len(done),
            f"{p}_skipped": len(frames) - len(done),
            f"{p}_fps_q": (len(done) / duration / self.target_fps) if duration > 0 else None,
            f"{p}_fps_satisfaction_rate": (sum(satisfied) / len(satisfied)) if satisfied else None,
            f"{p}_fps_satisfaction_windows": len(satisfied),
            f"{p}_fps_window_min": min(window_fps) if window_fps else None,
            f"{p}_inference_p50_ms": percentile(infer, 50) if infer else None,
            f"{p}_inference_p95_ms": percentile(infer, 95) if infer else None,
            f"{p}_inference_p99_ms": percentile(infer, 99) if infer else None,
            f"{p}_inference_mean_ms": (sum(infer) / len(infer)) if infer else None,
            f"{p}_response_p50_ms": percentile(resp, 50) if resp else None,
            f"{p}_response_p95_ms": percentile(resp, 95) if resp else None,
            f"{p}_response_p99_ms": percentile(resp, 99) if resp else None,
            f"{p}_e2e_p95_ms": percentile(e2e, 95) if e2e else None,
            f"{p}_deadline_miss_rate": (sum(1 for f in frames if not f["deadline_met"]) / len(frames)) if frames else None,
            f"{p}_stale_frame_rate": (sum(1 for f in done if f.get("stale_frame")) / len(done)) if done else None,
        }

    def _skipped_record(self, frame_id: int, release: float) -> Dict[str, Any]:
        return {
            "app": self.name,
            "frame_id": frame_id,
            "scheduled_release_ts": release,
            "deadline_ms": self.deadline_ms,
            "deadline_met": False,
            "skipped": True,
        }

    def start_thread(self, duration_s: float, t0: Optional[float] = None) -> threading.Thread:
        def _target() -> None:
            try:
                self.run(duration_s, t0)
            except BaseException as exc:  # surfaced to the caller via self.error
                self.error = exc

        thread = threading.Thread(target=_target, name=f"{self.name}-app", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


class YoloInfer:
    """Ultralytics YOLO wrapper. ``input_resolution`` is the tensor shape actually fed to the net."""

    def __init__(self, model_path: str, image_size: int = 640, device: Any = 0):
        from ultralytics import YOLO  # type: ignore

        self.model_name = Path(model_path).stem
        self.image_size = int(image_size)
        self.device = device
        self.model = YOLO(model_path)
        self.input_resolution: Optional[List[int]] = None

        def _hook(_module: Any, inputs: Any) -> None:
            self.input_resolution = list(inputs[0].shape[-2:])  # [H, W] after letterbox

        self.model.model.register_forward_pre_hook(_hook)

    def __call__(self, image: Any) -> Any:
        return self.model.predict(image, imgsz=self.image_size, device=self.device, verbose=False)


class SegformerInfer:
    """HuggingFace SegFormer wrapper; resize + normalise on the GPU, argmax copied back to host."""

    def __init__(self, model_dir: str, image_size: int = 512, device: str = "cuda", fp16: bool = False):
        import torch  # type: ignore
        from transformers import SegformerForSemanticSegmentation  # type: ignore

        self.torch = torch
        self.model_name = Path(model_dir).name
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device=device, dtype=self.dtype).eval()
        self.image_size = int(image_size)
        self.input_resolution = [self.image_size, self.image_size]
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=self.dtype).view(1, 3, 1, 1)

    def __call__(self, image: Any) -> Any:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(image[:, :, ::-1].copy()).to(self.device)  # BGR -> RGB
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
            x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            logits = self.model(pixel_values=x).logits
            return logits.argmax(1).cpu()  # .cpu() synchronises, so timing covers the GPU work


class SyntheticInfer:
    """Dry-run stand-in: sleeps for a fixed service time."""

    def __init__(self, model_name: str, service_ms: float, input_resolution: Optional[List[int]] = None):
        self.model_name = model_name
        self.service_s = service_ms / 1000.0
        self.input_resolution = input_resolution

    def __call__(self, image: Any) -> Any:
        time.sleep(self.service_s)
        return None


def build_yolo_infer_fn(model_path: str, image_size: int = 640, device: Any = 0) -> InferFn:
    return YoloInfer(model_path, image_size, device)
