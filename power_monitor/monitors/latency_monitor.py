from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List


class LatencyMonitor:
    """Tracks LLM and YOLO event timings using a monotonic clock."""

    def __init__(self):
        self.llm_requests: List[Dict[str, Any]] = []
        self.perception_frames: List[Dict[str, Any]] = []

    def start_llm_request(self, request_id: str, prompt_tokens: int = 0, max_tokens: int = 0) -> Dict[str, float]:
        event = {
            "request_id": request_id,
            "prompt_tokens": int(prompt_tokens),
            "max_tokens": int(max_tokens),
            "queue_start_monotonic": time.monotonic(),
            "queue_end_monotonic": None,
            "ttft_monotonic": None,
            "first_token_monotonic": None,
            "completion_monotonic": None,
            "status": "queued",
        }
        self.llm_requests.append(event)
        return event

    def complete_llm_request(self, request_id: str, actual_tokens: int = 0, status: str = "completed") -> Dict[str, Any]:
        for event in self.llm_requests:
            if event["request_id"] != request_id:
                continue
            event["completion_monotonic"] = time.monotonic()
            event["queue_end_monotonic"] = event["completion_monotonic"]
            event["actual_tokens"] = int(actual_tokens)
            event["status"] = status
            return event
        return {"request_id": request_id, "status": status, "actual_tokens": int(actual_tokens)}

    def add_yolo_frame(self, frame_id: int, scheduled_release_ts: float, actual_submit_ts: float, inference_start_ts: float, inference_end_ts: float, deadline_ms: float) -> Dict[str, Any]:
        record = {
            "frame_id": int(frame_id),
            "scheduled_release_ts": float(scheduled_release_ts),
            "actual_submit_ts": float(actual_submit_ts),
            "inference_start_ts": float(inference_start_ts),
            "inference_end_ts": float(inference_end_ts),
            "response_ms": max(0.0, (inference_end_ts - actual_submit_ts) * 1000.0),
            "deadline_ms": float(deadline_ms),
            "deadline_met": (inference_end_ts - actual_submit_ts) * 1000.0 <= float(deadline_ms),
        }
        self.perception_frames.append(record)
        return record

    def write_llm_csv(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "request_id",
                "prompt_tokens",
                "max_tokens",
                "actual_tokens",
                "queue_start_monotonic",
                "queue_end_monotonic",
                "ttft_monotonic",
                "first_token_monotonic",
                "completion_monotonic",
                "status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.llm_requests:
                writer.writerow(row)

    def write_yolo_csv(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, extrasaction="ignore", fieldnames=[
                "app",
                "frame_id",
                "camera_frame_id",
                "stale_frame",
                "skipped",
                "capture_ts",
                "scheduled_release_ts",
                "actual_submit_ts",
                "inference_start_ts",
                "inference_end_ts",
                "response_ms",
                "e2e_ms",
                "deadline_ms",
                "deadline_met",
            ])
            writer.writeheader()
            for row in self.perception_frames:
                writer.writerow(row)
