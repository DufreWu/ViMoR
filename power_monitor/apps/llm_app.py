"""LLM workload: requests released by a schedule, token-level timing, 100 ms time series.

Every generated token is timestamped (``time.monotonic()``). Per-request metrics
(queue delay, TTFT, TPOT, end-to-end latency) and a fixed-interval time series
(default 100 ms: phase, tokens in interval, tokens/s, queue length, ...) are derived
from those timestamps after the run, so the sampling itself adds no overhead or jitter.
"""

from __future__ import annotations

import bisect
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from power_monitor.analysis.metrics import percentile
from power_monitor.monitors.fps_monitor import FpsMonitor

_WORDS = (
    "robot camera obstacle corridor navigate battery sensor arm grasp table door kitchen "
    "person signal route map left right forward stop wait detect object chair floor light "
    "speed safety path plan goal status report explain describe summarize next action"
).split()


# ---------------------------------------------------------------------------
# Request schedule
# ---------------------------------------------------------------------------
@dataclass
class RequestSchedule:
    """When LLM requests are released.

    type:
      closed_loop - next request starts when the previous one finishes (+ think_time_s)
      periodic    - one request every interval_s (open loop; queues if the LLM is busy)
      poisson     - exponential inter-arrivals with rate_hz, drawn from the run seed
      trace       - explicit arrival offsets in seconds (arrivals_s)
    """

    type: str = "closed_loop"
    interval_s: float = 5.0
    rate_hz: float = 0.2
    think_time_s: float = 0.0
    arrivals_s: List[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "RequestSchedule":
        cfg = dict(cfg or {})
        known = {k: cfg[k] for k in ("type", "interval_s", "rate_hz", "think_time_s", "arrivals_s") if k in cfg}
        schedule = cls(**known)
        if schedule.type not in ("closed_loop", "periodic", "poisson", "trace"):
            raise ValueError(f"Unknown request_schedule.type: {schedule.type}")
        return schedule

    def arrival_offsets(self, duration_s: float, seed: int) -> Optional[List[float]]:
        """Open-loop arrival offsets (s) from the workload start; ``None`` for closed loop."""
        if self.type == "closed_loop":
            return None
        if self.type == "periodic":
            n = int(duration_s / self.interval_s) + 1
            return [i * self.interval_s for i in range(n) if i * self.interval_s < duration_s]
        if self.type == "trace":
            return sorted(t for t in self.arrivals_s if t < duration_s)
        rng = random.Random(seed)
        offsets, t = [], 0.0
        while True:
            t += rng.expovariate(self.rate_hz)
            if t >= duration_s:
                return offsets
            offsets.append(t)

    def describe(self, duration_s: float, seed: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": self.type}
        if self.type == "closed_loop":
            info["think_time_s"] = self.think_time_s
        elif self.type == "periodic":
            info["interval_s"] = self.interval_s
        elif self.type == "poisson":
            info["rate_hz"] = self.rate_hz
        offsets = self.arrival_offsets(duration_s, seed)
        if offsets is not None:
            info["arrivals_s"] = [round(t, 4) for t in offsets]
        return info


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LlamaCppBackend:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048, n_threads: Optional[int] = None, seed: int = 0, verbose: bool = False,
                 stop_at_eos: bool = False):
        from llama_cpp import Llama  # type: ignore

        if not Path(model_path).exists():
            raise FileNotFoundError(f"LLM model file missing: {model_path}")
        kwargs: Dict[str, Any] = {"model_path": model_path, "n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx, "seed": seed, "verbose": verbose}
        if n_threads:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)
        self.stop_at_eos = stop_at_eos
        import llama_cpp  # type: ignore

        vocab = llama_cpp.llama_model_get_vocab(self.llm._model.model)
        self._is_eog = lambda token: bool(llama_cpp.llama_vocab_is_eog(vocab, token))

    def make_prompt(self, n_tokens: int, rng: random.Random) -> List[int]:
        # Fresh random text per request: avoids llama.cpp's KV prefix reuse skipping prefill.
        tokens: List[int] = []
        while len(tokens) < n_tokens:
            text = " ".join(rng.choice(_WORDS) for _ in range(n_tokens))
            tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=True)
        return tokens[:n_tokens]

    def stream(self, prompt: List[int], max_tokens: int) -> Iterator[int]:
        self.llm.reset()
        produced = 0
        # generate() yields one token per decode step and never stops by itself.
        # stop_at_eos=False: always decode exactly max_tokens (fixed work per request).
        # stop_at_eos=True:  decode up to max_tokens, ending early at an end-of-generation token
        #                    (the EOG token is not counted as generated).
        for token in self.llm.generate(prompt, temp=0.0, top_k=1, reset=True):
            if self.stop_at_eos and self._is_eog(token):
                break
            yield token
            produced += 1
            if produced >= max_tokens:
                break


class SyntheticLlmBackend:
    """Dry-run stand-in: sleeps instead of computing."""

    def __init__(self, prefill_ms_per_token: float = 0.5, decode_ms_per_token: float = 25.0):
        self.prefill_s = prefill_ms_per_token / 1000.0
        self.decode_s = decode_ms_per_token / 1000.0

    def make_prompt(self, n_tokens: int, rng: random.Random) -> List[int]:
        return [rng.randrange(1000) for _ in range(n_tokens)]

    def stream(self, prompt: List[int], max_tokens: int) -> Iterator[int]:
        time.sleep(self.prefill_s * len(prompt))
        for i in range(max_tokens):
            if i:
                time.sleep(self.decode_s)
            yield i


def guess_quantization(model_path: str) -> Optional[str]:
    match = re.search(r"(q\d(?:_[a-z0-9]+)*|f16|fp16|bf16|f32)\.gguf$", Path(model_path).name, re.IGNORECASE)
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class LlmApp:
    def __init__(
        self,
        backend: Any,
        prompt_tokens: int,
        decode_tokens: int,
        schedule: RequestSchedule,
        seed: int = 0,
        name: str = "llm",
        fps_monitor: Optional[FpsMonitor] = None,
    ):
        self.backend = backend
        self.prompt_tokens = int(prompt_tokens)
        self.decode_tokens = int(decode_tokens)
        self.schedule = schedule
        self.seed = int(seed)
        self.name = name
        self.rng = random.Random(seed)
        self.token_stream = fps_monitor.register(name, "token") if fps_monitor is not None else None
        self.requests: List[Dict[str, Any]] = []
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()

    def warmup(self, n_requests: int = 1) -> None:
        for _ in range(n_requests):
            for _ in self.backend.stream(self.backend.make_prompt(self.prompt_tokens, self.rng), min(8, self.decode_tokens)):
                pass

    # ----- execution --------------------------------------------------------
    def run(self, duration_s: float, t0: Optional[float] = None) -> List[Dict[str, Any]]:
        t0 = time.monotonic() if t0 is None else t0
        end = t0 + duration_s
        offsets = self.schedule.arrival_offsets(duration_s, self.seed)
        req_idx = 0
        next_arrival = t0
        while not self._stop.is_set():
            if offsets is not None:
                if req_idx >= len(offsets):
                    break
                next_arrival = t0 + offsets[req_idx]
            if next_arrival >= end:
                break
            delay = next_arrival - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            record = self._run_request(f"req_{req_idx:04d}", next_arrival, end)
            self.requests.append(record)
            req_idx += 1
            if offsets is None:
                next_arrival = record["end_ts"] + self.schedule.think_time_s
        return self.requests

    def _run_request(self, request_id: str, arrival: float, hard_end: float) -> Dict[str, Any]:
        prompt = self.backend.make_prompt(self.prompt_tokens, self.rng)
        start = time.monotonic()
        token_ts: List[float] = []
        status = "completed"
        for _ in self.backend.stream(prompt, self.decode_tokens):
            stamp = time.monotonic()
            token_ts.append(stamp)
            if self.token_stream is not None:
                self.token_stream.tick(ts=stamp)
            if self._stop.is_set() or stamp >= hard_end:
                status = "aborted" if len(token_ts) < self.decode_tokens else "completed"
                break
        end_ts = time.monotonic()
        first = token_ts[0] if token_ts else None
        n = len(token_ts)
        return {
            "request_id": request_id,
            "prompt_tokens": len(prompt),
            "max_tokens": self.decode_tokens,
            "generated_tokens": n,
            "arrival_ts": arrival,
            "start_ts": start,
            "first_token_ts": first,
            "end_ts": end_ts,
            "queue_ms": max(0.0, (start - arrival) * 1000.0),
            "ttft_ms": (first - start) * 1000.0 if first is not None else None,
            "tpot_ms": ((token_ts[-1] - first) / (n - 1) * 1000.0) if n >= 2 else None,
            "e2e_ms": (end_ts - arrival) * 1000.0,
            "decode_tps": ((n - 1) / (token_ts[-1] - first)) if n >= 2 and token_ts[-1] > first else None,
            "status": status,
            "token_ts": token_ts,
        }

    def start_thread(self, duration_s: float, t0: Optional[float] = None) -> threading.Thread:
        def _target() -> None:
            try:
                self.run(duration_s, t0)
            except BaseException as exc:
                self.error = exc

        thread = threading.Thread(target=_target, name=f"{self.name}-app", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    # ----- analysis -----------------------------------------------------------
    def timeseries(self, start: float, end: float, interval_s: float = 0.1) -> List[Dict[str, Any]]:
        """One row per ``interval_s`` bin ``(t - interval_s, t]`` inside ``[start, end]``."""
        all_tokens = sorted(t for r in self.requests for t in r["token_ts"])
        rows: List[Dict[str, Any]] = []
        cumulative = bisect.bisect_right(all_tokens, start)
        base = cumulative
        k = 1
        while True:
            t = start + k * interval_s
            if t > end + 1e-9:
                break
            hi = bisect.bisect_right(all_tokens, t)
            lo = bisect.bisect_right(all_tokens, t - interval_s)
            phase, active = "idle", ""
            for r in self.requests:
                if r["start_ts"] <= t < r["end_ts"]:
                    active = r["request_id"]
                    phase = "prefill" if (r["first_token_ts"] is None or t < r["first_token_ts"]) else "decode"
                    break
            rows.append(
                {
                    "monotonic": t,
                    "t_rel_s": round(k * interval_s, 6),
                    "phase": phase,
                    "active_request": active,
                    "tokens_in_interval": hi - lo,
                    "tokens_per_s": (hi - lo) / interval_s,
                    "cumulative_tokens": hi - base,
                    "completed_requests": sum(1 for r in self.requests if start <= r["end_ts"] <= t and r["status"] == "completed"),
                    "queued_requests": sum(1 for r in self.requests if r["arrival_ts"] <= t < r["start_ts"]),
                }
            )
            k += 1
        return rows

    def summary(self, start: float, end: float) -> Dict[str, Any]:
        """Requests that *arrived* inside the window and completed; token rate over the window."""
        done = [r for r in self.requests if start <= r["arrival_ts"] < end and r["status"] == "completed"]
        tokens = sum(1 for r in self.requests for t in r["token_ts"] if start <= t <= end)

        def stats(key: str) -> Dict[str, float]:
            values = [r[key] for r in done if r.get(key) is not None]
            return {
                f"llm_{key}_mean": (sum(values) / len(values)) if values else None,
                f"llm_{key}_p50": percentile(values, 50) if values else None,
                f"llm_{key}_p95": percentile(values, 95) if values else None,
            }

        out: Dict[str, Any] = {
            "llm_requests_completed": len(done),
            "llm_requests_aborted": sum(1 for r in self.requests if r["status"] == "aborted"),
            "llm_tokens_in_window": tokens,
            "llm_tokens_per_s": tokens / (end - start) if end > start else 0.0,
        }
        for key in ("ttft_ms", "tpot_ms", "e2e_ms", "queue_ms"):
            out.update(stats(key))
        return out


# ---------------------------------------------------------------------------
# Process isolation
# ---------------------------------------------------------------------------
def _llm_worker(conn: Any, backend_kwargs: Optional[Dict[str, Any]], app_kwargs: Dict[str, Any], warmup_requests: int) -> None:
    """Child process: load the model, warm up, then run one measurement on request."""
    import traceback

    try:
        backend = LlamaCppBackend(**backend_kwargs) if backend_kwargs is not None else SyntheticLlmBackend()
        app_kwargs = dict(app_kwargs)
        app_kwargs["schedule"] = RequestSchedule(**app_kwargs["schedule"])
        app = LlmApp(backend, **app_kwargs)
        app.warmup(warmup_requests)
        conn.send(("ready", None))
        cmd, payload = conn.recv()
        if cmd == "run":
            duration_s, t0 = payload
            stopper = threading.Thread(target=lambda: (conn.poll(None), app.stop()), daemon=True)
            stopper.start()  # any further message from the parent aborts the run
            app.run(duration_s, t0)
            conn.send(("done", app.requests))
    except BaseException as exc:  # report to the parent instead of dying silently
        try:
            conn.send(("error", f"{exc!r}\n{traceback.format_exc()}"))
        except Exception:
            pass


class LlmProcessApp(LlmApp):
    """``LlmApp`` whose model runs in a separate process (own CUDA context, no shared GIL).

    llama.cpp captures decode steps as CUDA graphs; CUDA calls made by PyTorch from other
    threads of the *same* process invalidate the capture and abort llama.cpp. A separate
    process also matches a real deployment (LLM server next to perception nodes).
    ``time.monotonic()`` is system-wide on Linux, so child timestamps share the parent clock.
    Analysis (``summary``/``timeseries``) runs in the parent on the returned records.
    """

    def __init__(self, backend_kwargs: Optional[Dict[str, Any]], prompt_tokens: int, decode_tokens: int, schedule: RequestSchedule,
                 seed: int = 0, name: str = "llm", fps_monitor: Optional[FpsMonitor] = None, load_timeout_s: float = 600.0):
        import multiprocessing as mp

        super().__init__(None, prompt_tokens, decode_tokens, schedule, seed, name, fps_monitor)
        self._backend_kwargs = backend_kwargs
        self._load_timeout_s = load_timeout_s
        self._ctx = mp.get_context("spawn")  # fork would inherit the parent's CUDA state
        self._conn, self._child_conn = self._ctx.Pipe()
        self._proc: Any = None

    def warmup(self, n_requests: int = 1) -> None:
        app_kwargs = {"prompt_tokens": self.prompt_tokens, "decode_tokens": self.decode_tokens,
                      "schedule": asdict(self.schedule), "seed": self.seed, "name": self.name}
        self._proc = self._ctx.Process(target=_llm_worker, args=(self._child_conn, self._backend_kwargs, app_kwargs, n_requests),
                                       name=f"{self.name}-process", daemon=True)
        self._proc.start()
        self._receive("ready", self._load_timeout_s)

    def _receive(self, expected: str, timeout: Optional[float]) -> Any:
        if not self._conn.poll(timeout):
            raise TimeoutError(f"[{self.name}] no '{expected}' from LLM process within {timeout}s")
        try:
            kind, payload = self._conn.recv()
        except EOFError:
            self._proc.join(timeout=5)
            raise RuntimeError(f"[{self.name}] LLM process died (exit code {self._proc.exitcode})") from None
        if kind == "error":
            raise RuntimeError(f"[{self.name}] LLM process failed: {payload}")
        if kind != expected:
            raise RuntimeError(f"[{self.name}] unexpected message {kind!r} from LLM process")
        return payload

    def start_thread(self, duration_s: float, t0: Optional[float] = None) -> threading.Thread:
        t0 = time.monotonic() if t0 is None else t0
        self._conn.send(("run", (duration_s, t0)))

        def _target() -> None:
            try:
                self.requests = self._receive("done", duration_s + (t0 - time.monotonic()) + 120.0)
                if self.token_stream is not None:
                    for r in self.requests:
                        for stamp in r["token_ts"]:
                            self.token_stream.tick(ts=stamp)
            except BaseException as exc:
                self.error = exc

        thread = threading.Thread(target=_target, name=f"{self.name}-ipc", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.is_alive():
            try:
                self._conn.send(("stop", None))
            except (OSError, BrokenPipeError):
                pass
            self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        self._proc = None
