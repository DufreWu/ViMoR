"""VLM-only TTFT / TTA benchmark (Qwen2.5-VL via llama.cpp mtmd), no co-running apps.

Sweeps input (image resolution x number of images x text prompt length) and,
optionally, a CPU x GPU DVFS grid. Every request is run closed loop (no queueing)
and split into stages, each timed after a device sync:

    t_pre      mtmd_tokenize: image resize / patching + text tokenisation (CPU)
    t_enc      mtmd_encode_chunk: vision encoder (ViT + merger), summed over images
    t_img_pf   LLM prefill of the visual tokens
    t_txt_pf   LLM prefill of the text tokens (incl. logits of the last one)
    ttft       t_pre + t_enc + t_img_pf + t_txt_pf + first-token sampling
    tta_n      time from request start until the n-th generated token

Decoding is greedy and always runs --n-decode tokens (EOS ignored) so TTA(n) is
measured for every n in --n-list. One CSV row per request; power (INA3221 VDD_IN)
is integrated per request (energy_j, mean_w) and up to the n-th token (energy_n).
CPU/GPU clocks are sampled during each request (cpu_mean, gpu_mean).

DVFS points: the --cpu x --gpu grid (locked clocks), plus --governor default
(stock schedutil + nvhost_podgov, recorded as cpu=0, gpu=0).
--idle-s N: at every DVFS point also measure idle board power for N s with the
model loaded (written to --idle-out); --idle-only skips the inference sweep.

Examples:
  python speed_vlm/vlm_bench.py --no-dvfs --reps 5                # current clocks
  sudo -E python speed_vlm/vlm_bench.py --cpu 1984 --gpu 306,612,918 --governor default --idle-s 30
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llama_cpp  # noqa: E402
from llama_cpp import mtmd_cpp as M  # noqa: E402

# ---------------------------------------------------------------------------
# mtmd functions that llama-cpp-python 0.3.19 does not bind (exported by libmtmd)
# ---------------------------------------------------------------------------
_lib = M._libmtmd
_lib.mtmd_encode_chunk.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.mtmd_encode_chunk.restype = ctypes.c_int32
_lib.mtmd_get_output_embd.argtypes = [ctypes.c_void_p]
_lib.mtmd_get_output_embd.restype = ctypes.POINTER(ctypes.c_float)
_lib.mtmd_helper_decode_image_chunk.argtypes = [
    ctypes.c_void_p, llama_cpp.llama_context_p_ctypes, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    llama_cpp.llama_pos, llama_cpp.llama_seq_id, ctypes.c_int32, ctypes.POINTER(llama_cpp.llama_pos)]
_lib.mtmd_helper_decode_image_chunk.restype = ctypes.c_int32
_lib.mtmd_input_chunk_get_n_pos.argtypes = [ctypes.c_void_p]
_lib.mtmd_input_chunk_get_n_pos.restype = llama_cpp.llama_pos

FILLER = ("robot camera obstacle corridor navigate door person chair table left right forward stop "
          "wait detect object floor light path goal safety route sign stairs wall turn slow").split()
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"


class FreqSampler:
    """Background sampler of CPU0 / GPU current clocks (MHz) for per-request means."""

    def __init__(self, dvfs, interval_s: float = 0.05):
        import threading
        self.dvfs, self.interval, self.samples = dvfs, interval_s, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            c, g = self.dvfs.read_cpu_frequency(0), self.dvfs.read_gpu_frequency()
            self.samples.append((time.monotonic(), (c or 0) / 1000, (g or 0) / 1e6))
            if len(self.samples) > 200000:
                del self.samples[:100000]

    def mean(self, t0: float, t1: float):
        w = [(c, g) for t, c, g in self.samples if t0 <= t <= t1]
        if not w:
            return None, None
        return round(sum(c for c, _ in w) / len(w)), round(sum(g for _, g in w) / len(w))

    def stop(self):
        self._stop.set()


def parse_res(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


class VlmRunner:
    def __init__(self, model: str, mmproj: str, n_ctx: int, n_batch: int, n_threads: int | None,
                 image_max_tokens: int, flash_attn: bool, verbose: bool):
        kw = dict(model_path=model, n_gpu_layers=-1, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_batch,
                  seed=0, verbose=verbose, flash_attn=flash_attn)
        if n_threads:
            kw["n_threads"] = n_threads
        self.llm = llama_cpp.Llama(**kw)
        self.ctx = self.llm._ctx.ctx
        self.n_batch = n_batch
        p = M.mtmd_context_params_default()
        p.use_gpu = True
        p.print_timings = verbose
        p.n_threads = self.llm.n_threads
        p.flash_attn_type = (llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED if flash_attn
                             else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED)
        if image_max_tokens > 0:
            p.image_max_tokens = image_max_tokens
        self.mctx = M.mtmd_init_from_file(mmproj.encode(), self.llm.model, p)
        if not self.mctx:
            raise RuntimeError(f"failed to load mmproj {mmproj}")
        self.marker = M.mtmd_default_marker().decode()
        self.n_vocab = self.llm.n_vocab()
        self._prompt_cache: dict = {}

    # ----- inputs ------------------------------------------------------------
    def prompt(self, n_txt: int, n_img: int) -> str:
        """Chat prompt whose user text is ~n_txt tokens (filler), with n_img media markers."""
        key = (n_txt, n_img)
        if key not in self._prompt_cache:
            rng = np.random.default_rng(n_txt)
            words, body = [], ""
            while True:
                body = "Describe the scene. " + " ".join(words)
                if len(self.llm.tokenize(body.encode(), add_bos=False)) >= n_txt:
                    break
                words.append(str(rng.choice(FILLER)))
            self._prompt_cache[key] = SYSTEM + self.marker * n_img + body + TAIL
        return self._prompt_cache[key]

    @staticmethod
    def image(base: np.ndarray, w: int, h: int) -> np.ndarray:
        from PIL import Image
        return np.ascontiguousarray(np.asarray(Image.fromarray(base).resize((w, h), Image.BILINEAR)))

    # ----- one request ---------------------------------------------------------
    def run(self, images, text: str, n_decode: int):
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(self.ctx), True)
        llama_cpp.llama_synchronize(self.ctx)
        rec = {"t_enc": 0.0, "t_img_pf": 0.0, "t_txt_pf": 0.0, "n_vis": 0, "n_txt": 0}
        t0 = time.monotonic()

        bitmaps = [M.mtmd_bitmap_init(im.shape[1], im.shape[0],
                                      im.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))) for im in images]
        chunks = M.mtmd_input_chunks_init()
        try:
            inp = M.mtmd_input_text(text.encode(), True, True)
            arr = (M.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
            if M.mtmd_tokenize(self.mctx, chunks, ctypes.byref(inp), arr, len(bitmaps)) != 0:
                raise RuntimeError("mtmd_tokenize failed")
            rec["t_pre"] = time.monotonic() - t0

            n_past = llama_cpp.llama_pos(0)
            n_chunks = M.mtmd_input_chunks_size(chunks)
            for i in range(n_chunks):
                ch = M.mtmd_input_chunks_get(chunks, i)
                n_tok = M.mtmd_input_chunk_get_n_tokens(ch)
                new = llama_cpp.llama_pos(0)
                if M.mtmd_input_chunk_get_type(ch) == M.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    ta = time.monotonic()
                    if M.mtmd_helper_eval_chunk_single(self.mctx, self.ctx, ch, n_past, 0, self.n_batch,
                                                       i == n_chunks - 1, ctypes.byref(new)) != 0:
                        raise RuntimeError("text prefill failed")
                    llama_cpp.llama_synchronize(self.ctx)
                    rec["t_txt_pf"] += time.monotonic() - ta
                    rec["n_txt"] += n_tok
                else:
                    ta = time.monotonic()
                    if _lib.mtmd_encode_chunk(self.mctx, ch) != 0:
                        raise RuntimeError("image encode failed")
                    tb = time.monotonic()
                    if _lib.mtmd_helper_decode_image_chunk(self.mctx, self.ctx, ch, _lib.mtmd_get_output_embd(self.mctx),
                                                           n_past, 0, self.n_batch, ctypes.byref(new)) != 0:
                        raise RuntimeError("image prefill failed")
                    llama_cpp.llama_synchronize(self.ctx)
                    rec["t_enc"] += tb - ta
                    rec["t_img_pf"] += time.monotonic() - tb
                    rec["n_vis"] += n_tok
                # positions advance by n_pos (M-RoPE: max(nx, ny) for an image, n_tokens for text);
                # tracked here because the image helper does not reliably write new_n_past
                n_past = llama_cpp.llama_pos(n_past.value + _lib.mtmd_input_chunk_get_n_pos(ch))
        finally:
            M.mtmd_input_chunks_free(chunks)
            for b in bitmaps:
                M.mtmd_bitmap_free(b)

        # greedy decode, fixed length; positions continue from the KV cache
        token_t = []
        tok = (llama_cpp.llama_token * 1)()
        for k in range(n_decode):
            logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self.ctx, -1), shape=(self.n_vocab,))
            tok[0] = int(logits.argmax())
            token_t.append(time.monotonic() - t0)
            if k == n_decode - 1:
                break
            if llama_cpp.llama_decode(self.ctx, llama_cpp.llama_batch_get_one(tok, 1)) != 0:
                raise RuntimeError("decode failed")
        rec["ttft"] = token_t[0]
        rec["token_t"] = token_t
        rec["t_end"] = time.monotonic()
        rec["t_start"] = t0
        return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    md = ROOT / "workloads/vlm_models"
    ap.add_argument("--model", default=str(md / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(md / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"))
    ap.add_argument("--image", help="base image (default: synthetic); resized to every resolution")
    ap.add_argument("--res", default="224x224,336x336,448x448,672x376,896x504,1344x752")
    ap.add_argument("--n-img", default="1,2,4")
    ap.add_argument("--n-txt", default="32,128,512", help="target user-text tokens")
    ap.add_argument("--n-decode", type=int, default=64)
    ap.add_argument("--n-list", default="1,2,4,8,16,32,64", help="n for which TTA(n) is stored")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-vis", type=int, default=4096, help="skip configs with more visual tokens (n_img*est.)")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-batch", type=int, default=2048)
    ap.add_argument("--n-threads", type=int)
    ap.add_argument("--image-max-tokens", type=int, default=-1, help="-1 = mmproj default")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--cpu", default="", help="CPU MHz list (needs root); empty = do not touch")
    ap.add_argument("--gpu", default="", help="GPU MHz list (needs root); empty = do not touch")
    ap.add_argument("--no-dvfs", action="store_true", help="never touch clocks; record current ones")
    ap.add_argument("--governor", choices=["default"], help="also run under the stock governors (cpu=0, gpu=0)")
    ap.add_argument("--cpu-governor", default="schedutil")
    ap.add_argument("--gpu-governor", default="nvhost_podgov")
    ap.add_argument("--idle-s", type=float, default=0.0, help="idle power measurement per DVFS point [s]; 0 = off")
    ap.add_argument("--idle-only", action="store_true", help="only measure idle power, no inference")
    ap.add_argument("--idle-out", help="idle CSV (default: <out>_idle.csv)")
    ap.add_argument("--power-interval", type=float, default=0.02)
    ap.add_argument("--out", default=str(ROOT / "speed_vlm/data/vlm_measured.csv"))
    ap.add_argument("--resume", action="store_true", help="skip configs that already have --reps rows in --out")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from power_monitor.controllers.dvfs_controller import DvfsController
    from power_monitor.monitors.power_monitor import JetsonPowerMonitor

    for f in (a.model, a.mmproj):
        if not Path(f).exists():
            sys.exit(f"missing model file: {f}")
    use_dvfs = not a.no_dvfs and bool(a.cpu or a.gpu or a.governor)
    if use_dvfs and os.geteuid() != 0:
        sys.exit("DVFS control needs root: sudo -E python speed_vlm/vlm_bench.py ...  (or pass --no-dvfs)")

    if a.image:
        from PIL import Image
        base = np.asarray(Image.open(a.image).convert("RGB"))
    else:  # smooth gradient + noise; timing depends on size only, not content
        rng = np.random.default_rng(0)
        y, x = np.mgrid[0:752, 0:1344]
        base = np.stack([x * 255 // 1344, y * 255 // 752, (x + y) * 255 // 2096], -1)
        base = np.clip(base + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)

    res_list = [parse_res(r) for r in a.res.split(",")]
    inputs = [(r, k, t) for r, k, t in itertools.product(res_list, ints(a.n_img), ints(a.n_txt))
              if k * (r[0] // 28) * (r[1] // 28) <= a.max_vis]
    n_list = [n for n in ints(a.n_list) if n <= a.n_decode]

    dvfs = DvfsController()
    if a.no_dvfs or not (a.cpu or a.gpu or a.governor):
        points = [(None, None)]                       # current clocks, untouched
    else:
        points = ([(c, g) for c in (ints(a.cpu) or [None]) for g in (ints(a.gpu) or [None])]
                  if (a.cpu or a.gpu) else [])
        if a.governor == "default":
            points.append((0, 0))                     # 0/0 = stock governors
    if use_dvfs:
        dvfs.save_original_state()

    runner = VlmRunner(a.model, a.mmproj, a.n_ctx, a.n_batch, a.n_threads, a.image_max_tokens,
                       a.flash_attn, a.verbose)
    pm = JetsonPowerMonitor()
    pm.start(a.power_interval)
    fs = FreqSampler(dvfs)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = (["cpu", "gpu", "cpu_actual", "gpu_actual", "res_w", "res_h", "n_img", "n_txt_target", "rep",
               "n_vis", "n_txt", "t_pre", "t_enc", "t_img_pf", "t_txt_pf", "ttft", "tpot"]
              + [f"tta_{n}" for n in n_list] + ["energy_j", "mean_w"] + [f"energy_{n}" for n in n_list]
              + ["cpu_mean", "gpu_mean"])
    has_rows = out.exists() and out.stat().st_size > 0   # an empty file (e.g. from a crash) counts as new
    if has_rows and not a.idle_only:
        with open(out) as f0:
            old = f0.readline().strip().split(",")
        if old != fields:
            sys.exit(f"{out} has different columns (older format or other --n-list); use another --out")
    fh = wr = None
    if not a.idle_only:
        new_file = not has_rows
        fh = open(out, "w" if new_file else "a", newline="")
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            wr.writeheader()
            fh.flush()

    idle_out = Path(a.idle_out) if a.idle_out else out.with_name(out.stem + "_idle.csv")
    idle_fields = ["cpu", "gpu", "cpu_mean", "gpu_mean", "idle_w", "duration_s", "energy_j"]
    idle_done = set()
    if a.idle_s > 0 and idle_out.exists() and a.resume:
        import pandas as pd
        idle_done = {(int(c), int(g)) for c, g in pd.read_csv(idle_out)[["cpu", "gpu"]].values}

    finished = set()
    if a.resume and has_rows:
        import pandas as pd
        prev = pd.read_csv(out)
        cnt = prev.groupby(["cpu", "gpu", "res_w", "res_h", "n_img", "n_txt_target"]).size()
        finished = {tuple(int(v) for v in k) for k, n in cnt.items() if n >= a.reps}
        print(f"resume: {len(finished)} configs already complete in {out}")
    total = len(points) * (0 if a.idle_only else len(inputs)) * a.reps
    done = 0
    try:
        for c, g in points:
            if use_dvfs:
                if c == 0 and g == 0:
                    dvfs.set_default_governors(a.cpu_governor, a.gpu_governor)
                else:  # snap MHz to the nearest supported OPP (e.g. 1190 -> 1190400 kHz)
                    if c:
                        dvfs.set_cpu_frequency(min(dvfs.available_cpu_hz or [c * 1000], key=lambda f: abs(f - c * 1000)))
                    if g:
                        dvfs.set_gpu_frequency(min(dvfs.available_gpu_hz or [g * 10**6], key=lambda f: abs(f - g * 10**6)))
                time.sleep(1.0)
            act = dvfs.read_actual_frequencies()
            cpu_act = round(max(act["cpu_hz"]) / 1000) if act["cpu_hz"] else None
            gpu_act = round(act["gpu_hz"] / 1e6) if act["gpu_hz"] else None
            ckey, gkey = (c if c is not None else cpu_act), (g if g is not None else gpu_act)

            if a.idle_s > 0 and (ckey, gkey) not in idle_done:
                time.sleep(2.0)  # let the clocks / power settle after the switch
                t0 = time.monotonic()
                time.sleep(a.idle_s)
                t1 = time.monotonic()
                pw = pm.integrate_window(t0, t1)
                cm, gm = fs.mean(t0, t1)
                new_idle = not idle_out.exists()
                with open(idle_out, "a", newline="") as fi:
                    wi = csv.DictWriter(fi, fieldnames=idle_fields)
                    if new_idle:
                        wi.writeheader()
                    wi.writerow(dict(cpu=ckey, gpu=gkey, cpu_mean=cm, gpu_mean=gm, idle_w=pw["mean_w"],
                                     duration_s=round(t1 - t0, 2), energy_j=pw["energy_j"]))
                print(f"[idle] cpu={ckey} gpu={gkey}: {pw['mean_w']:.2f} W (clocks {cm}/{gm} MHz)", flush=True)
            if a.idle_only:
                continue

            for (w, h), k, nt in inputs:
                if (ckey, gkey, w, h, k, nt) in finished:
                    done += a.reps
                    continue
                imgs = [runner.image(base, w, h)] * k
                text = runner.prompt(nt, k)
                for _ in range(a.warmup):
                    runner.run(imgs, text, min(4, a.n_decode))
                for rep in range(a.reps):
                    r = runner.run(imgs, text, a.n_decode)
                    pw = pm.integrate_window(r["t_start"], r["t_end"])
                    tt = r["token_t"]
                    cm, gm = fs.mean(r["t_start"], r["t_end"])
                    row = dict(cpu=ckey, gpu=gkey, cpu_actual=cpu_act, gpu_actual=gpu_act,
                               cpu_mean=cm, gpu_mean=gm,
                               res_w=w, res_h=h, n_img=k, n_txt_target=nt, rep=rep,
                               n_vis=r["n_vis"], n_txt=r["n_txt"],
                               tpot=(tt[-1] - tt[0]) / max(1, len(tt) - 1),
                               energy_j=pw["energy_j"], mean_w=pw["mean_w"])
                    for key in ("t_pre", "t_enc", "t_img_pf", "t_txt_pf", "ttft"):
                        row[key] = r[key]
                    for n in n_list:
                        row[f"tta_{n}"] = tt[n - 1]
                        row[f"energy_{n}"] = pm.integrate_window(r["t_start"], r["t_start"] + tt[n - 1])["energy_j"]
                    for key in fields:
                        if key.startswith(("t_", "tta_")) or key in ("ttft", "tpot"):
                            row[key] = round(1000 * row[key], 3)  # ms
                    wr.writerow(row)
                    fh.flush()
                    done += 1
                print(f"[{done}/{total}] cpu={ckey} gpu={gkey} (mean {cm}/{gm}) {w}x{h} x{k} txt={nt}: "
                      f"n_vis={r['n_vis']} enc={1000*r['t_enc']:.0f} img_pf={1000*r['t_img_pf']:.0f} "
                      f"txt_pf={1000*r['t_txt_pf']:.0f} TTFT={1000*r['ttft']:.0f} ms "
                      f"TPOT={row['tpot']:.1f} ms", flush=True)
    finally:
        if fh:
            fh.close()
        pm.stop()
        fs.stop()
        if use_dvfs:
            dvfs.restore_original_state()
    print(f"wrote {out}" + (f" and {idle_out}" if a.idle_s > 0 else ""))


if __name__ == "__main__":
    main()
