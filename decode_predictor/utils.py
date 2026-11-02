"""
utils
==========
Reproducibility seeds, metric helpers, and logging utilities.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "decodepredictor") -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for full reproducibility.

    Sets Python ``random``, NumPy, PyTorch (CPU + CUDA), and enables
    cuDNN determinism.  Also sets the `CUBLAS_WORKSPACE_CONFIG` environment
    variable required by PyTorch ≥ 1.11 in deterministic mode.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE for length prediction.

    Parameters
    ----------
    preds:   np.ndarray of shape (N,)
    targets: np.ndarray of shape (N,)

    Returns
    -------
    dict with keys: mae
    """
    mae = float(np.mean(np.abs(preds - targets)))
    return {"mae": mae}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create *path* and all parents if they do not exist; return a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_jsonl(records: list[Dict[str, Any]], filepath: str | Path) -> None:
    """Append a list of records to a JSON-Lines file (one JSON object per line)."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with filepath.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str | Path) -> list[Dict[str, Any]]:
    """Load all records from a JSON-Lines file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    records = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    """Convert a string dtype name to a ``torch.dtype``."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str!r}. Choose from {list(mapping)}.")
    return mapping[dtype_str]
