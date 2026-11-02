"""
predictor
==============
Predictor – high-level end-to-end pipeline.

This class orchestrates the full DecodeLenPredictor pipeline for a single
(model, scenario) combination:

  1. Load the CSV data splits (train / test).
  2. Extract entropy-guided features via :class:`feature_extractor.FeatureExtractor`.
  3. Determine the target length range from the training set.
  4. Initialise the :class:`model.PredictorHead`.
  5. Train using :class:`trainer.Trainer`.
  6. Evaluate on the test set and log all metrics + per-sample predictions.

All artefacts (predictions CSV, checkpoint, results JSONL) are written to
``output_dir``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from feature_extractor import TokenPoolingFeatureExtractor
from model import PredictorHead
from trainer import DecodeLenPredictorTrainer
from utils import compute_metrics, ensure_dir, get_logger, save_jsonl

logger = get_logger(__name__)


class DecodeLenPredictor:
    """
    End-to-end Decode Length predictor for one dataset split (train / val / test).

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the backbone LLM.
    output_dir : str | Path
        Directory to save all output artefacts.
    prompt_prefix_k : int
        Number of prefix tokens for entropy-guided pooling. Default: 4.
    num_bins : int
        Number of length bins in the soft-label head. Default: 20.
    lambda_val : float
        KL-vs-MSE mixing weight. Default: 0.95.
    epochs : int
        Maximum training epochs. Default: 200.
    batch_size : int
        Training / inference batch size. Default: 256.
    lr : float
        AdamW learning rate. Default: 2e-5.
    extractor_batch_size : int
        Batch size for feature extraction. Default: 64.
    torch_dtype : str
        Backbone weight dtype. Default: ``"bfloat16"``.
    seed : int
        Fixed for reproducibility. Default: 42.
    experiment_id : str | None
        Human-readable identifier logged in the JSONL record. Auto-generated
        from model/scenario if ``None``.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str | Path,
        prompt_prefix_k: int = 4,
        num_bins: int = 20,
        lambda_val: float = 0.95,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        extractor_batch_size: int = 64,
        torch_dtype: str = "bfloat16",
        seed: int = 42,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.prompt_prefix_k = prompt_prefix_k
        self.num_bins = num_bins
        self.lambda_val = lambda_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.extractor_batch_size = extractor_batch_size
        self.torch_dtype = torch_dtype
        self.seed = seed
        self.experiment_id = experiment_id

    # ──────────────────────────────────────────────────────────────────────
    # Data helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_split(data_dir: Path, test_size: float = 0.2, random_state: int = 42):
        jsonl_path = data_dir / "decode_dataset.jsonl"  
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Expected {jsonl_path} but file does not exist.")
        
        df = pd.read_json(jsonl_path, lines=True)

        # Check required columns
        for col in ("prompt", "decode_length"):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {jsonl_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Split dataset
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self, data_dir: str | Path) -> Dict[str, Any]:
        """
        Run the full pipeline on a dataset split directory.

        Parameters
        ----------
        data_dir : str | Path
            Path to the directory with CSV splits.

        Returns
        -------
        record : dict
            Experiment record that is also appended to ``output_dir/results.jsonl``.
        """
        from utils import set_seed
        set_seed(self.seed)

        data_dir = Path(data_dir)
        ensure_dir(self.output_dir)

        experiment_id = self.experiment_id or (
            f"{data_dir.parent.name}_{data_dir.name}"
            f"_k{self.prompt_prefix_k}"
            f"_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        logger.info(f"\n{'='*60}")
        logger.info(f" Experiment : {experiment_id}")
        logger.info(f" Data dir   : {data_dir}")
        logger.info(f" Model      : {self.model_id}")
        logger.info(f"{'='*60}\n")

        # ── 1. Load splits ────────────────────────────────────────────────
        logger.info("Loading CSV splits …")
        train_df, test_df = self._load_split(data_dir)
        logger.info(
            f"  train={len(train_df):,}  test={len(test_df):,}"
        )

        # ── 2. Feature extraction ─────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = TokenPoolingFeatureExtractor(
            model_id=self.model_id,
            prompt_prefix_k=self.prompt_prefix_k,
            batch_size=self.extractor_batch_size,
            device=device,
            torch_dtype=self.torch_dtype,
        )
        extractor.load_backbone()

        logger.info("Extracting features for train split …")
        train_x = extractor.extract(train_df["prompt"].tolist())
        logger.info("Extracting features for test split …")
        test_x = extractor.extract(test_df["prompt"].tolist())

        feature_dim = train_x.shape[1]
        print(f"  Extracted feature dimension: {feature_dim}")
        extractor.unload_backbone()  # free GPU memory before training

        # ── 3. Move tensors to device ─────────────────────────────────────
        train_x = train_x.to(device)
        test_x = test_x.to(device)

        train_y = torch.tensor(train_df["decode_length"].values, dtype=torch.float32).to(device)
        test_y = torch.tensor(test_df["decode_length"].values, dtype=torch.float32).to(device)

        # ── 4. Determine target range from training set ───────────────────
        train_y_np = train_df["decode_length"].values
        p1, p99 = float(np.quantile(train_y_np, 0.01)), float(np.quantile(train_y_np, 0.99))
        target_range = (p1, p99) if p99 > p1 else (p1, p1 + 1.0)
        logger.info(f"  Target range (p1-p99): [{target_range[0]:.1f}, {target_range[1]:.1f}]")

        # ── 5. Initialise head & trainer ──────────────────────────────────
        head = PredictorHead(
            input_dim=feature_dim,
            num_bins=self.num_bins,
            target_range=target_range,
        ).to(device)

        ckpt_dir = self.output_dir / "checkpoints"
        trainer = DecodeLenPredictorTrainer(
            head=head,
            device=device,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lambda_val=self.lambda_val,
            batch_size=self.batch_size,
            epochs=self.epochs,
            checkpoint_dir=ckpt_dir,
        )

        # ── 6. Train ──────────────────────────────────────────────────────
        logger.info("Training DecodeLenPredictor head …")
        history = trainer.fit(train_x, train_y)

        # ── 7. Evaluate on test set ───────────────────────────────────────
        test_preds = trainer.predict(test_x).numpy()
        test_y_np = test_y.cpu().numpy()
        metrics = compute_metrics(test_preds, test_y_np)

        logger.info(
            f"  Test → MAE={metrics['mae']:.4f}"
        )

        # ── 8. Save per-sample predictions ───────────────────────────────
        pred_df = pd.DataFrame({
            "sample_idx": np.arange(len(test_preds)),
            "pred_length": test_preds,
            "gt_length": test_y_np,
            "abs_error": np.abs(test_preds - test_y_np),
        })
        pred_csv_path = self.output_dir / "test_predictions.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        logger.info(f"  Per-sample predictions → {pred_csv_path}")

        # ── 9. Build & save experiment record ─────────────────────────────
        config = {
            "model_id": self.model_id,
            "data_dir": str(data_dir),
            "prompt_prefix_k": self.prompt_prefix_k,
            "num_bins": self.num_bins,
            "lambda_val": self.lambda_val,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "target_range": list(target_range),
            "feature_dim": feature_dim,
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
        record: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "config": config,
            "results": {
                **{k: round(v, 6) for k, v in metrics.items()},
                "final_train_loss": round(history["final_loss"], 6),
            },
            "train_loss_history": [round(v, 4) for v in history["train_loss"]],
            "predictions_csv": str(pred_csv_path),
        }

        jsonl_path = self.output_dir / "results.jsonl"
        save_jsonl([record], jsonl_path)
        logger.info(f"  Experiment record → {jsonl_path}")

        return record
