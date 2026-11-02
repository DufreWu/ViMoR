"""
trainer
============
Trainer – training loop for the soft-label regression head.

The trainer operates on pre-extracted feature tensors (from
:class:`feature_extractor.FeatureExtractor`) and handles:

  * Mini-batch training with AdamW.
  * Combined loss: λ·KL(log_softmax(logits)‖soft_labels) + (1-λ)·MSE(pred, y).
  * Mini-batch training with AdamW.
  * Combined loss: λ·KL(log_softmax(logits)‖soft_labels) + (1-λ)·MSE(pred, y).
  * In-memory history recording for downstream plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PredictorHead
from utils import compute_metrics, ensure_dir, get_logger

logger = get_logger(__name__)


class DecodeLenPredictorTrainer:
    """
    Trains an :class:`~model.PredictorHead` given pre-computed feature tensors.

    Parameters
    ----------
    head : PredictorHead
        The (freshly initialised) prediction head to train.
    device : torch.device
        Target computation device (head must already be on this device).
    lr : float
        AdamW learning rate. Default: 2e-5.
    weight_decay : float
        AdamW weight decay. Default: 0.0.
    lambda_val : float
        Loss mixing coefficient (weight on the KL term). Default: 0.95.
    batch_size : int
        Mini-batch size for both training and validation. Default: 256.
    epochs : int
        Maximum number of training epochs. Default: 200.
    checkpoint_dir : str | Path | None
        Directory to save ``final_head.pt``. If ``None``, no checkpoint is saved.
    """

    def __init__(
        self,
        head: PredictorHead,
        device: torch.device,
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        lambda_val: float = 0.95,
        batch_size: int = 256,
        epochs: int = 200,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.head = head
        self.device = device
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> float:
        """Run one epoch and return the mean training Loss."""
        self.head.train()
        perm = torch.randperm(len(train_x), device=self.device)
        running_loss = 0.0

        for i in range(0, len(train_x), self.batch_size):
            idx = perm[i : i + self.batch_size]
            bx, by = train_x[idx], train_y[idx]

            self.optimizer.zero_grad()
            logits, pred = self.head(bx)
            soft = self.head.soft_labels(by)

            loss_kl = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(logits, dim=-1), soft
            )
            loss_mse = F.mse_loss(pred, by)
            loss = self.lambda_val * loss_kl + (1 - self.lambda_val) * loss_mse
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * bx.size(0)

        return running_loss / len(train_x)

    @torch.no_grad()
    def _eval_split(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """Evaluate and return mean MAE on a given split."""
        self.head.eval()
        total_abs_err = 0.0
        for i in range(0, len(x), self.batch_size):
            bx, by = x[i : i + self.batch_size], y[i : i + self.batch_size]
            _, pred = self.head(bx)
            total_abs_err += torch.abs(pred - by).sum().item()
        return total_abs_err / len(x)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Train the head and return training history.

        Parameters
        ----------
        train_x : torch.Tensor  – shape (N, d) on ``self.device``.
        train_y : torch.Tensor  – shape (N,)  on ``self.device``.

        Returns
        -------
        history : dict with keys ``"train_loss"`` and ``"final_loss"``.
        """
        train_loss_history: List[float] = []

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_x, train_y)
            train_loss_history.append(train_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch [{epoch+1:>4}/{self.epochs}] "
                    f"train Loss={train_loss:.4f}"
                )

        # Optionally persist final checkpoint
        if self.checkpoint_dir is not None:
            ensure_dir(self.checkpoint_dir)
            ckpt_path = self.checkpoint_dir / "final_head.pt"
            torch.save(self.head.state_dict(), ckpt_path)
            logger.info(f"  Saved final checkpoint → {ckpt_path}")

        return {
            "train_loss": train_loss_history,
            "final_loss": train_loss_history[-1] if train_loss_history else float("nan"),
            "epochs": self.epochs,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference using the trained head.

        Parameters
        ----------
        x : torch.Tensor of shape (N, d) on ``self.device``.

        Returns
        -------
        preds : torch.Tensor of shape (N,) on CPU, float32.
        """
        self.head.eval()
        preds_list: List[torch.Tensor] = []
        for i in range(0, len(x), self.batch_size):
            _, batch_pred = self.head(x[i : i + self.batch_size])
            preds_list.append(batch_pred.cpu())
        return torch.cat(preds_list, dim=0)
