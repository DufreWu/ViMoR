#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import pandas as pd

# Ensure the EGTP package is importable when running from this directory
sys.path.insert(0, str(Path(__file__).parent))

from predictor import DecodeLenPredictor
from utils import ensure_dir, get_logger, load_jsonl, set_seed

logger = get_logger("decode.predictor")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entropy-Guided Token Pooling for LLM output length prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--eval_all",
        action="store_true",
        help="Sweep all model X scenario directories under --final_data_root.",
    )

    # ── Single-run data ───────────────────────────────────────────────────
    sg = p.add_argument_group("Single-run arguments (used when --eval_all is NOT set)")
    sg.add_argument("--data_dir", type=str, default=None,
                    help="Path to directory containing train.csv and test.csv.")
    sg.add_argument("--model_id", type=str, default=None,
                    help="HuggingFace model ID or local path for the backbone LLM.")

    # ── Full-benchmark ────────────────────────────────────────────────────
    bg = p.add_argument_group("Full-benchmark arguments (used when --eval_all IS set)")
    bg.add_argument("--final_data_root", type=str, default="../final_data",
                    help="Root directory containing model-level sub-directories.")
    bg.add_argument("--model_map", type=str, default=None,
                    help=(
                        "JSON file mapping model dir-name → HuggingFace model ID. "
                        "Defaults to the canonical ForeLen map when not provided."
                    ))
    bg.add_argument("--scenarios", nargs="+", default=["LongSeq", "Reasoning", "RL"],
                    help="Scenario names to evaluate.")
    bg.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip (model, scenario) pairs already in results.jsonl.")

    # ── Shared hyper-parameters ───────────────────────────────────────────
    hg = p.add_argument_group("Hyper-parameters")
    hg.add_argument("--prompt_prefix_k", type=int, default=4,
                    help="Number of prefix tokens for entropy-guided pooling.")
    hg.add_argument("--num_bins", type=int, default=20,
                    help="Number of length bins in the soft-label head.")
    hg.add_argument("--lambda_val", type=float, default=0.95,
                    help="KL-vs-MSE loss mixing coefficient.")
    hg.add_argument("--epochs", type=int, default=200,
                    help="Maximum training epochs.")
    hg.add_argument("--batch_size", type=int, default=256,
                    help="Training / prediction mini-batch size.")
    hg.add_argument("--lr", type=float, default=2e-5, help="AdamW learning rate.")
    hg.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay.")
    hg.add_argument("--patience", type=int, default=200,
                    help="Early-stopping patience (epochs). Set to --epochs to disable.")
    hg.add_argument("--extractor_batch_size", type=int, default=64,
                    help="Batch size for backbone feature extraction.")
    hg.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Backbone weight precision.")
    hg.add_argument("--seed", type=int, default=42, help="Global random seed.")

    # ── Config file (optional) ────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help=(
                       "Path to a YAML config file. Values are merged on top of "
                       "CLI defaults. Model-family-specific configs are applied "
                       "automatically in --eval_all mode (llama.yaml / qwen.yaml)."
                   ))

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="outputs/run",
                   help="Directory to save all output artefacts.")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Default model map (ForeLen benchmark)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_MAP = {
   "qwen2.5_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
   "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
   "qwen2.5_3b":   "Qwen/Qwen2.5-3B-Instruct",
   "qwen2.5_7b":   "Qwen/Qwen2.5-7B-Instruct",
   "llama3.2_1b":  "meta-llama/Llama-3.2-1B-Instruct",
   "llama3.2_3b":  "meta-llama/Llama-3.2-3B-Instruct",
   "deepseekr1_distill_qwen_1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
   "deepseekr1_distill_qwen_7b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CONFIGS_DIR = Path(__file__).parent / "configs"

# Mapping: keyword in model dir-name → family config YAML
_FAMILY_CONFIG_MAP = {
    "llama": _CONFIGS_DIR / "llama.yaml",   # 200 epochs
    "qwen":  _CONFIGS_DIR / "qwen.yaml",    # 90 epochs
}


def resolve_model_config(model_name: str) -> dict:
    """
    Return training hyper-parameters for *model_name* by looking up the
    model-family-specific YAML (llama.yaml or qwen.yaml), falling back to
    default.yaml if no family matches.

    Only epoch-relevant keys are returned; CLI flags take precedence when
    explicitly set by the user.
    """
    model_lower = model_name.lower()
    for keyword, cfg_path in _FAMILY_CONFIG_MAP.items():
        if keyword in model_lower and cfg_path.exists():
            with cfg_path.open() as f:
                return yaml.safe_load(f)
    default_cfg = _CONFIGS_DIR / "default.yaml"
    if default_cfg.exists():
        with default_cfg.open() as f:
            return yaml.safe_load(f)
    return {}


def load_config_file(path: str | None) -> dict:
    """Load a YAML config file (returns empty dict if path is None)."""
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_predictor(
    model_id: str,
    output_dir: Path,
    args: argparse.Namespace,
    experiment_id: str | None = None,
    config_overrides: dict | None = None,
) -> DecodeLenPredictor:
    """
    Build an DecodeLenPredictor, merging (in priority order):
      1. model-family YAML defaults
      2. explicit --config YAML overrides
      3. CLI flag values (highest priority)
    """
    cfg = config_overrides or {}
    # Use config values as fallbacks only when the CLI arg matches its default
    epochs   = cfg.get("epochs",   args.epochs)
    return DecodeLenPredictor(
        model_id=model_id,
        output_dir=output_dir,
        prompt_prefix_k=args.prompt_prefix_k,
        num_bins=args.num_bins,
        lambda_val=args.lambda_val,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        extractor_batch_size=args.extractor_batch_size,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        experiment_id=experiment_id,
    )


def save_summary(records: list[dict], output_dir: Path) -> None:
    """Write a human-readable Markdown summary table of all results."""
    rows = []
    for rec in records:
        cfg = rec.get("config", {})
        res = rec.get("results", {})
        rows.append({
            "Model": Path(cfg.get("data_dir", "")).parent.name,
            "Scenario": Path(cfg.get("data_dir", "")).name,
            "k": cfg.get("prompt_prefix_k", "-"),
            "Test MAE": round(res.get("mae", float("nan")), 2),
        })
    df = pd.DataFrame(rows)
    md_path = output_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    logger.info(f"Summary table → {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:  
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)

    if args.eval_all:
        # ── Full-benchmark mode ────────────────────────────────────────────
        final_data_root = Path(args.final_data_root)
        if not final_data_root.exists():
            logger.error(f"final_data_root does not exist: {final_data_root}")
            sys.exit(1)

        # Load model map
        if args.model_map:
            with open(args.model_map) as f:
                model_map = json.load(f)
        else:
            model_map = DEFAULT_MODEL_MAP

        # Load already-evaluated pairs (for --skip_existing)
        evaluated_pairs: set[tuple[str, str]] = set()
        if args.skip_existing:
            for rec in load_jsonl(output_dir / "results.jsonl"):
                cfg = rec.get("config", {})
                dd = Path(cfg.get("data_dir", ""))
                evaluated_pairs.add((dd.parent.name, dd.name))

        all_records: list[dict] = []

        for model_name in sorted(os.listdir(final_data_root)):
            model_dir = final_data_root / model_name
            if not model_dir.is_dir():
                continue
            model_id = model_map.get(model_name)
            if not model_id:
                logger.warning(f"No model ID mapping for '{model_name}', skipping.")
                continue

            for scenario in args.scenarios:
                scenario_dir = model_dir / scenario
                if not scenario_dir.is_dir():
                    continue

                if (model_name, scenario) in evaluated_pairs:
                    logger.info(f"Skipping {model_name}/{scenario} (already evaluated).")
                    continue

                exp_id = f"{model_name}_{scenario}_k{args.prompt_prefix_k}"
                run_out = output_dir / model_name / scenario
                # Auto-resolve model-family config (llama→200 ep, qwen→90 ep)
                family_cfg = resolve_model_config(model_name)
                predictor = build_predictor(
                    model_id, run_out, args,
                    experiment_id=exp_id,
                    config_overrides=family_cfg,
                )

                try:
                    record = predictor.run(scenario_dir)
                    all_records.append(record)
                except Exception as exc:
                    logger.error(f"Failed {model_name}/{scenario}: {exc}", exc_info=True)

        # Reload full JSONL for summary
        all_records = load_jsonl(output_dir / "results.jsonl")
        if all_records:
            save_summary(all_records, output_dir)
        logger.info("Full-benchmark evaluation complete.")

    else:
        # ── Single-run mode ────────────────────────────────────────────────
        if not args.data_dir or not args.model_id:
            parser.error("--data_dir and --model_id are required in single-run mode.")

        # Load optional --config file for single-run mode
        cli_cfg = load_config_file(args.config)
        predictor = build_predictor(args.model_id, output_dir, args,
                                    config_overrides=cli_cfg)
        record = predictor.run(args.data_dir)

        logger.info(
            f"\nFinal Test   MAE : {record['results']['mae']:.4f}\n"
            f"Final Train Loss : {record['results']['final_train_loss']:.4f}\n"
        )
        save_summary([record], output_dir)


if __name__ == "__main__":
    main()

# Example command for single-run mode:
# python decode_predictor/main.py --model_id Qwen/Qwen2.5-3B-Instruct --data_dir decode_predictor/datasets/ --lambda_val 0.95 --batch_size 64 --output_dir decode_predictor/outputs
