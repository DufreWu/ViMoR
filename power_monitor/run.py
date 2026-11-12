from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from power_monitor.controllers.dvfs_controller import DvfsController
from power_monitor.experiments.interference import (
    check_dependencies,
    load_experiment_config,
    run_sweep,
    run_workload,
    validate_experiment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workload-combination (W0-W7) DVFS benchmark runner")
    parser.add_argument("--check", action="store_true", help="Validate configuration and hardware without changing frequencies")
    parser.add_argument("--workload", type=str, help="Run one workload, e.g. W5")
    parser.add_argument("--cpu-mhz", type=float, default=None, help="CPU frequency in MHz (snapped to nearest supported)")
    parser.add_argument("--gpu-mhz", type=float, default=None, help="GPU frequency in MHz (snapped to nearest supported)")
    parser.add_argument("--sweep", action="store_true", help="Run workloads x DVFS matrix x repetitions")
    parser.add_argument("--workloads", type=str, default=None, help="Comma-separated subset for --sweep, e.g. W1,W3,W5")
    parser.add_argument("--no-dvfs", action="store_true", help="Do not touch clocks (no root needed); record current clocks")
    parser.add_argument("--governor", choices=["default"], default=None,
                        help="default: stock dynamic governors (dvfs.default_governor, full frequency range). "
                             "Alone: run/sweep only this mode; with --sweep --dvfs: added as an extra point next to the fixed CPU/GPU matrix. Needs root")
    parser.add_argument("--grid", choices=["coarse", "full"], default="coarse",
                        help="With --sweep --dvfs: coarse = dvfs.cpu_freq_mhz x gpu_freq_mhz (3x3); "
                             "full = all jetson.cpu_available_frequencies_hz x gpu_available_frequencies_hz combinations (12x7 = 84)")
    parser.add_argument("--dvfs", action="store_true", help="Force DVFS control on (overrides dvfs_control: false in the config; needs root)")
    parser.add_argument("--measurement-seconds", type=float, default=None, help="Override measurement_seconds")
    parser.add_argument("--warmup-seconds", type=float, default=None, help="Override warmup_seconds")
    parser.add_argument("--repetitions", "--repeat", dest="repetitions", type=int, default=None,
                        help="Number of repetitions (default: `repetitions` in experiments.yaml). With --sweep: per DVFS point; "
                             "with --workload: repeat that one configuration N times (as a 1-point sweep with a plan, resumable)")
    parser.add_argument("--config", type=str, default=None, help="Path to the experiment config file")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic camera/models/power; exercises the full pipeline")
    parser.add_argument("--resume", type=str, default=None, metavar="SWEEP_ID", help="Continue an interrupted --sweep from its plan.csv")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume: also re-run entries that failed")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate runs_index.csv into per-(workload, CPU, GPU) statistics and plots")
    parser.add_argument("--sweep-id", type=str, default=None, help="With --aggregate: only runs of this sweep")
    parser.add_argument("--include-flagged", action="store_true", help="With --aggregate: keep runs with valid=false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        args.sweep = True
    if args.resume and args.repetitions is not None:
        print("WARNING: --repetitions is ignored with --resume (the saved plan fixes the repetitions)")
        args.repetitions = None
    if args.repetitions is not None and args.repetitions < 1:
        print("ERROR: --repetitions must be >= 1")
        return 1
    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "configs" / "experiments.yaml"
    cfg = load_experiment_config(config_path)
    if args.aggregate:
        from power_monitor.analysis.aggregate import aggregate_runs

        out = aggregate_runs(Path(cfg.get("output_root", "outputs/power_monitor")), sweep_id=args.sweep_id, include_flagged=args.include_flagged)
        print(f"Aggregates written to {out}")
        return 0
    for key in ("measurement_seconds", "warmup_seconds", "repetitions"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    validation = validate_experiment_config(cfg)
    print(f"Configuration valid: {validation['valid']}")
    for error in validation["errors"]:
        print(f"ERROR: {error}")
    for warning in validation["warnings"]:
        print(f"WARNING: {warning}")

    controller = DvfsController()
    env_report = controller.check_environment()
    print(f"Environment valid: {env_report['valid']}")
    for issue in env_report.get("issues", []):
        print(f"ENV ISSUE: {issue}")
    for warning in env_report.get("warnings", []):
        print(f"ENV WARNING: {warning}")

    if args.check:
        return 0 if validation["valid"] and env_report["valid"] else 1
    if not validation["valid"]:
        return 1

    selected = args.workloads.split(",") if (args.sweep and args.workloads) else ([args.workload] if args.workload else None)
    if args.sweep or args.workload:
        dep_errors = check_dependencies(cfg, selected, dry_run=args.dry_run)
        for error in dep_errors:
            print(f"ERROR: {error}")
        if dep_errors:
            return 1

    if args.dvfs and args.no_dvfs:
        print("ERROR: --dvfs and --no-dvfs are mutually exclusive")
        return 1
    governor_default = args.governor == "default"
    if governor_default and args.no_dvfs:
        print("ERROR: --governor default changes the governors; it cannot be combined with --no-dvfs")
        return 1
    if governor_default and args.workload and (args.cpu_mhz is not None or args.gpu_mhz is not None):
        print("ERROR: --governor default and --cpu-mhz/--gpu-mhz are mutually exclusive for a single run")
        return 1
    if args.workload and (args.cpu_mhz is not None or args.gpu_mhz is not None) and not args.no_dvfs:
        args.dvfs = True  # explicit clocks on the command line imply DVFS control
    if not args.dvfs and not cfg.get("dvfs_control", True):
        args.no_dvfs = True  # config pins clocks to the current governor
    if args.grid != "coarse" and args.no_dvfs:
        print(f"ERROR: --grid {args.grid} selects fixed CPU/GPU points; add --dvfs")
        return 1
    mode = f"fixed CPU/GPU matrix ({args.grid} grid)" if not args.no_dvfs else "off (current governor)"
    if governor_default:
        mode = (mode + " + default governor") if not args.no_dvfs else "default governor only"
    print(f"DVFS control: {mode}")
    needs_root = (not args.no_dvfs) or governor_default
    if (args.sweep or args.workload) and needs_root and not args.dry_run and os.geteuid() != 0:
        print("ERROR: DVFS control writes CPU/GPU sysfs clocks and needs root. Run with sudo -E (keeps your ~/.local packages visible):")
        print("  sudo -E python -m power_monitor.run ...")
        return 1

    if args.sweep:
        workloads = args.workloads.split(",") if args.workloads else None
        results = run_sweep(cfg, workloads=workloads, dry_run=args.dry_run, no_dvfs=args.no_dvfs, resume=args.resume, retry_failed=args.retry_failed,
                            governor_default=governor_default, grid=args.grid)
        failed = [r for r in results if r.get("status") == "failed"]
        print(f"Sweep done: {len(results) - len(failed)} completed, {len(failed)} failed")
        return 1 if failed else 0

    if args.workload:
        if governor_default:
            cpu_mhz = gpu_mhz = "default"
        elif args.no_dvfs:
            cpu_mhz = gpu_mhz = None
        else:
            cpu_mhz = args.cpu_mhz if args.cpu_mhz is not None else 1190
            gpu_mhz = args.gpu_mhz if args.gpu_mhz is not None else 612
        if args.repetitions is None or args.repetitions == 1:
            result = run_workload(args.workload, cpu_mhz, gpu_mhz, repetition=1, config=cfg, dry_run=args.dry_run)
            print(json.dumps(result, indent=2, default=str))
            return 0 if result.get("valid") else 2
        results = run_sweep(cfg, workloads=[args.workload], dry_run=args.dry_run, points=[(cpu_mhz, gpu_mhz)])
        failed = [r for r in results if r.get("status") == "failed"]
        invalid = [r for r in results if r.get("status") != "failed" and not r.get("valid")]
        print(f"Done: {len(results) - len(failed)} completed ({len(invalid)} flagged), {len(failed)} failed")
        return 1 if failed else (2 if invalid else 0)

    print("No action requested. Use --check, --workload, or --sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
