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
    parser.add_argument("--repeat", type=int, default=1, help="Repetition index for a single run")
    parser.add_argument("--sweep", action="store_true", help="Run workloads x DVFS matrix x repetitions")
    parser.add_argument("--workloads", type=str, default=None, help="Comma-separated subset for --sweep, e.g. W1,W3,W5")
    parser.add_argument("--no-dvfs", action="store_true", help="Do not touch clocks (no root needed); record current clocks")
    parser.add_argument("--dvfs", action="store_true", help="Force DVFS control on (overrides dvfs_control: false in the config; needs root)")
    parser.add_argument("--measurement-seconds", type=float, default=None, help="Override measurement_seconds")
    parser.add_argument("--warmup-seconds", type=float, default=None, help="Override warmup_seconds")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetitions for --sweep")
    parser.add_argument("--config", type=str, default=None, help="Path to the experiment config file")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic camera/models/power; exercises the full pipeline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "configs" / "experiments.yaml"
    cfg = load_experiment_config(config_path)
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
    if args.workload and (args.cpu_mhz is not None or args.gpu_mhz is not None) and not args.no_dvfs:
        args.dvfs = True  # explicit clocks on the command line imply DVFS control
    if not args.dvfs and not cfg.get("dvfs_control", True):
        args.no_dvfs = True  # config pins clocks to the current governor
    print(f"DVFS control: {'off (current governor)' if args.no_dvfs else 'on'}")
    if (args.sweep or args.workload) and not args.no_dvfs and not args.dry_run and os.geteuid() != 0:
        print("ERROR: DVFS control writes CPU/GPU sysfs clocks and needs root. Run with sudo -E (keeps your ~/.local packages visible):")
        print("  sudo -E python -m power_monitor.run ...")
        return 1

    if args.sweep:
        workloads = args.workloads.split(",") if args.workloads else None
        results = run_sweep(cfg, workloads=workloads, dry_run=args.dry_run, no_dvfs=args.no_dvfs)
        failed = [r for r in results if r.get("status") == "failed"]
        print(f"Sweep done: {len(results) - len(failed)} completed, {len(failed)} failed")
        return 1 if failed else 0

    if args.workload:
        if args.no_dvfs:
            cpu_mhz = gpu_mhz = None
        else:
            cpu_mhz = args.cpu_mhz if args.cpu_mhz is not None else 1190
            gpu_mhz = args.gpu_mhz if args.gpu_mhz is not None else 612
        result = run_workload(args.workload, cpu_mhz, gpu_mhz, repetition=args.repeat, config=cfg, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, default=str))
        return 0

    print("No action requested. Use --check, --workload, or --sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
