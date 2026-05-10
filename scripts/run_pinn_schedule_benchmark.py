#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.pinn_benchmark import (
    PINNBenchmarkConfig,
    available_pinn_schedule_names,
    normalize_pinn_schedule_names,
    pinn_evaluation_report_to_dict,
    run_pinn_evaluation_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run schedule benchmarking for the Poisson PINN derivative workload "
            "with runtime, IR, numerical-error, and optional warmup auto-selection metrics."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("tanh",), default="tanh")
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--target-error", type=float, default=1e-4)

    parser.add_argument(
        "--schedule",
        action="append",
        choices=available_pinn_schedule_names(),
        dest="schedules",
        help="Fixed PINN schedule baseline to benchmark. Repeat for multiple.",
    )
    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="Run warmup-based automatic schedule selection and compare vs fixed baselines.",
    )
    parser.add_argument(
        "--auto-candidate",
        action="append",
        choices=available_pinn_schedule_names(),
        dest="auto_candidates",
        help="Candidate schedule considered by auto tuner. Repeat for multiple.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of warmup executions profiled per candidate schedule.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=None,
        help="Optional memory budget used to reject candidates during warmup.",
    )
    parser.add_argument(
        "--warmup-loss-tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance for the warmup numerical-error guard.",
    )
    parser.add_argument(
        "--disable-warmup-cache",
        action="store_true",
        help="Disable warmup selection cache.",
    )
    parser.add_argument(
        "--warmup-cache-path",
        type=str,
        default=".adscheduler_pinn_warmup_cache.json",
        help="Path to warmup selection cache file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON after the text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_baselines = normalize_pinn_schedule_names(args.schedules)
    selected_auto_candidates = (
        normalize_pinn_schedule_names(args.auto_candidates)
        if args.auto_candidates
        else selected_baselines
    )

    config = PINNBenchmarkConfig(
        seed=args.seed,
        outer_steps=args.outer_steps,
        eval_every=args.eval_every,
        grid_size=args.grid_size,
        input_dim=args.input_dim,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        output_dim=args.output_dim,
        target_max_abs_error=args.target_error,
    )

    report = run_pinn_evaluation_protocol(
        config,
        baseline_schedules=selected_baselines,
        include_auto=args.include_auto,
        auto_candidate_schedules=selected_auto_candidates,
        warmup_steps=args.warmup_steps,
        memory_budget_mb=args.memory_budget_mb,
        error_guard_tolerance=args.warmup_loss_tolerance,
        use_cache=not args.disable_warmup_cache,
        cache_path=args.warmup_cache_path,
    )

    print("=== PINN Benchmark Summary ===")
    print(f"fixed_schedules: {', '.join(selected_baselines)}")
    print(f"outer_steps: {args.outer_steps} eval_every: {args.eval_every}")
    print(
        "workload: "
        f"grid_size={args.grid_size} input_dim={args.input_dim} "
        f"hidden_layers={args.hidden_layers} hidden_dim={args.hidden_dim} "
        f"activation={args.activation} output_dim={args.output_dim}"
    )
    print(f"target_max_abs_error: {args.target_error}")
    if args.include_auto:
        print(f"auto_candidates: {', '.join(selected_auto_candidates)}")
        print(
            "warmup: "
            f"steps={args.warmup_steps} "
            f"error_tolerance={args.warmup_loss_tolerance} "
            f"cache={'off' if args.disable_warmup_cache else args.warmup_cache_path}"
        )
    print()

    print("=== Fixed Schedule Results ===")
    for result in report.fixed_results:
        _print_result(result)

    if report.auto_result is not None:
        auto = report.auto_result
        warmup = report.warmup_selection
        print("=== Auto Selection Result ===")
        print(f"  selected_schedule: {auto.selected_schedule_name}")
        print(f"  used_warmup_cache: {auto.used_warmup_cache}")
        print(f"  warmup_overhead_ms: {auto.warmup_overhead_sec * 1e3:.3f}")
        _print_result(auto, indent="  ")

        if warmup is not None:
            print("=== Warmup Profiles ===")
            print(f"  from_cache: {warmup.from_cache}")
            print(f"  cache_signature: {warmup.cache_signature}")
            if warmup.profiles:
                for profile in warmup.profiles:
                    status = profile.rejected_reason or "accepted"
                    peak_mem = profile.peak_device_memory_mb or profile.peak_host_memory_mb
                    print(
                        f"  - {profile.schedule_name}: score={profile.score:.6f} "
                        f"compile_ms={profile.compile_overhead_sec * 1e3:.3f} "
                        f"median_ms={profile.median_step_time_sec * 1e3:.3f} "
                        f"p90_ms={profile.p90_step_time_sec * 1e3:.3f} "
                        f"peak_mem_mb={peak_mem:.2f} "
                        f"final_error={profile.final_max_abs_error:.6e} "
                        f"status={status}"
                    )
            print()

    if report.selection_quality is not None:
        quality = report.selection_quality
        print("=== Selection Quality (Auto vs Oracle Fixed) ===")
        print(f"  oracle_schedule: {quality.oracle_schedule_name}")
        print(f"  oracle_estimated_runtime_sec: {quality.oracle_estimated_runtime_sec:.6f}")
        print(f"  auto_selected_schedule: {quality.auto_selected_schedule_name}")
        print(f"  auto_estimated_runtime_sec: {quality.auto_estimated_runtime_sec:.6f}")
        print(f"  runtime_regret_sec: {quality.runtime_regret_sec:.6f}")
        print(f"  runtime_regret_pct: {quality.runtime_regret_pct:.3f}")
        print()

    if args.json:
        print("=== PINN Benchmark JSON ===")
        print(json.dumps(pinn_evaluation_report_to_dict(report), indent=2))


def _print_result(result, *, indent: str = "") -> None:
    iters_to_target = (
        str(result.iterations_to_target_error)
        if result.iterations_to_target_error is not None
        else "not reached"
    )
    peak_device = (
        f"{result.peak_device_memory_mb:.2f}"
        if result.peak_device_memory_mb is not None
        else "n/a"
    )
    if not indent:
        print(f"[{result.schedule_name}]")
    print(f"{indent}final_max_abs_error: {result.final_max_abs_error:.6e}")
    print(f"{indent}best_max_abs_error: {result.best_max_abs_error:.6e}")
    print(f"{indent}iterations_to_target_error: {iters_to_target}")
    print(f"{indent}loss: {result.loss:.6f}")
    print(f"{indent}avg_step_time_ms: {result.avg_step_time_sec * 1e3:.3f}")
    print(f"{indent}p50_step_time_ms: {result.p50_step_time_sec * 1e3:.3f}")
    print(f"{indent}p90_step_time_ms: {result.p90_step_time_sec * 1e3:.3f}")
    print(f"{indent}compile_overhead_ms: {result.compile_overhead_sec * 1e3:.3f}")
    print(f"{indent}peak_host_memory_mb: {result.peak_host_memory_mb:.2f}")
    print(f"{indent}peak_device_memory_mb: {peak_device}")
    print(f"{indent}ir_total_equations: {result.ir_summary.total_equations}")
    print(f"{indent}ir_max_loop_nesting: {result.ir_summary.max_loop_nesting}")
    print(f"{indent}ir_num_higher_order_sites: {result.ir_summary.num_higher_order_sites}")
    print(f"{indent}output: {result.output_summary}")
    print()


if __name__ == "__main__":
    main()
