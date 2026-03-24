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

from adscheduler.benchmark import (
    BenchmarkConfig,
    evaluation_report_to_dict,
    run_evaluation_protocol,
)
from adscheduler.maml import available_schedule_names, normalize_schedule_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run schedule benchmarking for MAML meta-learning with statistical, "
            "hardware, and optional warmup-based auto selection metrics."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--meta-test-tasks", type=int, default=64)
    parser.add_argument("--target-accuracy", type=float, default=0.85)

    parser.add_argument("--in-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--n-support", type=int, default=16)
    parser.add_argument("--n-query", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--outer-lr", type=float, default=0.01)

    parser.add_argument(
        "--schedule",
        action="append",
        choices=available_schedule_names(),
        dest="schedules",
        help="Fixed schedule baseline to benchmark. Repeat for multiple.",
    )

    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="Run warmup-based automatic schedule selection and compare vs fixed baselines.",
    )
    parser.add_argument(
        "--auto-candidate",
        action="append",
        choices=available_schedule_names(),
        dest="auto_candidates",
        help="Candidate schedule considered by auto tuner. Repeat for multiple.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of warmup outer steps profiled per candidate schedule.",
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
        help="Relative tolerance for warmup loss guard (e.g., 0.1 = 10%%).",
    )
    parser.add_argument(
        "--max-params-for-forward-like",
        type=int,
        default=50000,
        help="Simple guard: skip for/for_remat if parameter count exceeds this threshold.",
    )
    parser.add_argument(
        "--disable-warmup-cache",
        action="store_true",
        help="Disable warmup selection cache.",
    )
    parser.add_argument(
        "--warmup-cache-path",
        type=str,
        default=".adscheduler_warmup_cache.json",
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
    selected_baselines = normalize_schedule_names(args.schedules)
    selected_auto_candidates = (
        normalize_schedule_names(args.auto_candidates)
        if args.auto_candidates
        else selected_baselines
    )

    config = BenchmarkConfig(
        seed=args.seed,
        outer_steps=args.outer_steps,
        eval_every=args.eval_every,
        meta_batch_size=args.meta_batch_size,
        meta_test_tasks=args.meta_test_tasks,
        target_meta_test_accuracy=args.target_accuracy,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_support=args.n_support,
        n_query=args.n_query,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
    )

    report = run_evaluation_protocol(
        config,
        baseline_schedules=selected_baselines,
        include_auto=args.include_auto,
        auto_candidate_schedules=selected_auto_candidates,
        warmup_steps=args.warmup_steps,
        memory_budget_mb=args.memory_budget_mb,
        loss_guard_tolerance=args.warmup_loss_tolerance,
        max_params_for_forward_like=args.max_params_for_forward_like,
        use_cache=not args.disable_warmup_cache,
        cache_path=args.warmup_cache_path,
    )

    print("=== Benchmark Summary ===")
    print(f"fixed_schedules: {', '.join(selected_baselines)}")
    print(f"outer_steps: {args.outer_steps} eval_every: {args.eval_every}")
    print(f"target_meta_test_accuracy: {args.target_accuracy}")
    if args.include_auto:
        print(f"auto_candidates: {', '.join(selected_auto_candidates)}")
        print(
            "warmup: "
            f"steps={args.warmup_steps} "
            f"loss_tolerance={args.warmup_loss_tolerance} "
            f"cache={'off' if args.disable_warmup_cache else args.warmup_cache_path}"
        )
    print()

    print("=== Fixed Schedule Results ===")
    for result in report.fixed_results:
        iters_to_target = (
            str(result.outer_iterations_to_target)
            if result.outer_iterations_to_target is not None
            else "not reached"
        )
        peak_device = (
            f"{result.peak_device_memory_mb:.2f}"
            if result.peak_device_memory_mb is not None
            else "n/a"
        )

        print(f"[{result.schedule_name}]")
        print(f"  final_meta_test_accuracy: {result.final_meta_test_accuracy:.4f}")
        print(f"  best_meta_test_accuracy: {result.best_meta_test_accuracy:.4f}")
        print(f"  outer_iterations_to_target: {iters_to_target}")
        print(f"  final_meta_train_loss: {result.final_meta_train_loss:.6f}")
        print(f"  avg_outer_step_time_ms: {result.avg_outer_step_time_sec * 1e3:.3f}")
        print(f"  p50_outer_step_time_ms: {result.p50_outer_step_time_sec * 1e3:.3f}")
        print(f"  p90_outer_step_time_ms: {result.p90_outer_step_time_sec * 1e3:.3f}")
        print(f"  compile_overhead_ms: {result.compile_overhead_sec * 1e3:.3f}")
        print(f"  peak_host_memory_mb: {result.peak_host_memory_mb:.2f}")
        print(f"  peak_device_memory_mb: {peak_device}")
        print(f"  ir_total_equations: {result.ir_summary.total_equations}")
        print(f"  ir_max_loop_nesting: {result.ir_summary.max_loop_nesting}")
        print(f"  ir_num_higher_order_sites: {result.ir_summary.num_higher_order_sites}")
        print()

    if report.auto_result is not None:
        auto = report.auto_result
        warmup = report.warmup_selection
        iters_to_target = (
            str(auto.outer_iterations_to_target)
            if auto.outer_iterations_to_target is not None
            else "not reached"
        )
        peak_device = (
            f"{auto.peak_device_memory_mb:.2f}"
            if auto.peak_device_memory_mb is not None
            else "n/a"
        )

        print("=== Auto Selection Result ===")
        print(f"  selected_schedule: {auto.selected_schedule_name}")
        print(f"  used_warmup_cache: {auto.used_warmup_cache}")
        print(f"  warmup_overhead_ms: {auto.warmup_overhead_sec * 1e3:.3f}")
        print(f"  final_meta_test_accuracy: {auto.final_meta_test_accuracy:.4f}")
        print(f"  best_meta_test_accuracy: {auto.best_meta_test_accuracy:.4f}")
        print(f"  outer_iterations_to_target: {iters_to_target}")
        print(f"  final_meta_train_loss: {auto.final_meta_train_loss:.6f}")
        print(f"  avg_outer_step_time_ms: {auto.avg_outer_step_time_sec * 1e3:.3f}")
        print(f"  p50_outer_step_time_ms: {auto.p50_outer_step_time_sec * 1e3:.3f}")
        print(f"  p90_outer_step_time_ms: {auto.p90_outer_step_time_sec * 1e3:.3f}")
        print(f"  compile_overhead_ms: {auto.compile_overhead_sec * 1e3:.3f}")
        print(f"  peak_host_memory_mb: {auto.peak_host_memory_mb:.2f}")
        print(f"  peak_device_memory_mb: {peak_device}")
        print(f"  ir_total_equations: {auto.ir_summary.total_equations}")
        print(f"  ir_max_loop_nesting: {auto.ir_summary.max_loop_nesting}")
        print(f"  ir_num_higher_order_sites: {auto.ir_summary.num_higher_order_sites}")
        print()

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
                        f"final_loss={profile.final_meta_loss:.6f} "
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
        print("=== Benchmark JSON ===")
        print(json.dumps(evaluation_report_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
