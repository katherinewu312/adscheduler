#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.tree_util as jtu

from adscheduler.laplacian_benchmark import (
    LaplacianBenchmarkConfig,
    available_laplacian_schedule_names,
    build_laplacian_schedule_fn,
    choose_laplacian_schedule_via_warmup,
)
from adscheduler.pinn_benchmark import (
    PINNBenchmarkConfig,
    available_pinn_schedule_names,
    build_pinn_schedule_fn,
    choose_pinn_schedule_via_warmup,
)
from adscheduler.workloads import (
    available_derivative_workload_names,
    make_derivative_workload,
    normalize_derivative_workload_names,
)

AUTO_MLP_LAPLACIAN_WORKLOAD = "mlp_laplacian_auto"
AUTO_POISSON_PINN_WORKLOAD = "poisson_pinn_auto"


@dataclass(frozen=True)
class WorkloadBenchmarkResult:
    workload_name: str
    description: str
    selected_schedule_name: str | None
    selection_overhead_sec: float
    compile_overhead_sec: float
    avg_runtime_sec: float
    p50_runtime_sec: float
    p90_runtime_sec: float
    min_runtime_sec: float
    max_runtime_sec: float
    output_summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX derivative workloads with JIT compile and runtime metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workload",
        action="append",
        choices=(
            *available_derivative_workload_names(),
            AUTO_MLP_LAPLACIAN_WORKLOAD,
            AUTO_POISSON_PINN_WORKLOAD,
        ),
        dest="workloads",
        help="Derivative workload to benchmark. Repeat for multiple.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of untimed warmup executions after compilation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of timed executions.",
    )
    parser.add_argument(
        "--auto-warmup-steps",
        type=int,
        default=3,
        help="Number of warmup executions used by mlp_laplacian_auto selection.",
    )
    parser.add_argument(
        "--auto-memory-budget-mb",
        type=float,
        default=None,
        help="Optional memory budget used by mlp_laplacian_auto selection.",
    )
    parser.add_argument(
        "--auto-warmup-error-tolerance",
        type=float,
        default=0.10,
        help="Relative numerical-error tolerance used by mlp_laplacian_auto selection.",
    )
    parser.add_argument(
        "--auto-max-params-for-forward-like",
        type=int,
        default=50000,
        help="Skip for/for_remat during auto selection if parameter count exceeds this.",
    )
    parser.add_argument(
        "--auto-cache-path",
        type=str,
        default=".adscheduler_laplacian_warmup_cache.json",
        help="Warmup selection cache path used by mlp_laplacian_auto.",
    )
    parser.add_argument(
        "--disable-auto-cache",
        action="store_true",
        help="Disable warmup cache for mlp_laplacian_auto.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full benchmark results as JSON after the text summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload_names = _normalize_benchmark_workload_names(
        args.workloads
        or (
            "mlp_laplacian_hessian",
            "mlp_laplacian_jvp_grad",
            "mlp_laplacian_jet",
            AUTO_MLP_LAPLACIAN_WORKLOAD,
        )
    )
    if args.warmup_runs < 0:
        raise ValueError("warmup-runs must be >= 0")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")

    results = [
        benchmark_workload(
            workload_name,
            seed=args.seed,
            warmup_runs=args.warmup_runs,
            runs=args.runs,
            auto_warmup_steps=args.auto_warmup_steps,
            auto_memory_budget_mb=args.auto_memory_budget_mb,
            auto_warmup_error_tolerance=args.auto_warmup_error_tolerance,
            auto_max_params_for_forward_like=args.auto_max_params_for_forward_like,
            auto_cache_path=args.auto_cache_path,
            use_auto_cache=not args.disable_auto_cache,
        )
        for workload_name in workload_names
    ]

    print("=== Workload Benchmark Summary ===")
    print(f"workloads: {', '.join(workload_names)}")
    print(f"warmup_runs: {args.warmup_runs} runs: {args.runs}")
    print(f"backend: {jax.default_backend()}")
    print()

    for result in results:
        print(f"[{result.workload_name}]")
        print(f"  description: {result.description}")
        if result.selected_schedule_name is not None:
            print(f"  selected_schedule: {result.selected_schedule_name}")
            print(f"  selection_overhead_ms: {result.selection_overhead_sec * 1e3:.3f}")
        print(f"  compile_overhead_ms: {result.compile_overhead_sec * 1e3:.3f}")
        print(f"  avg_runtime_ms: {result.avg_runtime_sec * 1e3:.3f}")
        print(f"  p50_runtime_ms: {result.p50_runtime_sec * 1e3:.3f}")
        print(f"  p90_runtime_ms: {result.p90_runtime_sec * 1e3:.3f}")
        print(f"  min_runtime_ms: {result.min_runtime_sec * 1e3:.3f}")
        print(f"  max_runtime_ms: {result.max_runtime_sec * 1e3:.3f}")
        print(f"  output: {result.output_summary}")
        print()

    if args.json:
        print("=== Workload Benchmark JSON ===")
        print(json.dumps([asdict(result) for result in results], indent=2))


def benchmark_workload(
    workload_name: str,
    *,
    seed: int,
    warmup_runs: int,
    runs: int,
    auto_warmup_steps: int,
    auto_memory_budget_mb: float | None,
    auto_warmup_error_tolerance: float,
    auto_max_params_for_forward_like: int,
    auto_cache_path: str,
    use_auto_cache: bool,
) -> WorkloadBenchmarkResult:
    if workload_name == AUTO_MLP_LAPLACIAN_WORKLOAD:
        return benchmark_auto_mlp_laplacian(
            seed=seed,
            warmup_runs=warmup_runs,
            runs=runs,
            auto_warmup_steps=auto_warmup_steps,
            auto_memory_budget_mb=auto_memory_budget_mb,
            auto_warmup_error_tolerance=auto_warmup_error_tolerance,
            auto_max_params_for_forward_like=auto_max_params_for_forward_like,
            auto_cache_path=auto_cache_path,
            use_auto_cache=use_auto_cache,
        )

    if workload_name == AUTO_POISSON_PINN_WORKLOAD:
        return benchmark_auto_poisson_pinn(
            seed=seed,
            warmup_runs=warmup_runs,
            runs=runs,
            auto_warmup_steps=auto_warmup_steps,
        )

    workload = make_derivative_workload(workload_name, seed=seed)
    return _benchmark_callable(
        workload_name=workload.name,
        description=workload.description,
        selected_schedule_name=None,
        selection_overhead_sec=0.0,
        fn=workload.derivative_task,
        args=workload.args,
        warmup_runs=warmup_runs,
        runs=runs,
    )


def benchmark_auto_mlp_laplacian(
    *,
    seed: int,
    warmup_runs: int,
    runs: int,
    auto_warmup_steps: int,
    auto_memory_budget_mb: float | None,
    auto_warmup_error_tolerance: float,
    auto_max_params_for_forward_like: int,
    auto_cache_path: str,
    use_auto_cache: bool,
) -> WorkloadBenchmarkResult:
    workload = make_derivative_workload("mlp_laplacian_hessian", seed=seed)
    params, points = workload.args
    points_arr = np.asarray(points)
    config = LaplacianBenchmarkConfig(
        seed=seed,
        outer_steps=runs,
        eval_every=max(1, runs),
        num_points=int(points_arr.shape[0]),
        input_dim=int(points_arr.shape[1]),
    )

    selection_start = time.perf_counter()
    selected_schedule, selection = choose_laplacian_schedule_via_warmup(
        config,
        params=params,
        candidate_schedules=available_laplacian_schedule_names(),
        warmup_steps=auto_warmup_steps,
        memory_budget_mb=auto_memory_budget_mb,
        error_guard_tolerance=auto_warmup_error_tolerance,
        max_params_for_forward_like=auto_max_params_for_forward_like,
        use_cache=use_auto_cache,
        cache_path=auto_cache_path,
    )
    selection_overhead_sec = time.perf_counter() - selection_start

    schedule_fn = build_laplacian_schedule_fn(selected_schedule)
    description = (
        "Input-space Laplacian of a 64-layer JAX MLP with hidden size 128 via warmup-selected "
        f"schedule '{selected_schedule}' over ror/for/jacrev/jacfwd/remat candidates."
    )
    if selection.from_cache:
        description += " Selection came from cache."
    return _benchmark_callable(
        workload_name=AUTO_MLP_LAPLACIAN_WORKLOAD,
        description=description,
        selected_schedule_name=selected_schedule,
        selection_overhead_sec=selection_overhead_sec,
        fn=schedule_fn,
        args=(params, points),
        warmup_runs=warmup_runs,
        runs=runs,
    )


def benchmark_auto_poisson_pinn(
    *,
    seed: int,
    warmup_runs: int,
    runs: int,
    auto_warmup_steps: int,
) -> WorkloadBenchmarkResult:
    workload = make_derivative_workload("poisson_pinn_hessian", seed=seed)
    params, points = workload.args
    selection_start = time.perf_counter()
    selected_schedule, selection = choose_pinn_schedule_via_warmup(
        PINNBenchmarkConfig(
            seed=seed,
            outer_steps=runs,
            eval_every=max(1, runs),
            grid_size=int(round(np.sqrt(np.asarray(points).shape[0]))),
            input_dim=int(np.asarray(params[0][0]).shape[0]),
            hidden_layers=len(params) - 1,
            hidden_dim=int(np.asarray(params[0][1]).shape[0]),
            activation="tanh",
            output_dim=int(np.asarray(params[-1][1]).shape[0]),
        ),
        params=params,
        points=points,
        candidate_schedules=available_pinn_schedule_names(),
        warmup_steps=auto_warmup_steps,
        memory_budget_mb=None,
        use_cache=False,
    )
    selection_overhead_sec = time.perf_counter() - selection_start
    selected_fn = build_pinn_schedule_fn(selected_schedule)
    selected_profile = next(
        profile
        for profile in selection.profiles
        if profile.schedule_name == selected_schedule
    )
    description = (
        "Training loss and parameter gradient for a Poisson PINN via "
        f"warmup-selected strategy '{selected_schedule}'. "
        f"selection_score={selected_profile.score:.6f} "
        f"max_reference_error={selected_profile.final_max_abs_error:.6e}."
    )
    return _benchmark_callable(
        workload_name=AUTO_POISSON_PINN_WORKLOAD,
        description=description,
        selected_schedule_name=selected_schedule,
        selection_overhead_sec=selection_overhead_sec,
        fn=selected_fn,
        args=(params, points),
        warmup_runs=warmup_runs,
        runs=runs,
    )


def _benchmark_callable(
    *,
    workload_name: str,
    description: str,
    selected_schedule_name: str | None,
    selection_overhead_sec: float,
    fn,
    args: tuple,
    warmup_runs: int,
    runs: int,
) -> WorkloadBenchmarkResult:
    jitted_task = jax.jit(fn)

    compile_start = time.perf_counter()
    try:
        compiled_task = jitted_task.lower(*args).compile()
    except AttributeError:
        compiled_task = jitted_task
    output = compiled_task(*args)
    _block_tree(output)
    compile_overhead_sec = time.perf_counter() - compile_start

    for _ in range(warmup_runs):
        output = compiled_task(*args)
        _block_tree(output)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        output = compiled_task(*args)
        _block_tree(output)
        timings.append(time.perf_counter() - start)

    timings_arr = np.asarray(timings, dtype=np.float64)
    return WorkloadBenchmarkResult(
        workload_name=workload_name,
        description=description,
        selected_schedule_name=selected_schedule_name,
        selection_overhead_sec=selection_overhead_sec,
        compile_overhead_sec=compile_overhead_sec,
        avg_runtime_sec=float(np.mean(timings_arr)),
        p50_runtime_sec=float(np.percentile(timings_arr, 50)),
        p90_runtime_sec=float(np.percentile(timings_arr, 90)),
        min_runtime_sec=float(np.min(timings_arr)),
        max_runtime_sec=float(np.max(timings_arr)),
        output_summary=_summarize_output(output),
    )


def _normalize_benchmark_workload_names(workload_names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(workload_names))
    static_names = [
        name
        for name in normalized
        if name not in {AUTO_MLP_LAPLACIAN_WORKLOAD, AUTO_POISSON_PINN_WORKLOAD}
    ]
    normalize_derivative_workload_names(static_names)
    return normalized


def _block_tree(tree):
    return jtu.tree_map(lambda x: x.block_until_ready(), tree)


def _summarize_output(output) -> str:
    leaves = jtu.tree_leaves(output)
    if not leaves:
        return "<empty>"

    summaries = []
    for leaf in leaves[:3]:
        arr = np.asarray(leaf)
        summaries.append(
            f"shape={arr.shape} dtype={arr.dtype} "
            f"mean={float(np.mean(arr)):.6f} min={float(np.min(arr)):.6f} "
            f"max={float(np.max(arr)):.6f}"
        )
    if len(leaves) > 3:
        summaries.append(f"... {len(leaves) - 3} more leaves")
    return "; ".join(summaries)


if __name__ == "__main__":
    main()
