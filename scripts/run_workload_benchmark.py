#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from adscheduler.stablehlo_passes import (
    StableHLOProgram,
    lower_laplacian_schedule_to_stablehlo,
    lower_pinn_schedule_to_stablehlo,
    lower_workload_to_stablehlo,
    run_stablehlo_pass_pipeline,
    run_stablehlo_transform_pipeline,
    score_stablehlo_optimization_surface,
)
from adscheduler.stablehlo_execution import (
    StableHLOExecutionUnavailable,
    compile_stablehlo_with_xla,
)
from adscheduler.laplacian_benchmark import (
    LaplacianBenchmarkConfig,
    available_laplacian_schedule_names,
)
from adscheduler.pinn_benchmark import (
    PINNBenchmarkConfig,
    available_pinn_schedule_names,
)
from adscheduler.workloads import (
    available_derivative_workload_names,
    make_derivative_workload,
    normalize_derivative_workload_names,
)

AUTO_MLP_LAPLACIAN_WORKLOAD = "mlp_laplacian_auto"
AUTO_POISSON_PINN_WORKLOAD = "poisson_pinn_auto"


@dataclass(frozen=True)
class RuntimeStats:
    compile_overhead_sec: float
    avg_runtime_sec: float
    p50_runtime_sec: float
    p90_runtime_sec: float
    min_runtime_sec: float
    max_runtime_sec: float
    output_summary: str
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class WorkloadBenchmarkResult:
    workload_name: str
    description: str
    selected_schedule_name: str | None
    compiler_score_summary: str | None
    selection_overhead_sec: float
    before_compiler_pass: RuntimeStats
    after_compiler_pass: RuntimeStats


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
        help=(
            "Retained for compatibility. Workload auto selection is compiler-pass "
            "informed and does not run timing warmups."
        ),
    )
    parser.add_argument(
        "--auto-memory-budget-mb",
        type=float,
        default=None,
        help="Retained for compatibility with schedule benchmarks.",
    )
    parser.add_argument(
        "--auto-warmup-error-tolerance",
        type=float,
        default=0.10,
        help="Retained for compatibility with schedule benchmarks.",
    )
    parser.add_argument(
        "--auto-max-params-for-forward-like",
        type=int,
        default=50000,
        help="Retained for compatibility with schedule benchmarks.",
    )
    parser.add_argument(
        "--auto-cache-path",
        type=str,
        default=".adscheduler_laplacian_warmup_cache.json",
        help="Retained for compatibility with schedule benchmarks.",
    )
    parser.add_argument(
        "--disable-auto-cache",
        action="store_true",
        help="Retained for compatibility with schedule benchmarks.",
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
        if result.compiler_score_summary is not None:
            print(f"  compiler_score: {result.compiler_score_summary}")
        if result.selected_schedule_name is not None:
            print(f"  selected_schedule: {result.selected_schedule_name}")
            print(f"  selection_overhead_ms: {result.selection_overhead_sec * 1e3:.3f}")
        print(f"  compile_overhead_ms: {result.before_compiler_pass.compile_overhead_sec * 1e3:.3f}")
        _print_runtime_stats(
            "before compiler pass",
            result.before_compiler_pass,
            include_compile_overhead=False,
        )
        _print_runtime_stats(
            "after compiler pass",
            result.after_compiler_pass,
            include_compile_overhead=True,
        )
        _print_avg_runtime_delta(result.before_compiler_pass, result.after_compiler_pass)
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
    program = lower_workload_to_stablehlo(workload_name, seed=seed)
    pipeline_result = run_stablehlo_pass_pipeline(program)
    compiler_score = score_stablehlo_optimization_surface(pipeline_result)
    return _benchmark_callable(
        workload_name=workload.name,
        description=workload.description,
        selected_schedule_name=None,
        compiler_score_summary=_compiler_score_summary(compiler_score),
        selection_overhead_sec=0.0,
        args=workload.args,
        stablehlo_program=program,
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
    hidden_dim = int(np.asarray(params[0][1]).shape[0])
    config = LaplacianBenchmarkConfig(
        seed=seed,
        outer_steps=runs,
        eval_every=max(1, runs),
        num_points=int(points_arr.shape[0]),
        input_dim=int(points_arr.shape[1]),
        hidden_dim=hidden_dim,
        hidden_layers=len(params) - 1,
    )
    selected_schedule, selected_score, scored_candidates, selection_overhead_sec = (
        _select_laplacian_schedule_by_stablehlo(config)
    )
    description = _compiler_auto_description(
        candidate_kind="Laplacian schedules",
        selected_name=selected_schedule,
        selected_score=selected_score,
        scored_candidates=scored_candidates,
    )
    return _benchmark_callable(
        workload_name=AUTO_MLP_LAPLACIAN_WORKLOAD,
        description=description,
        selected_schedule_name=selected_schedule,
        compiler_score_summary=_compiler_score_summary(selected_score),
        selection_overhead_sec=selection_overhead_sec,
        args=(params, points),
        stablehlo_program=lower_laplacian_schedule_to_stablehlo(
            selected_schedule,
            config=config,
        ),
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
    config = PINNBenchmarkConfig(
        seed=seed,
        outer_steps=runs,
        eval_every=max(1, runs),
        grid_size=int(round(np.sqrt(np.asarray(points).shape[0]))),
        input_dim=int(np.asarray(params[0][0]).shape[0]),
        hidden_layers=len(params) - 1,
        hidden_dim=int(np.asarray(params[0][1]).shape[0]),
        activation="tanh",
        output_dim=int(np.asarray(params[-1][1]).shape[0]),
    )
    selected_schedule, selected_score, scored_candidates, selection_overhead_sec = (
        _select_pinn_schedule_by_stablehlo(config)
    )
    description = _compiler_auto_description(
        candidate_kind="PINN schedules",
        selected_name=selected_schedule,
        selected_score=selected_score,
        scored_candidates=scored_candidates,
    )
    return _benchmark_callable(
        workload_name=AUTO_POISSON_PINN_WORKLOAD,
        description=description,
        selected_schedule_name=selected_schedule,
        compiler_score_summary=_compiler_score_summary(selected_score),
        selection_overhead_sec=selection_overhead_sec,
        args=(params, points),
        stablehlo_program=lower_pinn_schedule_to_stablehlo(
            selected_schedule,
            config=config,
        ),
        warmup_runs=warmup_runs,
        runs=runs,
    )


def _select_laplacian_schedule_by_stablehlo(config: LaplacianBenchmarkConfig):
    selection_start = time.perf_counter()
    scored_candidates = []
    for schedule_name in available_laplacian_schedule_names():
        program = lower_laplacian_schedule_to_stablehlo(schedule_name, config=config)
        pipeline_result = run_stablehlo_pass_pipeline(program)
        compiler_score = score_stablehlo_optimization_surface(pipeline_result)
        scored_candidates.append((schedule_name, compiler_score))
    selected_name, selected_score = min(scored_candidates, key=lambda item: item[1].score)
    return selected_name, selected_score, scored_candidates, time.perf_counter() - selection_start


def _select_pinn_schedule_by_stablehlo(config: PINNBenchmarkConfig):
    selection_start = time.perf_counter()
    scored_candidates = []
    for schedule_name in available_pinn_schedule_names():
        program = lower_pinn_schedule_to_stablehlo(schedule_name, config=config)
        pipeline_result = run_stablehlo_pass_pipeline(program)
        compiler_score = score_stablehlo_optimization_surface(pipeline_result)
        scored_candidates.append((schedule_name, compiler_score))
    selected_name, selected_score = min(scored_candidates, key=lambda item: item[1].score)
    return selected_name, selected_score, scored_candidates, time.perf_counter() - selection_start


def _compiler_auto_description(
    *,
    candidate_kind: str,
    selected_name: str,
    selected_score,
    scored_candidates,
) -> str:
    score_summary = "; ".join(
        f"{name}:score={score.score:.3f},ops={score.total_operations},"
        f"lap_recur={score.laplacian_recurrence_rewrites},"
        f"const_folds={score.constant_foldable_operations},"
        f"mixed_partial={score.mixed_partial_cse_rewrites},"
        f"symm_kernel={score.symmetric_kernel_rewrites}"
        for name, score in scored_candidates
    )
    return (
        "Compiler-pass-informed selection over "
        f"{candidate_kind}. "
        f"selected={selected_name} compiler_score={selected_score.score:.3f} "
        f"estimated_optimized_ops={selected_score.estimated_optimized_operations:.3f}. "
        f"candidate_scores=[{score_summary}]"
    )


def _compiler_score_summary(score) -> str:
    return (
        f"score={score.score:.3f} "
        f"estimated_optimized_ops={score.estimated_optimized_operations:.3f} "
        f"ops={score.total_operations} "
        f"lap_recur={score.laplacian_recurrence_rewrites} "
        f"const_folds={score.constant_foldable_operations} "
        f"mixed_partial={score.mixed_partial_cse_rewrites} "
        f"symm_kernel={score.symmetric_kernel_rewrites}"
    )


def _benchmark_callable(
    *,
    workload_name: str,
    description: str,
    selected_schedule_name: str | None,
    compiler_score_summary: str | None,
    selection_overhead_sec: float,
    args: tuple,
    stablehlo_program: StableHLOProgram,
    warmup_runs: int,
    runs: int,
) -> WorkloadBenchmarkResult:
    before_stats = _benchmark_stablehlo_program(
        program=stablehlo_program,
        args=args,
        warmup_runs=warmup_runs,
        runs=runs,
        execution_label="original StableHLO",
    )
    transform_result = run_stablehlo_transform_pipeline(
        stablehlo_program,
        source_args=args,
    )
    original_fingerprint = _stablehlo_fingerprint(transform_result.original_program)
    transformed_fingerprint = _stablehlo_fingerprint(transform_result.transformed_program)
    print(f"\n[{workload_name}] StableHLO transform results:")
    print(f"  original_stablehlo_sha256: {original_fingerprint}")
    print(f"  transformed_stablehlo_sha256: {transformed_fingerprint}")
    print(
        "  transformed_text_changed: "
        f"{transform_result.original_program.stablehlo_text != transform_result.transformed_program.stablehlo_text}"
    )
    if not transform_result.transform_results:
        print("  no transform passes ran")
    else:
        for pass_result in transform_result.transform_results:
            print(f"  - {pass_result.pass_name}")
            print(f"    metrics: {pass_result.metrics}")
            if pass_result.details:
                print("    details:")
                for detail in pass_result.details[:5]:
                    print(f"      - {detail}")

    after_stats = _benchmark_stablehlo_program(
        original_program=transform_result.original_program,
        program=transform_result.transformed_program,
        args=args,
        warmup_runs=warmup_runs,
        runs=runs,
        execution_label="transformed StableHLO",
    )
    return WorkloadBenchmarkResult(
        workload_name=workload_name,
        description=description,
        selected_schedule_name=selected_schedule_name,
        compiler_score_summary=compiler_score_summary,
        selection_overhead_sec=selection_overhead_sec,
        before_compiler_pass=before_stats,
        after_compiler_pass=after_stats,
    )


def _benchmark_stablehlo_program(
    *,
    program: StableHLOProgram,
    args: tuple,
    warmup_runs: int,
    runs: int,
    execution_label: str,
    original_program: StableHLOProgram | None = None,
) -> RuntimeStats:
    if original_program is not None and program.stablehlo_text == original_program.stablehlo_text:
        return _unavailable_runtime_stats(
            "StableHLO transform produced identical text; refusing to report "
            "after-pass timing as optimized code."
        )

    try:
        executable = compile_stablehlo_with_xla(program)
    except StableHLOExecutionUnavailable as exc:
        return _unavailable_runtime_stats(str(exc))

    try:
        if executable.prepare_args is not None and executable.call_prepared is not None:
            # Prepare arguments once outside the warmup and timed loops.
            prepared_args = executable.prepare_args(*args)

            def run_once():
                return executable.call_prepared(prepared_args)
        else:
            # Fallback for older executables.
            def run_once():
                return executable.callable(*args)

        output = run_once()
        _block_tree(output)

        for _ in range(warmup_runs):
            output = run_once()
            _block_tree(output)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            output = run_once()
            _block_tree(output)
            timings.append(time.perf_counter() - start)

    except Exception as exc:
        return _unavailable_runtime_stats(
            f"XLA/PJRT compiled {execution_label} but execution failed: {exc}"
        )

    timings_arr = np.asarray(timings, dtype=np.float64)
    return RuntimeStats(
        compile_overhead_sec=executable.compile_overhead_sec,
        avg_runtime_sec=float(np.mean(timings_arr)),
        p50_runtime_sec=float(np.percentile(timings_arr, 50)),
        p90_runtime_sec=float(np.percentile(timings_arr, 90)),
        min_runtime_sec=float(np.min(timings_arr)),
        max_runtime_sec=float(np.max(timings_arr)),
        output_summary=_summarize_output(output),
    )


def _unavailable_runtime_stats(reason: str) -> RuntimeStats:
    return RuntimeStats(
        compile_overhead_sec=float("nan"),
        avg_runtime_sec=float("nan"),
        p50_runtime_sec=float("nan"),
        p90_runtime_sec=float("nan"),
        min_runtime_sec=float("nan"),
        max_runtime_sec=float("nan"),
        output_summary="<unavailable>",
        unavailable_reason=reason,
    )


def _stablehlo_fingerprint(program: StableHLOProgram) -> str:
    return hashlib.sha256(program.stablehlo_text.encode("utf-8")).hexdigest()[:16]


def _print_runtime_stats(
    label: str,
    stats: RuntimeStats,
    *,
    include_compile_overhead: bool,
) -> None:
    print(f"  {label}:")
    if stats.unavailable_reason is not None:
        print(f"    unavailable: {stats.unavailable_reason}")
        return
    if include_compile_overhead:
        print(f"    compile_overhead_ms: {stats.compile_overhead_sec * 1e3:.3f}")
    print(f"    avg_runtime_ms: {stats.avg_runtime_sec * 1e3:.3f}")
    print(f"    p50_runtime_ms: {stats.p50_runtime_sec * 1e3:.3f}")
    print(f"    p90_runtime_ms: {stats.p90_runtime_sec * 1e3:.3f}")
    print(f"    min_runtime_ms: {stats.min_runtime_sec * 1e3:.3f}")
    print(f"    max_runtime_ms: {stats.max_runtime_sec * 1e3:.3f}")
    print(f"    output: {stats.output_summary}")


def _print_avg_runtime_delta(before: RuntimeStats, after: RuntimeStats) -> None:
    print("  avg_runtime_delta_ms_before_minus_after:")
    if before.unavailable_reason is not None or after.unavailable_reason is not None:
        print("    unavailable")
        return
    delta_ms = (before.avg_runtime_sec - after.avg_runtime_sec) * 1e3
    print(f"    {delta_ms:.3f}")


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
    return jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


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
