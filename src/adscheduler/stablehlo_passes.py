from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax

from adscheduler.laplacian_benchmark import (
    LaplacianBenchmarkConfig,
    build_laplacian_schedule_fn,
    normalize_laplacian_schedule_names,
)
from adscheduler.pinn_benchmark import (
    PINNBenchmarkConfig,
    build_pinn_schedule_fn,
    normalize_pinn_schedule_names,
)
from adscheduler.workloads import make_derivative_workload


SSA_VALUE_RE = re.compile(r"%[\w.:\-#]+")
STABLEHLO_OP_RE = re.compile(r'"?(stablehlo\.[\w_]+)"?')
LOCATION_RE = re.compile(r"\s+loc\([^)]*\)")
RESULT_PREFIX_RE = re.compile(r"^\s*(?:%[\w.:\-#]+(?:\s*,\s*%[\w.:\-#]+)*\s*=\s*)")
TYPE_SUFFIX_RE = re.compile(r"\s*:\s*.+$")
ELEMENTWISE_OPS = {
    "stablehlo.abs",
    "stablehlo.add",
    "stablehlo.and",
    "stablehlo.atan2",
    "stablehlo.broadcast_in_dim",
    "stablehlo.ceil",
    "stablehlo.clamp",
    "stablehlo.compare",
    "stablehlo.convert",
    "stablehlo.cosine",
    "stablehlo.divide",
    "stablehlo.exponential",
    "stablehlo.floor",
    "stablehlo.log",
    "stablehlo.logistic",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.or",
    "stablehlo.power",
    "stablehlo.remainder",
    "stablehlo.rsqrt",
    "stablehlo.select",
    "stablehlo.shift_left",
    "stablehlo.shift_right_arithmetic",
    "stablehlo.shift_right_logical",
    "stablehlo.sign",
    "stablehlo.sine",
    "stablehlo.sqrt",
    "stablehlo.subtract",
    "stablehlo.tanh",
    "stablehlo.xor",
}


@dataclass(frozen=True)
class StableHLOProgram:
    name: str
    stablehlo_text: str


@dataclass(frozen=True)
class StableHLOPassResult:
    pass_name: str
    summary: str
    metrics: dict[str, int | float | str]
    details: list[str]


@dataclass(frozen=True)
class StableHLOPipelineResult:
    program_name: str
    total_operations: int
    operation_histogram: dict[str, int]
    pass_results: list[StableHLOPassResult]


@dataclass(frozen=True)
class StableHLOCompilerScore:
    program_name: str
    score: float
    estimated_optimized_operations: float
    total_operations: int
    duplicated_operations: int
    fusion_regions: int
    elementwise_ops_in_regions: int
    expensive_operations: int


def lower_to_stablehlo(
    name: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
) -> StableHLOProgram:
    lowered = jax.jit(fn).lower(*args)
    try:
        stablehlo = lowered.compiler_ir(dialect="stablehlo")
    except TypeError:
        stablehlo = lowered.compiler_ir("stablehlo")
    return StableHLOProgram(name=name, stablehlo_text=str(stablehlo))


def lower_workload_to_stablehlo(
    workload_name: str,
    *,
    seed: int = 0,
) -> StableHLOProgram:
    workload = make_derivative_workload(workload_name, seed=seed)
    return lower_to_stablehlo(workload.name, workload.derivative_task, workload.args)


def lower_laplacian_schedule_to_stablehlo(
    schedule_name: str,
    *,
    config: LaplacianBenchmarkConfig | None = None,
) -> StableHLOProgram:
    selected_schedule = normalize_laplacian_schedule_names((schedule_name,))[0]
    cfg = config or LaplacianBenchmarkConfig(outer_steps=1)
    params, points = _make_laplacian_lowering_args(cfg)
    return lower_to_stablehlo(
        f"laplacian_{selected_schedule}",
        build_laplacian_schedule_fn(selected_schedule),
        (params, points),
    )


def lower_pinn_schedule_to_stablehlo(
    schedule_name: str,
    *,
    config: PINNBenchmarkConfig | None = None,
) -> StableHLOProgram:
    selected_schedule = normalize_pinn_schedule_names((schedule_name,))[0]
    cfg = config or PINNBenchmarkConfig(outer_steps=1)
    params, points = _make_pinn_lowering_args(cfg)
    return lower_to_stablehlo(
        f"pinn_{selected_schedule}",
        build_pinn_schedule_fn(selected_schedule),
        (params, points),
    )


def run_stablehlo_pass_pipeline(
    program: StableHLOProgram,
    *,
    passes: Sequence[str] | None = None,
) -> StableHLOPipelineResult:
    selected_passes = tuple(passes or available_stablehlo_pass_names())
    unknown = sorted(set(selected_passes) - set(_PASS_REGISTRY))
    if unknown:
        known = ", ".join(available_stablehlo_pass_names())
        raise ValueError(f"Unknown StableHLO passes: {', '.join(unknown)}. Known: {known}")

    op_histogram = stablehlo_operation_histogram(program.stablehlo_text)
    pass_results = [
        _PASS_REGISTRY[pass_name](program.stablehlo_text) for pass_name in selected_passes
    ]
    return StableHLOPipelineResult(
        program_name=program.name,
        total_operations=sum(op_histogram.values()),
        operation_histogram=op_histogram,
        pass_results=pass_results,
    )


def available_stablehlo_pass_names() -> tuple[str, ...]:
    return tuple(_PASS_REGISTRY)


def stablehlo_operation_histogram(stablehlo_text: str) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for op_name in STABLEHLO_OP_RE.findall(stablehlo_text):
        histogram[op_name] = histogram.get(op_name, 0) + 1
    return dict(sorted(histogram.items()))


def format_stablehlo_pipeline_report(result: StableHLOPipelineResult) -> str:
    lines = [
        f"program: {result.program_name}",
        f"total_operations: {result.total_operations}",
        "operation_histogram:",
    ]
    for op_name, count in result.operation_histogram.items():
        lines.append(f"  - {op_name}: {count}")

    lines.append("pass_results:")
    for pass_result in result.pass_results:
        lines.append(f"  - pass: {pass_result.pass_name}")
        lines.append(f"    summary: {pass_result.summary}")
        lines.append("    metrics:")
        for key, value in pass_result.metrics.items():
            lines.append(f"      {key}: {value}")
        if pass_result.details:
            lines.append("    details:")
            for detail in pass_result.details:
                lines.append(f"      - {detail}")
    return "\n".join(lines)


def score_stablehlo_optimization_surface(
    result: StableHLOPipelineResult,
) -> StableHLOCompilerScore:
    """Heuristic score for choosing IR that looks amenable to compiler cleanup.

    Lower is better. This is intentionally conservative: it rewards obvious
    duplicate operations and simple elementwise fusion regions, but still keeps
    the unoptimized operation count and expensive-op count in the objective.
    """

    pass_metrics = {pass_result.pass_name: pass_result.metrics for pass_result in result.pass_results}
    duplicated_operations = int(
        pass_metrics.get("duplicate_operations", {}).get("duplicated_operations", 0)
    )
    fusion_regions = int(
        pass_metrics.get("elementwise_fusion_regions", {}).get("fusion_regions", 0)
    )
    elementwise_ops_in_regions = int(
        pass_metrics.get("elementwise_fusion_regions", {}).get(
            "elementwise_ops_in_regions",
            0,
        )
    )
    expensive_operations = sum(
        int(value)
        for value in pass_metrics.get("expensive_operations", {}).values()
        if isinstance(value, int | float)
    )

    estimated_optimized_operations = max(
        0.0,
        result.total_operations
        - 0.75 * duplicated_operations
        - 0.25 * max(0, elementwise_ops_in_regions - fusion_regions),
    )
    score = estimated_optimized_operations + 2.0 * expensive_operations
    return StableHLOCompilerScore(
        program_name=result.program_name,
        score=score,
        estimated_optimized_operations=estimated_optimized_operations,
        total_operations=result.total_operations,
        duplicated_operations=duplicated_operations,
        fusion_regions=fusion_regions,
        elementwise_ops_in_regions=elementwise_ops_in_regions,
        expensive_operations=expensive_operations,
    )


def _duplicate_operation_pass(stablehlo_text: str) -> StableHLOPassResult:
    signatures: dict[str, int] = {}
    for line in _operation_lines(stablehlo_text):
        signature = _canonical_operation_signature(line)
        signatures[signature] = signatures.get(signature, 0) + 1

    duplicate_groups = {
        signature: count for signature, count in signatures.items() if count > 1
    }
    duplicated_ops = sum(count - 1 for count in duplicate_groups.values())
    details = [
        f"{count}x {signature[:160]}"
        for signature, count in sorted(
            duplicate_groups.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]
    return StableHLOPassResult(
        pass_name="duplicate_operations",
        summary="Finds repeated canonical StableHLO operation lines as CSE candidates.",
        metrics={
            "duplicate_groups": len(duplicate_groups),
            "duplicated_operations": duplicated_ops,
        },
        details=details,
    )


def _dead_result_pass(stablehlo_text: str) -> StableHLOPassResult:
    defined_values: set[str] = set()
    used_values: set[str] = set()
    for line in stablehlo_text.splitlines():
        if "=" not in line:
            used_values.update(SSA_VALUE_RE.findall(line))
            continue

        lhs, rhs = line.split("=", 1)
        lhs_values = set(SSA_VALUE_RE.findall(lhs))
        defined_values.update(lhs_values)
        used_values.update(set(SSA_VALUE_RE.findall(rhs)) - lhs_values)

    dead_values = sorted(defined_values - used_values)
    return StableHLOPassResult(
        pass_name="dead_results",
        summary="Finds SSA results that appear to have no textual uses after definition.",
        metrics={
            "defined_values": len(defined_values),
            "used_values": len(used_values),
            "dead_values": len(dead_values),
        },
        details=dead_values[:20],
    )


def _elementwise_fusion_pass(stablehlo_text: str) -> StableHLOPassResult:
    runs: list[int] = []
    current_run = 0
    for line in _operation_lines(stablehlo_text):
        op_name = _operation_name(line)
        if op_name in ELEMENTWISE_OPS:
            current_run += 1
        else:
            if current_run:
                runs.append(current_run)
            current_run = 0
    if current_run:
        runs.append(current_run)

    useful_runs = [run for run in runs if run >= 2]
    return StableHLOPassResult(
        pass_name="elementwise_fusion_regions",
        summary="Counts contiguous elementwise operation runs as simple fusion opportunities.",
        metrics={
            "fusion_regions": len(useful_runs),
            "elementwise_ops_in_regions": sum(useful_runs),
            "largest_region_size": max(useful_runs, default=0),
        },
        details=[f"region_size={run}" for run in sorted(useful_runs, reverse=True)[:20]],
    )


def _expensive_operation_pass(stablehlo_text: str) -> StableHLOPassResult:
    histogram = stablehlo_operation_histogram(stablehlo_text)
    selected = {
        op_name: count
        for op_name, count in histogram.items()
        if op_name
        in {
            "stablehlo.convolution",
            "stablehlo.dot_general",
            "stablehlo.reduce",
            "stablehlo.reduce_window",
            "stablehlo.transpose",
            "stablehlo.while",
        }
    }
    return StableHLOPassResult(
        pass_name="expensive_operations",
        summary="Highlights operations likely to dominate runtime or block fusion.",
        metrics=selected,
        details=[],
    )


def _operation_lines(stablehlo_text: str) -> list[str]:
    return [
        line.strip()
        for line in stablehlo_text.splitlines()
        if STABLEHLO_OP_RE.search(line)
    ]


def _operation_name(line: str) -> str:
    match = STABLEHLO_OP_RE.search(line)
    return match.group(1) if match else ""


def _canonical_operation_signature(line: str) -> str:
    signature = LOCATION_RE.sub("", line.strip())
    signature = RESULT_PREFIX_RE.sub("", signature)
    signature = TYPE_SUFFIX_RE.sub("", signature)
    signature = re.sub(r"%[\w.:\-#]+", "%v", signature)
    signature = re.sub(r"\s+", " ", signature)
    return signature


def _make_laplacian_lowering_args(config: LaplacianBenchmarkConfig):
    from adscheduler.laplacian_benchmark import _initialize_laplacian_state

    params, _, sample_points = _initialize_laplacian_state(config)
    return params, sample_points


def _make_pinn_lowering_args(config: PINNBenchmarkConfig):
    from adscheduler.pinn_benchmark import _initialize_pinn_state

    params, _, sample_points = _initialize_pinn_state(config)
    return params, sample_points


_PASS_REGISTRY: dict[str, Callable[[str], StableHLOPassResult]] = {
    "duplicate_operations": _duplicate_operation_pass,
    "dead_results": _dead_result_pass,
    "elementwise_fusion_regions": _elementwise_fusion_pass,
    "expensive_operations": _expensive_operation_pass,
}
