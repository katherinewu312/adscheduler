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
STABLEHLO_BINARY_OP_LINE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<result>%[\w.:\-#]+)\s*=\s*"
    r'"?(?P<op>stablehlo\.[\w_]+)"?\s+'
    r"(?P<lhs>%[\w.:\-#]+)\s*,\s*"
    r"(?P<rhs>%[\w.:\-#]+)"
    r"(?P<tail>\s*:.*)$"
)
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
COMMUTATIVE_BINARY_OPS = {
    "stablehlo.add",
    "stablehlo.multiply",
    "stablehlo.and",
    "stablehlo.or",
    "stablehlo.xor",
    "stablehlo.maximum",
    "stablehlo.minimum",
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
class StableHLOTransformResult:
    original_program: StableHLOProgram
    transformed_program: StableHLOProgram
    analysis: StableHLOPipelineResult
    transform_results: list[StableHLOPassResult]


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


def run_stablehlo_transform_pipeline(
    program: StableHLOProgram,
    *,
    passes: Sequence[str] | None = None,
) -> StableHLOTransformResult:
    """Run source-level StableHLO optimization passes.

    The analysis pipeline is intentionally preserved for scoring. This function
    additionally rewrites the StableHLO module with conservative SSA transforms
    that keep the module in StableHLO text form so another backend can compile it.

    Transform order matters:
      1. canonicalize commutative operands so CSE sees more equivalent ops
      2. eliminate duplicate operations
      3. eliminate newly-dead results
    """

    selected_passes = tuple(passes or available_stablehlo_pass_names())
    analysis = run_stablehlo_pass_pipeline(program, passes=selected_passes)
    stablehlo_text = program.stablehlo_text
    transform_results: list[StableHLOPassResult] = []

    if "commutative_canonicalization" in selected_passes:
        stablehlo_text, pass_result = _canonicalize_commutative_operations(stablehlo_text)
        transform_results.append(pass_result)

    if "duplicate_operations" in selected_passes:
        stablehlo_text, pass_result = _eliminate_duplicate_operations(stablehlo_text)
        transform_results.append(pass_result)

    # Run DCE last, because CSE can create newly dead ops.
    if "dead_results" in selected_passes:
        stablehlo_text, pass_result = _eliminate_dead_results(stablehlo_text)
        transform_results.append(pass_result)

    transformed_program = StableHLOProgram(
        name=f"{program.name}_stablehlo_passes",
        stablehlo_text=stablehlo_text,
    )
    return StableHLOTransformResult(
        original_program=program,
        transformed_program=transformed_program,
        analysis=analysis,
        transform_results=transform_results,
    )


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
    canonicalized_operations = int(
        pass_metrics.get("commutative_canonicalization", {}).get(
            "canonicalizable_operations",
            0,
        )
    )

    estimated_optimized_operations = max(
        0.0,
        result.total_operations
        - 0.75 * duplicated_operations
        - 0.10 * canonicalized_operations
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


def _commutative_canonicalization_pass(stablehlo_text: str) -> StableHLOPassResult:
    canonicalizable = 0
    details: list[str] = []

    for line in stablehlo_text.splitlines():
        binary = STABLEHLO_BINARY_OP_LINE_RE.match(line)
        if binary is None:
            continue
        op_name = binary.group("op")
        lhs = binary.group("lhs")
        rhs = binary.group("rhs")
        if op_name in COMMUTATIVE_BINARY_OPS and rhs < lhs:
            canonicalizable += 1
            details.append(f"{op_name}: {lhs}, {rhs} -> {rhs}, {lhs}")

    return StableHLOPassResult(
        pass_name="commutative_canonicalization",
        summary="Finds commutative binary operations whose operands can be canonically ordered.",
        metrics={"canonicalizable_operations": canonicalizable},
        details=details[:20],
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


def _canonicalize_commutative_operations(
    stablehlo_text: str,
) -> tuple[str, StableHLOPassResult]:
    new_lines: list[str] = []
    changed_details: list[str] = []
    region_mask = _nested_region_line_mask(stablehlo_text.splitlines())

    for line, in_nested_region in zip(stablehlo_text.splitlines(), region_mask, strict=True):
        if in_nested_region:
            new_lines.append(line)
            continue

        binary = STABLEHLO_BINARY_OP_LINE_RE.match(line)
        if binary is None:
            new_lines.append(line)
            continue

        op_name = binary.group("op")
        lhs = binary.group("lhs")
        rhs = binary.group("rhs")
        if op_name not in COMMUTATIVE_BINARY_OPS or lhs <= rhs:
            new_lines.append(line)
            continue

        rewritten = (
            f"{binary.group('indent')}{binary.group('result')} = "
            f"{op_name} {rhs}, {lhs}{binary.group('tail')}"
        )
        new_lines.append(rewritten)
        changed_details.append(f"{op_name}: {lhs}, {rhs} -> {rhs}, {lhs}")

    return "\n".join(new_lines), StableHLOPassResult(
        pass_name="commutative_canonicalization_transform",
        summary="Canonicalizes operand order for commutative StableHLO binary operations.",
        metrics={"canonicalized_operations": len(changed_details)},
        details=changed_details[:20],
    )

def _eliminate_duplicate_operations(stablehlo_text: str) -> tuple[str, StableHLOPassResult]:
    canonical_values: dict[str, str] = {}
    replacements: dict[str, str] = {}
    removed_details: list[str] = []
    new_lines: list[str] = []
    region_mask = _nested_region_line_mask(stablehlo_text.splitlines())

    for line, in_nested_region in zip(stablehlo_text.splitlines(), region_mask, strict=True):
        if line.strip().startswith("func.func"):
            canonical_values = {}
            replacements = {}
            new_lines.append(line)
            continue
        rewritten_line = _replace_ssa_values(line, replacements)
        if in_nested_region:
            new_lines.append(rewritten_line)
            continue
        if not STABLEHLO_OP_RE.search(rewritten_line):
            new_lines.append(rewritten_line)
            continue

        result_name = _single_result_name(rewritten_line)
        if result_name is None or not _is_cse_safe_operation(rewritten_line):
            new_lines.append(rewritten_line)
            continue

        signature = _cse_operation_signature(rewritten_line)
        canonical_result = canonical_values.get(signature)
        if canonical_result is None:
            canonical_values[signature] = result_name
            new_lines.append(rewritten_line)
            continue

        replacements[result_name] = canonical_result
        removed_details.append(f"{result_name} -> {canonical_result}: {signature[:160]}")

    return "\n".join(new_lines), StableHLOPassResult(
        pass_name="duplicate_operations_transform",
        summary="Eliminates repeated single-result StableHLO operations and rewrites later uses.",
        metrics={
            "removed_operations": len(removed_details),
            "replacement_values": len(replacements),
        },
        details=removed_details[:20],
    )


def _eliminate_dead_results(stablehlo_text: str) -> tuple[str, StableHLOPassResult]:
    lines = stablehlo_text.splitlines()
    removed_details: list[str] = []
    total_removed = 0

    while True:
        use_counts = _ssa_use_counts(lines)
        region_mask = _nested_region_line_mask(lines)
        next_lines: list[str] = []
        removed_this_round = 0
        for line, in_nested_region in zip(lines, region_mask, strict=True):
            result_name = _single_result_name(line)
            if (
                not in_nested_region
                and result_name is not None
                and STABLEHLO_OP_RE.search(line)
                and _is_dce_safe_operation(line)
                and use_counts.get(result_name, 0) == 0
            ):
                removed_this_round += 1
                removed_details.append(f"removed {result_name}: {line.strip()[:160]}")
                continue
            next_lines.append(line)

        lines = next_lines
        total_removed += removed_this_round
        if removed_this_round == 0:
            break

    return "\n".join(lines), StableHLOPassResult(
        pass_name="dead_results_transform",
        summary="Iteratively removes unused single-result StableHLO operations.",
        metrics={"removed_operations": total_removed},
        details=removed_details[:20],
    )


def _replace_ssa_values(line: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return line

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        return replacements.get(value, value)

    return SSA_VALUE_RE.sub(replace, line)


def _nested_region_line_mask(lines: Sequence[str]) -> list[bool]:
    mask: list[bool] = []
    in_nested_region = False
    for line in lines:
        stripped = line.strip()
        mask.append(in_nested_region)
        if stripped.startswith("^bb"):
            in_nested_region = True
        elif in_nested_region and stripped == "}":
            in_nested_region = False
    return mask


def _cse_operation_signature(line: str) -> str:
    signature = LOCATION_RE.sub("", line.strip())
    signature = RESULT_PREFIX_RE.sub("", signature)
    signature = re.sub(r"\s+", " ", signature)
    return signature


def _single_result_name(line: str) -> str | None:
    if "=" not in line:
        return None
    lhs = line.split("=", 1)[0]
    values = SSA_VALUE_RE.findall(lhs)
    if len(values) != 1:
        return None
    return values[0]


def _ssa_use_counts(lines: Sequence[str]) -> dict[str, int]:
    use_counts: dict[str, int] = {}
    for line in lines:
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            defined_values = set(SSA_VALUE_RE.findall(lhs))
            used_values = [value for value in SSA_VALUE_RE.findall(rhs) if value not in defined_values]
        else:
            used_values = SSA_VALUE_RE.findall(line)
        for value in used_values:
            use_counts[value] = use_counts.get(value, 0) + 1
    return use_counts


def _is_cse_safe_operation(line: str) -> bool:
    return _is_dce_safe_operation(line)


def _is_dce_safe_operation(line: str) -> bool:
    op_name = _operation_name(line)
    if op_name == "stablehlo.custom_call":
        return "has_side_effect = true" not in line
    return op_name.startswith("stablehlo.")


def _make_laplacian_lowering_args(config: LaplacianBenchmarkConfig):
    from adscheduler.laplacian_benchmark import _initialize_laplacian_state

    params, _, sample_points = _initialize_laplacian_state(config)
    return params, sample_points


def _make_pinn_lowering_args(config: PINNBenchmarkConfig):
    from adscheduler.pinn_benchmark import _initialize_pinn_state

    params, _, sample_points = _initialize_pinn_state(config)
    return params, sample_points


_PASS_REGISTRY: dict[str, Callable[[str], StableHLOPassResult]] = {
    "commutative_canonicalization": _commutative_canonicalization_pass,
    "duplicate_operations": _duplicate_operation_pass,
    "dead_results": _dead_result_pass,
    "elementwise_fusion_regions": _elementwise_fusion_pass,
    "expensive_operations": _expensive_operation_pass,
}
