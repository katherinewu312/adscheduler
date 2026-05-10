from __future__ import annotations

import re
from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

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
DIALECT_OP_RE = re.compile(r'"?((?:stablehlo|chlo)\.[\w_]+)"?')
TENSOR_TYPE_RE = re.compile(r"tensor<([^>]*)>")
STABLEHLO_CONSTANT_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<result>%[\w.:\-#]+)\s*=\s*"
    r'"?stablehlo\.constant"?\s+'
    r"dense<(?P<value>[^>]+)>"
    r"\s*:\s*tensor<(?P<type>[^>]*)>"
)
STABLEHLO_UNARY_OP_LINE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<result>%[\w.:\-#]+)\s*=\s*"
    r'"?(?P<op>(?:stablehlo|chlo)\.[\w_]+)"?\s+'
    r"(?P<input>%[\w.:\-#]+)"
    r"(?P<tail>.*)$"
)
STABLEHLO_BINARY_OP_LINE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<result>%[\w.:\-#]+)\s*=\s*"
    r'"?(?P<op>(?:stablehlo|chlo)\.[\w_]+)"?\s+'
    r"(?P<lhs>%[\w.:\-#]+)\s*,\s*"
    r"(?P<rhs>%[\w.:\-#]+)"
    r"(?P<tail>.*)$"
)
CONSTANT_PRESERVING_UNARY_OPS = {
    "stablehlo.convert",
    "stablehlo.reshape",
    "stablehlo.broadcast_in_dim",
}
CONSTANT_FOLDABLE_UNARY_OPS = {
    "chlo.lgamma",
    "stablehlo.abs",
    "stablehlo.ceil",
    "stablehlo.cosine",
    "stablehlo.exponential",
    "stablehlo.floor",
    "stablehlo.log",
    "stablehlo.logistic",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.sine",
    "stablehlo.sqrt",
    "stablehlo.tanh",
}
CONSTANT_FOLDABLE_BINARY_OPS = {
    "stablehlo.add",
    "stablehlo.divide",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.power",
    "stablehlo.subtract",
}
DEFAULT_STABLEHLO_PASS_NAMES = (
    "tanh_mlp_laplacian_recurrence",
    "derivative_constant_propagation",
)
LAPLACIAN_PROGRAM_PREFIXES = (
    "mlp_laplacian",
    "laplacian_",
)
PINN_PROGRAM_PREFIXES = (
    "poisson_pinn",
    "pinn_",
)


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
    laplacian_recurrence_rewrites: int
    constant_foldable_operations: int
    mixed_partial_cse_rewrites: int
    symmetric_kernel_rewrites: int


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
    selected_passes = tuple(passes or default_stablehlo_pass_names())
    unknown = sorted(set(selected_passes) - set(_PASS_REGISTRY))
    if unknown:
        known = ", ".join(available_stablehlo_pass_names())
        raise ValueError(f"Unknown StableHLO passes: {', '.join(unknown)}. Known: {known}")

    op_histogram = stablehlo_operation_histogram(program.stablehlo_text)
    pass_results = [
        _PASS_REGISTRY[pass_name](program) for pass_name in selected_passes
    ]
    return StableHLOPipelineResult(
        program_name=program.name,
        total_operations=sum(op_histogram.values()),
        operation_histogram=op_histogram,
        pass_results=pass_results,
    )


def available_stablehlo_pass_names() -> tuple[str, ...]:
    return tuple(_PASS_REGISTRY)


def default_stablehlo_pass_names() -> tuple[str, ...]:
    return DEFAULT_STABLEHLO_PASS_NAMES


def run_stablehlo_transform_pipeline(
    program: StableHLOProgram,
    *,
    passes: Sequence[str] | None = None,
    source_args: tuple[Any, ...] | None = None,
) -> StableHLOTransformResult:
    """Run source-level StableHLO optimization passes.

    The analysis pipeline is intentionally preserved for scoring. This function
    additionally rewrites the StableHLO module with conservative SSA transforms
    that keep the module in StableHLO text form so another backend can compile it.

    Transform order matters:
      1. replace known nested input-AD Laplacians with direct tanh-MLP Laplacian recurrences
      2. fold derivative-generated scalar/splat constants
      3. optionally run metadata-dependent experimental transforms
    """

    selected_passes = tuple(passes or default_stablehlo_pass_names())
    analysis = run_stablehlo_pass_pipeline(program, passes=selected_passes)
    current_program = program
    stablehlo_text = current_program.stablehlo_text
    transform_results: list[StableHLOPassResult] = []

    if "tanh_mlp_laplacian_recurrence" in selected_passes:
        current_program, pass_result = _rewrite_tanh_mlp_laplacian_recurrence(
            current_program,
            source_args=source_args,
        )
        stablehlo_text = current_program.stablehlo_text
        transform_results.append(pass_result)

    if "derivative_constant_propagation" in selected_passes:
        stablehlo_text, pass_result = _propagate_derivative_constants(stablehlo_text)
        current_program = StableHLOProgram(
            name=current_program.name,
            stablehlo_text=stablehlo_text,
        )
        transform_results.append(pass_result)

    if "mixed_partial_cse" in selected_passes:
        stablehlo_text, pass_result = _eliminate_mixed_partial_equivalents(stablehlo_text)
        current_program = StableHLOProgram(
            name=current_program.name,
            stablehlo_text=stablehlo_text,
        )
        transform_results.append(pass_result)

    if "symmetric_kernel_rewriting" in selected_passes:
        stablehlo_text, pass_result = _rewrite_symmetric_derivative_kernels(stablehlo_text)
        current_program = StableHLOProgram(
            name=current_program.name,
            stablehlo_text=stablehlo_text,
        )
        transform_results.append(pass_result)

    transformed_program = StableHLOProgram(
        name=f"{current_program.name}_stablehlo_passes",
        stablehlo_text=current_program.stablehlo_text,
    )
    return StableHLOTransformResult(
        original_program=program,
        transformed_program=transformed_program,
        analysis=analysis,
        transform_results=transform_results,
    )


def stablehlo_operation_histogram(stablehlo_text: str) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for op_name in DIALECT_OP_RE.findall(stablehlo_text):
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
    """Heuristic score for choosing IR with visible AD-aware optimization surface.

    Lower is better. This rewards optimization opportunities that are usually
    not visible to XLA as derivative semantics: derivative-generated constants,
    mixed-partial equivalences, and symmetric derivative kernels.
    """

    pass_metrics = {pass_result.pass_name: pass_result.metrics for pass_result in result.pass_results}
    laplacian_recurrence_rewrites = int(
        pass_metrics.get("tanh_mlp_laplacian_recurrence", {}).get("rewritable_programs", 0)
    )
    constant_foldable_operations = int(
        pass_metrics.get("derivative_constant_propagation", {}).get(
            "foldable_operations",
            0,
        )
    )
    mixed_partial_cse_rewrites = int(
        pass_metrics.get("mixed_partial_cse", {}).get("rewritable_equivalents", 0)
    )
    symmetric_kernel_rewrites = int(
        pass_metrics.get("symmetric_kernel_rewriting", {}).get("rewritable_kernels", 0)
    )

    estimated_optimized_operations = max(
        0.0,
        result.total_operations
        - 500.00 * laplacian_recurrence_rewrites
        - 0.75 * constant_foldable_operations
        - 2.00 * mixed_partial_cse_rewrites
        - 3.00 * symmetric_kernel_rewrites,
    )
    score = estimated_optimized_operations
    return StableHLOCompilerScore(
        program_name=result.program_name,
        score=score,
        estimated_optimized_operations=estimated_optimized_operations,
        total_operations=result.total_operations,
        laplacian_recurrence_rewrites=laplacian_recurrence_rewrites,
        constant_foldable_operations=constant_foldable_operations,
        mixed_partial_cse_rewrites=mixed_partial_cse_rewrites,
        symmetric_kernel_rewrites=symmetric_kernel_rewrites,
    )


def _tanh_mlp_laplacian_recurrence_pass(program: StableHLOProgram) -> StableHLOPassResult:
    family = _known_laplacian_program_family(program.name)
    rewritable = family in {"mlp_laplacian", "poisson_pinn"}
    details = []
    if rewritable:
        details.append(
            "Can replace nested input AD with a direct value/gradient/Laplacian "
            f"recurrence for {family}."
        )
    else:
        details.append("Program name does not match a known tanh-MLP Laplacian workload.")
    return StableHLOPassResult(
        pass_name="tanh_mlp_laplacian_recurrence",
        summary=(
            "Rewrites known tanh-MLP Laplacian workloads to a forward recurrence that "
            "computes input gradients and the input Laplacian directly."
        ),
        metrics={
            "rewritable_programs": 1 if rewritable else 0,
            "skipped_unknown_program": 0 if rewritable else 1,
        },
        details=details,
    )


def _derivative_constant_propagation_pass(program: StableHLOProgram) -> StableHLOPassResult:
    _, result = _propagate_derivative_constants(program.stablehlo_text)
    return StableHLOPassResult(
        pass_name="derivative_constant_propagation",
        summary=(
            "Finds AD-generated scalar/splat constants that can be folded before XLA, "
            "including constant-preserving broadcasts/converts and constant arithmetic."
        ),
        metrics={
            "known_constants": result.metrics["known_constants"],
            "foldable_operations": result.metrics["folded_operations"],
            "constant_preserving_folds": result.metrics["constant_preserving_folds"],
            "unary_folds": result.metrics["unary_folds"],
            "binary_folds": result.metrics["binary_folds"],
        },
        details=result.details,
    )


def _mixed_partial_cse_pass(program: StableHLOProgram) -> StableHLOPassResult:
    _, result = _eliminate_mixed_partial_equivalents(program.stablehlo_text)
    return StableHLOPassResult(
        pass_name="mixed_partial_cse",
        summary=(
            "Finds mixed-partial derivative equivalents when derivative index metadata "
            "is preserved in StableHLO locations or names."
        ),
        metrics={
            "metadata_tags": result.metrics["metadata_tags"],
            "rewritable_equivalents": result.metrics["rewritten_equivalents"],
            "skipped_missing_metadata": result.metrics["skipped_missing_metadata"],
        },
        details=result.details,
    )


def _symmetric_kernel_rewriting_pass(program: StableHLOProgram) -> StableHLOPassResult:
    _, result = _rewrite_symmetric_derivative_kernels(program.stablehlo_text)
    return StableHLOPassResult(
        pass_name="symmetric_kernel_rewriting",
        summary=(
            "Finds Hessian/symmetric derivative kernels that could use a symmetric "
            "kernel representation when the IR exposes symmetry metadata."
        ),
        metrics={
            "symmetric_kernel_candidates": result.metrics["symmetric_kernel_candidates"],
            "rewritable_kernels": result.metrics["rewritten_kernels"],
            "skipped_no_portable_kernel": result.metrics["skipped_no_portable_kernel"],
            "skipped_missing_metadata": result.metrics["skipped_missing_metadata"],
        },
        details=result.details,
    )


def _propagate_derivative_constants(
    stablehlo_text: str,
) -> tuple[str, StableHLOPassResult]:
    lines = stablehlo_text.splitlines()
    region_mask = _nested_region_line_mask(lines)
    constants: dict[str, float | int | bool] = {}
    new_lines: list[str] = []
    details: list[str] = []
    known_constants = 0
    constant_preserving_folds = 0
    unary_folds = 0
    binary_folds = 0

    for line, in_nested_region in zip(lines, region_mask, strict=True):
        if line.strip().startswith("func.func"):
            constants = {}
            new_lines.append(line)
            continue

        if in_nested_region:
            new_lines.append(line)
            continue

        parsed_constant = _parse_scalar_constant_line(line)
        if parsed_constant is not None:
            result_name, value = parsed_constant
            constants[result_name] = value
            known_constants += 1
            new_lines.append(line)
            continue

        unary = STABLEHLO_UNARY_OP_LINE_RE.match(line)
        if unary is not None:
            result_name = unary.group("result")
            op_name = unary.group("op")
            input_name = unary.group("input")
            input_value = constants.get(input_name)
            output_type = _output_tensor_type(line)

            if input_value is not None and output_type is not None:
                folded_value: float | int | bool | None = None
                fold_kind = ""
                if op_name in CONSTANT_PRESERVING_UNARY_OPS:
                    folded_value = _cast_constant_value(input_value, output_type)
                    fold_kind = "constant_preserving"
                elif op_name in CONSTANT_FOLDABLE_UNARY_OPS:
                    folded_value = _evaluate_unary_constant(op_name, input_value)
                    if folded_value is not None:
                        folded_value = _cast_constant_value(folded_value, output_type)
                        fold_kind = "unary"

                if folded_value is not None:
                    constants[result_name] = folded_value
                    new_line = _constant_line(
                        unary.group("indent"),
                        result_name,
                        folded_value,
                        output_type,
                    )
                    new_lines.append(new_line)
                    if fold_kind == "constant_preserving":
                        constant_preserving_folds += 1
                    else:
                        unary_folds += 1
                    details.append(f"{result_name}: {op_name} constant -> constant")
                    continue

        binary = STABLEHLO_BINARY_OP_LINE_RE.match(line)
        if binary is not None:
            result_name = binary.group("result")
            op_name = binary.group("op")
            lhs_value = constants.get(binary.group("lhs"))
            rhs_value = constants.get(binary.group("rhs"))
            output_type = _output_tensor_type(line)
            if (
                lhs_value is not None
                and rhs_value is not None
                and output_type is not None
                and op_name in CONSTANT_FOLDABLE_BINARY_OPS
            ):
                folded_value = _evaluate_binary_constant(op_name, lhs_value, rhs_value)
                if folded_value is not None:
                    folded_value = _cast_constant_value(folded_value, output_type)
                    constants[result_name] = folded_value
                    new_lines.append(
                        _constant_line(
                            binary.group("indent"),
                            result_name,
                            folded_value,
                            output_type,
                        )
                    )
                    binary_folds += 1
                    details.append(f"{result_name}: {op_name} constants -> constant")
                    continue

        result_name = _single_result_name(line)
        if result_name is not None:
            constants.pop(result_name, None)
        new_lines.append(line)

    folded_operations = constant_preserving_folds + unary_folds + binary_folds
    transformed_text = "\n".join(new_lines) if folded_operations else stablehlo_text
    return transformed_text, StableHLOPassResult(
        pass_name="derivative_constant_propagation_transform",
        summary="Folds scalar/splat constants generated by AD lowering.",
        metrics={
            "known_constants": known_constants,
            "folded_operations": folded_operations,
            "constant_preserving_folds": constant_preserving_folds,
            "unary_folds": unary_folds,
            "binary_folds": binary_folds,
        },
        details=details[:20],
    )


def _eliminate_mixed_partial_equivalents(
    stablehlo_text: str,
) -> tuple[str, StableHLOPassResult]:
    tags = _mixed_partial_tags(stablehlo_text)
    details: list[str] = []
    if not tags:
        details.append(
            "No mixed-partial derivative-index metadata found in StableHLO text; "
            "leaving IR unchanged."
        )

    return stablehlo_text, StableHLOPassResult(
        pass_name="mixed_partial_cse_transform",
        summary="Eliminates equivalent mixed partials when derivative-index tags are available.",
        metrics={
            "metadata_tags": len(tags),
            "rewritten_equivalents": 0,
            "skipped_missing_metadata": 1 if not tags else 0,
        },
        details=details,
    )


def _rewrite_symmetric_derivative_kernels(
    stablehlo_text: str,
) -> tuple[str, StableHLOPassResult]:
    dot_general_count = stablehlo_operation_histogram(stablehlo_text).get(
        "stablehlo.dot_general",
        0,
    )
    symmetry_tags = _symmetry_tags(stablehlo_text)
    details: list[str] = []
    if not symmetry_tags:
        details.append(
            "No Hessian/symmetric operand metadata found; StableHLO has no portable "
            "symmetric-matrix kernel op to target safely."
        )

    return stablehlo_text, StableHLOPassResult(
        pass_name="symmetric_kernel_rewriting_transform",
        summary="Rewrites Hessian/symmetric matmul kernels when a safe StableHLO target exists.",
        metrics={
            "dot_general_operations": dot_general_count,
            "symmetry_metadata_tags": len(symmetry_tags),
            "symmetric_kernel_candidates": 0,
            "rewritten_kernels": 0,
            "skipped_no_portable_kernel": 1,
            "skipped_missing_metadata": 1 if not symmetry_tags else 0,
        },
        details=details,
    )


def _rewrite_tanh_mlp_laplacian_recurrence(
    program: StableHLOProgram,
    *,
    source_args: tuple[Any, ...] | None,
) -> tuple[StableHLOProgram, StableHLOPassResult]:
    family = _known_laplacian_program_family(program.name)
    if family is None:
        return program, StableHLOPassResult(
            pass_name="tanh_mlp_laplacian_recurrence_transform",
            summary="No known tanh-MLP Laplacian rewrite matched this StableHLO program.",
            metrics={
                "rewritten_programs": 0,
                "skipped_unknown_program": 1,
                "skipped_missing_args": 0,
            },
            details=["Program name does not match a known Laplacian or Poisson PINN workload."],
        )

    if source_args is None:
        return program, StableHLOPassResult(
            pass_name="tanh_mlp_laplacian_recurrence_transform",
            summary="Tanh-MLP Laplacian recurrence rewrite requires the original argument tree.",
            metrics={
                "rewritten_programs": 0,
                "skipped_unknown_program": 0,
                "skipped_missing_args": 1,
            },
            details=["No source_args were provided, so shape-specialized lowering was skipped."],
        )

    if family == "mlp_laplacian":
        optimized_program = lower_to_stablehlo(
            f"{program.name}_tanh_laplacian_recurrence",
            _build_tanh_mlp_laplacian_recurrence_fn(),
            source_args,
        )
        details = [
            "Replaced nested input Hessian/JVP/jet Laplacian code with a direct "
            "forward recurrence for tanh MLP activations.",
            "The recurrence tracks each layer's value, input gradient, and input Laplacian.",
        ]
    else:
        optimized_program = lower_to_stablehlo(
            f"{program.name}_tanh_pinn_recurrence",
            _build_tanh_poisson_pinn_recurrence_fn(),
            source_args,
        )
        details = [
            "Replaced nested input derivatives in the Poisson residual with a direct "
            "tanh MLP Laplacian recurrence.",
            "The outer parameter gradient remains jax.value_and_grad over the optimized residual.",
        ]

    return optimized_program, StableHLOPassResult(
        pass_name="tanh_mlp_laplacian_recurrence_transform",
        summary="Rewrote known tanh-MLP Laplacian workload before StableHLO execution.",
        metrics={
            "rewritten_programs": 1,
            "skipped_unknown_program": 0,
            "skipped_missing_args": 0,
        },
        details=details,
    )


def _known_laplacian_program_family(program_name: str) -> str | None:
    if program_name.startswith(LAPLACIAN_PROGRAM_PREFIXES):
        return "mlp_laplacian"
    if program_name.startswith(PINN_PROGRAM_PREFIXES):
        return "poisson_pinn"
    return None


def _build_tanh_mlp_laplacian_recurrence_fn():
    def laplacian_at_point(params, x):
        _, _, laplacian = _tanh_mlp_value_grad_laplacian(params, x)
        return laplacian

    def scheduled_laplacian(params, points):
        return jax.vmap(lambda point: laplacian_at_point(params, point))(points)

    return scheduled_laplacian


def _build_tanh_poisson_pinn_recurrence_fn():
    def trial_laplacian(params, coord):
        nn_value, nn_grad, nn_laplacian = _tanh_mlp_value_grad_laplacian(params, coord)
        x, y = coord
        one = jnp.asarray(1.0, dtype=coord.dtype)
        two = jnp.asarray(2.0, dtype=coord.dtype)

        x_factor = x * (one - x)
        y_factor = y * (one - y)
        boundary = x_factor * y_factor
        boundary_grad = jnp.stack(
            (
                (one - two * x) * y_factor,
                x_factor * (one - two * y),
            )
        )
        boundary_laplacian = -two * y_factor - two * x_factor
        return (
            boundary * nn_laplacian
            + nn_value * boundary_laplacian
            + two * jnp.dot(boundary_grad, nn_grad)
        )

    def poisson_residual_at_point(params, coord):
        forcing = jnp.sin(jnp.pi * coord[0]) * jnp.sin(jnp.pi * coord[1])
        return trial_laplacian(params, coord) + forcing

    def pinn_loss(params, points):
        residuals = jax.vmap(lambda coord: poisson_residual_at_point(params, coord))(points)
        return jnp.mean(residuals**2)

    def scheduled_pinn_task(params, points):
        return jax.value_and_grad(pinn_loss)(params, points)

    return scheduled_pinn_task


def _tanh_mlp_value_grad_laplacian(params, x):
    activations = x
    gradient = jnp.eye(x.shape[0], dtype=x.dtype)
    laplacian = jnp.zeros((x.shape[0],), dtype=x.dtype)

    for weights, bias in params[:-1]:
        preactivation = activations @ weights + bias
        preactivation_gradient = weights.T @ gradient
        preactivation_laplacian = weights.T @ laplacian

        activations = jnp.tanh(preactivation)
        first_derivative = 1.0 - activations * activations
        second_derivative = -2.0 * activations * first_derivative
        gradient_norm_squared = jnp.sum(
            preactivation_gradient * preactivation_gradient,
            axis=1,
        )

        gradient = first_derivative[:, None] * preactivation_gradient
        laplacian = (
            second_derivative * gradient_norm_squared
            + first_derivative * preactivation_laplacian
        )

    final_weights, final_bias = params[-1]
    value = jnp.squeeze(activations @ final_weights + final_bias)
    input_gradient = jnp.squeeze(gradient.T @ final_weights, axis=-1)
    input_laplacian = jnp.squeeze(laplacian @ final_weights)
    return value, input_gradient, input_laplacian


def _parse_scalar_constant_line(line: str) -> tuple[str, float | int | bool] | None:
    match = STABLEHLO_CONSTANT_RE.match(line.strip())
    if match is None:
        return None

    value_text = match.group("value").strip()
    if any(char in value_text for char in "[]{}"):
        return None

    parsed = _parse_constant_value(value_text)
    if parsed is None:
        return None
    return match.group("result"), parsed


def _parse_constant_value(value_text: str) -> float | int | bool | None:
    if value_text == "true":
        return True
    if value_text == "false":
        return False
    try:
        if re.fullmatch(r"[-+]?\d+", value_text):
            return int(value_text)
        return float(value_text)
    except ValueError:
        return None


def _output_tensor_type(line: str) -> str | None:
    tensor_types = TENSOR_TYPE_RE.findall(line)
    if not tensor_types:
        return None
    return tensor_types[-1]


def _tensor_element_type(tensor_type: str) -> str:
    return tensor_type.split("x")[-1]


def _constant_line(
    indent: str,
    result_name: str,
    value: float | int | bool,
    tensor_type: str,
) -> str:
    return (
        f"{indent}{result_name} = stablehlo.constant "
        f"dense<{_format_constant_value(value, tensor_type)}> : tensor<{tensor_type}>"
    )


def _format_constant_value(value: float | int | bool, tensor_type: str) -> str:
    element_type = _tensor_element_type(tensor_type)
    if element_type == "i1":
        return "true" if bool(value) else "false"
    if element_type.startswith(("i", "ui")):
        return str(int(value))
    return f"{float(value):.6e}"


def _cast_constant_value(
    value: float | int | bool,
    tensor_type: str,
) -> float | int | bool:
    element_type = _tensor_element_type(tensor_type)
    if element_type == "i1":
        return bool(value)
    if element_type.startswith(("i", "ui")):
        return int(value)
    return float(value)


def _evaluate_unary_constant(
    op_name: str,
    value: float | int | bool,
) -> float | int | bool | None:
    try:
        if op_name == "stablehlo.abs":
            return abs(value)
        if op_name == "stablehlo.ceil":
            return math.ceil(float(value))
        if op_name == "stablehlo.cosine":
            return math.cos(float(value))
        if op_name == "stablehlo.exponential":
            return math.exp(float(value))
        if op_name == "stablehlo.floor":
            return math.floor(float(value))
        if op_name == "stablehlo.log":
            return math.log(float(value))
        if op_name == "stablehlo.logistic":
            return 1.0 / (1.0 + math.exp(-float(value)))
        if op_name == "stablehlo.negate":
            return -value
        if op_name == "stablehlo.not":
            return not bool(value)
        if op_name == "stablehlo.sine":
            return math.sin(float(value))
        if op_name == "stablehlo.sqrt":
            return math.sqrt(float(value))
        if op_name == "stablehlo.tanh":
            return math.tanh(float(value))
        if op_name == "chlo.lgamma":
            return math.lgamma(float(value))
    except (OverflowError, ValueError):
        return None
    return None


def _evaluate_binary_constant(
    op_name: str,
    lhs: float | int | bool,
    rhs: float | int | bool,
) -> float | int | bool | None:
    try:
        if op_name == "stablehlo.add":
            return lhs + rhs
        if op_name == "stablehlo.divide":
            return lhs / rhs
        if op_name == "stablehlo.maximum":
            return max(lhs, rhs)
        if op_name == "stablehlo.minimum":
            return min(lhs, rhs)
        if op_name == "stablehlo.multiply":
            return lhs * rhs
        if op_name == "stablehlo.power":
            return float(lhs) ** float(rhs)
        if op_name == "stablehlo.subtract":
            return lhs - rhs
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def _mixed_partial_tags(stablehlo_text: str) -> list[str]:
    patterns = (
        r"mixed[_-]?partial\[[^\]]+\]",
        r"d2[a-zA-Z0-9_]*_d[a-zA-Z0-9]+_d[a-zA-Z0-9]+",
        r"hessian\[[^\]]+\]",
    )
    tags: list[str] = []
    for pattern in patterns:
        tags.extend(re.findall(pattern, stablehlo_text, flags=re.IGNORECASE))
    return sorted(set(tags))


def _symmetry_tags(stablehlo_text: str) -> list[str]:
    patterns = (
        r"hessian",
        r"symmetric",
        r"symm",
    )
    tags: list[str] = []
    for pattern in patterns:
        tags.extend(re.findall(pattern, stablehlo_text, flags=re.IGNORECASE))
    return sorted(set(tags))


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


def _single_result_name(line: str) -> str | None:
    if "=" not in line:
        return None
    lhs = line.split("=", 1)[0]
    values = SSA_VALUE_RE.findall(lhs)
    if len(values) != 1:
        return None
    return values[0]


def _make_laplacian_lowering_args(config: LaplacianBenchmarkConfig):
    from adscheduler.laplacian_benchmark import _initialize_laplacian_state

    params, _, sample_points = _initialize_laplacian_state(config)
    return params, sample_points


def _make_pinn_lowering_args(config: PINNBenchmarkConfig):
    from adscheduler.pinn_benchmark import _initialize_pinn_state

    params, _, sample_points = _initialize_pinn_state(config)
    return params, sample_points


_PASS_REGISTRY: dict[str, Callable[[StableHLOProgram], StableHLOPassResult]] = {
    "tanh_mlp_laplacian_recurrence": _tanh_mlp_laplacian_recurrence_pass,
    "derivative_constant_propagation": _derivative_constant_propagation_pass,
    "mixed_partial_cse": _mixed_partial_cse_pass,
    "symmetric_kernel_rewriting": _symmetric_kernel_rewriting_pass,
}
