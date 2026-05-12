from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from adscheduler.ir_analysis import analyze_closed_jaxpr

Array = jax.Array
PINNParams = tuple[tuple[Array, Array], ...]

SCHEDULE_ROR = "ror"
SCHEDULE_FOR = "for"
SCHEDULE_JACREV = "jacrev"
SCHEDULE_JACFWD = "jacfwd"
SCHEDULE_ROR_REMAT = "ror_remat"
SCHEDULE_FOR_REMAT = "for_remat"
SCHEDULE_JACREV_REMAT = "jacrev_remat"
SCHEDULE_JACFWD_REMAT = "jacfwd_remat"

PINN_SCHEDULES = (
    SCHEDULE_ROR,
    SCHEDULE_FOR,
    SCHEDULE_JACREV,
    SCHEDULE_JACFWD,
    SCHEDULE_ROR_REMAT,
    SCHEDULE_FOR_REMAT,
    SCHEDULE_JACREV_REMAT,
    SCHEDULE_JACFWD_REMAT,
)


@dataclass(frozen=True)
class PINNBenchmarkConfig:
    seed: int = 0
    outer_steps: int = 100
    eval_every: int = 10
    grid_size: int = 16
    input_dim: int = 2
    hidden_layers: int = 128
    hidden_dim: int = 256
    activation: str = "tanh"
    output_dim: int = 1
    target_max_abs_error: float = 1e-4


@dataclass(frozen=True)
class PINNIRScheduleSummary:
    total_equations: int
    max_loop_nesting: int
    num_higher_order_sites: int


@dataclass(frozen=True)
class PINNScheduleBenchmarkResult:
    schedule_name: str
    selected_schedule_name: str
    compile_overhead_sec: float
    avg_step_time_sec: float
    p50_step_time_sec: float
    p90_step_time_sec: float
    peak_host_memory_mb: float
    peak_device_memory_mb: float | None
    final_max_abs_error: float
    best_max_abs_error: float
    iterations_to_target_error: int | None
    time_to_target_error_sec: float | None
    time_to_target_error_with_compile_sec: float | None
    eval_history: list[tuple[int, float]]
    loss: float
    output_summary: str
    ir_summary: PINNIRScheduleSummary
    warmup_overhead_sec: float = 0.0
    used_warmup_cache: bool = False


@dataclass(frozen=True)
class PINNWarmupCandidateProfile:
    schedule_name: str
    compile_overhead_sec: float
    total_step_time_sec: float
    median_step_time_sec: float
    p90_step_time_sec: float
    peak_host_memory_mb: float
    peak_device_memory_mb: float | None
    mean_max_abs_error: float
    final_max_abs_error: float
    stable: bool
    rejected_reason: str | None
    score: float


@dataclass(frozen=True)
class PINNWarmupSelection:
    chosen_schedule_name: str
    profiles: list[PINNWarmupCandidateProfile]
    total_warmup_overhead_sec: float
    from_cache: bool
    cache_signature: str | None


@dataclass(frozen=True)
class PINNSelectionQuality:
    oracle_schedule_name: str
    oracle_estimated_runtime_sec: float
    auto_selected_schedule_name: str
    auto_estimated_runtime_sec: float
    runtime_regret_sec: float
    runtime_regret_pct: float


@dataclass(frozen=True)
class PINNEvaluationReport:
    config: PINNBenchmarkConfig
    fixed_results: list[PINNScheduleBenchmarkResult]
    auto_result: PINNScheduleBenchmarkResult | None
    warmup_selection: PINNWarmupSelection | None
    selection_quality: PINNSelectionQuality | None


def available_pinn_schedule_names() -> tuple[str, ...]:
    return PINN_SCHEDULES


def normalize_pinn_schedule_names(
    schedule_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if schedule_names is None:
        return available_pinn_schedule_names()

    normalized = tuple(dict.fromkeys(schedule_names))
    unknown = sorted(set(normalized) - set(PINN_SCHEDULES))
    if unknown:
        known = ", ".join(available_pinn_schedule_names())
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown PINN schedules: {unknown_names}. Known schedules: {known}")
    return normalized


def run_pinn_evaluation_protocol(
    config: PINNBenchmarkConfig,
    *,
    baseline_schedules: Sequence[str] | None = None,
    include_auto: bool = False,
    auto_candidate_schedules: Sequence[str] | None = None,
    warmup_steps: int = 3,
    memory_budget_mb: float | None = None,
    error_guard_tolerance: float = 0.10,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_pinn_warmup_cache.json",
) -> PINNEvaluationReport:
    _validate_config(config)
    _validate_warmup_args(warmup_steps)
    selected_baselines = normalize_pinn_schedule_names(baseline_schedules)
    params, step_points, sample_points = _initialize_pinn_state(config)
    fixed_results = [
        _run_pinn_schedule_with_state(
            config,
            report_schedule_name=schedule_name,
            schedule_name=schedule_name,
            params=params,
            step_points=step_points,
            sample_points=sample_points,
            warmup_overhead_sec=0.0,
            used_warmup_cache=False,
        )
        for schedule_name in selected_baselines
    ]

    auto_result: PINNScheduleBenchmarkResult | None = None
    warmup_selection: PINNWarmupSelection | None = None
    selection_quality: PINNSelectionQuality | None = None
    if include_auto:
        selected_candidates = normalize_pinn_schedule_names(
            auto_candidate_schedules or selected_baselines
        )
        selected_schedule, warmup_selection = choose_pinn_schedule_via_warmup(
            config,
            params=params,
            points=sample_points,
            candidate_schedules=selected_candidates,
            warmup_steps=warmup_steps,
            memory_budget_mb=memory_budget_mb,
            error_guard_tolerance=error_guard_tolerance,
            use_cache=use_cache,
            cache_path=cache_path,
        )
        auto_result = _run_pinn_schedule_with_state(
            config,
            report_schedule_name="auto",
            schedule_name=selected_schedule,
            params=params,
            step_points=step_points,
            sample_points=sample_points,
            warmup_overhead_sec=warmup_selection.total_warmup_overhead_sec,
            used_warmup_cache=warmup_selection.from_cache,
        )
        selection_quality = compute_pinn_selection_quality(
            config,
            fixed_results=fixed_results,
            auto_result=auto_result,
        )

    return PINNEvaluationReport(
        config=config,
        fixed_results=fixed_results,
        auto_result=auto_result,
        warmup_selection=warmup_selection,
        selection_quality=selection_quality,
    )


def choose_pinn_schedule_via_warmup(
    config: PINNBenchmarkConfig,
    *,
    params: PINNParams,
    points: Array,
    candidate_schedules: Sequence[str],
    warmup_steps: int,
    memory_budget_mb: float | None,
    error_guard_tolerance: float = 0.10,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_pinn_warmup_cache.json",
) -> tuple[str, PINNWarmupSelection]:
    normalized_candidates = normalize_pinn_schedule_names(candidate_schedules)
    cache_signature = _make_warmup_signature(
        config=config,
        candidate_schedules=normalized_candidates,
        warmup_steps=warmup_steps,
        memory_budget_mb=memory_budget_mb,
        param_count=_tree_num_parameters(params),
    )

    cached_choice = None
    if use_cache:
        cache = _load_warmup_cache(cache_path)
        cached_choice = cache.get(cache_signature)
        if cached_choice not in normalized_candidates:
            cached_choice = None

    if cached_choice is not None:
        return cached_choice, PINNWarmupSelection(
            chosen_schedule_name=cached_choice,
            profiles=[],
            total_warmup_overhead_sec=0.0,
            from_cache=True,
            cache_signature=cache_signature,
        )

    warmup_start = time.perf_counter()
    reference_fn = jax.jit(_build_pinn_schedule_fn(SCHEDULE_ROR))
    reference_output = _block_tree(reference_fn(params, points))
    reference_error_scale = max(_tree_max_abs(reference_output), 1.0)

    profiles: list[PINNWarmupCandidateProfile] = []
    for schedule_name in normalized_candidates:
        compile_overhead_sec = 0.0
        timings: list[float] = []
        errors: list[float] = []
        rejected_reason = None
        peak_host_memory_mb = _current_peak_host_memory_mb()
        peak_device_memory_mb = _current_peak_device_memory_mb()

        try:
            schedule_fn = jax.jit(_build_pinn_schedule_fn(schedule_name))
            compiled_fn, compile_overhead_sec = _compile_jitted_schedule(
                schedule_fn,
                params,
                points,
            )
            for _ in range(warmup_steps):
                start = time.perf_counter()
                output = _block_tree(compiled_fn(params, points))
                timings.append(time.perf_counter() - start)
                errors.append(_tree_max_abs_error(output, reference_output))
            peak_host_memory_mb = _current_peak_host_memory_mb()
            peak_device_memory_mb = _current_peak_device_memory_mb()
        except Exception as exc:
            rejected_reason = f"failed:{type(exc).__name__}"

        if rejected_reason is None and memory_budget_mb is not None:
            peak_mem = peak_device_memory_mb or peak_host_memory_mb
            if peak_mem > memory_budget_mb:
                rejected_reason = f"memory_guard:peak={peak_mem:.2f}>{memory_budget_mb:.2f}"

        if rejected_reason is None and errors:
            allowed_error = error_guard_tolerance * reference_error_scale
            final_error = errors[-1]
            if final_error > allowed_error:
                rejected_reason = f"error_guard:final_error={final_error:.6e}>{allowed_error:.6e}"

        total_step_time = float(np.sum(timings)) if timings else float("inf")
        median_step_time = float(np.median(timings)) if timings else float("inf")
        p90_step_time = float(np.percentile(timings, 90)) if timings else float("inf")
        mean_error = float(np.mean(errors)) if errors else float("inf")
        final_error = errors[-1] if errors else float("inf")
        score = compile_overhead_sec + median_step_time * config.outer_steps
        if rejected_reason is not None:
            score = float("inf")

        profiles.append(
            PINNWarmupCandidateProfile(
                schedule_name=schedule_name,
                compile_overhead_sec=compile_overhead_sec,
                total_step_time_sec=total_step_time,
                median_step_time_sec=median_step_time,
                p90_step_time_sec=p90_step_time,
                peak_host_memory_mb=peak_host_memory_mb,
                peak_device_memory_mb=peak_device_memory_mb,
                mean_max_abs_error=mean_error,
                final_max_abs_error=final_error,
                stable=rejected_reason is None,
                rejected_reason=rejected_reason,
                score=score,
            )
        )

    chosen_schedule = _pick_best_pinn_warmup_candidate(profiles)
    total_warmup_overhead_sec = time.perf_counter() - warmup_start
    selection = PINNWarmupSelection(
        chosen_schedule_name=chosen_schedule,
        profiles=profiles,
        total_warmup_overhead_sec=total_warmup_overhead_sec,
        from_cache=False,
        cache_signature=cache_signature,
    )
    if use_cache:
        cache = _load_warmup_cache(cache_path)
        cache[cache_signature] = chosen_schedule
        _save_warmup_cache(cache_path, cache)
    return chosen_schedule, selection


def compute_pinn_selection_quality(
    config: PINNBenchmarkConfig,
    *,
    fixed_results: Sequence[PINNScheduleBenchmarkResult],
    auto_result: PINNScheduleBenchmarkResult,
) -> PINNSelectionQuality:
    if not fixed_results:
        raise ValueError("At least one fixed result is required to compute selection quality")

    def estimated_runtime(result: PINNScheduleBenchmarkResult) -> float:
        return result.compile_overhead_sec + result.avg_step_time_sec * config.outer_steps

    oracle = min(fixed_results, key=estimated_runtime)
    oracle_runtime = estimated_runtime(oracle)
    auto_runtime = estimated_runtime(auto_result)
    regret = auto_runtime - oracle_runtime
    regret_pct = 0.0 if oracle_runtime == 0 else (regret / oracle_runtime) * 100.0
    return PINNSelectionQuality(
        oracle_schedule_name=oracle.schedule_name,
        oracle_estimated_runtime_sec=oracle_runtime,
        auto_selected_schedule_name=auto_result.selected_schedule_name,
        auto_estimated_runtime_sec=auto_runtime,
        runtime_regret_sec=regret,
        runtime_regret_pct=regret_pct,
    )


def build_pinn_schedule_fn(schedule_name: str) -> Callable[[PINNParams, Array], Any]:
    return _build_pinn_schedule_fn(schedule_name)


def pinn_evaluation_report_to_dict(report: PINNEvaluationReport) -> dict[str, Any]:
    return {
        "config": asdict(report.config),
        "fixed_results": [pinn_result_to_dict(result) for result in report.fixed_results],
        "auto_result": (
            pinn_result_to_dict(report.auto_result)
            if report.auto_result is not None
            else None
        ),
        "warmup_selection": (
            {
                **asdict(report.warmup_selection),
                "profiles": [
                    asdict(profile) for profile in report.warmup_selection.profiles
                ],
            }
            if report.warmup_selection is not None
            else None
        ),
        "selection_quality": (
            asdict(report.selection_quality)
            if report.selection_quality is not None
            else None
        ),
    }


def pinn_result_to_dict(result: PINNScheduleBenchmarkResult) -> dict[str, Any]:
    return asdict(result)


def _run_pinn_schedule_with_state(
    config: PINNBenchmarkConfig,
    *,
    report_schedule_name: str,
    schedule_name: str,
    params: PINNParams,
    step_points: Array,
    sample_points: Array,
    warmup_overhead_sec: float,
    used_warmup_cache: bool,
) -> PINNScheduleBenchmarkResult:
    ir_summary = _ir_summary_for_schedule(schedule_name, params, sample_points)
    schedule_fn = jax.jit(_build_pinn_schedule_fn(schedule_name))
    reference_fn = jax.jit(_build_pinn_schedule_fn(SCHEDULE_ROR))
    compiled_fn, compile_overhead_sec = _compile_jitted_schedule(
        schedule_fn,
        params,
        sample_points,
    )
    compiled_reference, _ = _compile_jitted_schedule(
        reference_fn,
        params,
        sample_points,
    )

    timings: list[float] = []
    eval_history: list[tuple[int, float]] = []
    cumulative_step_time_sec = 0.0
    time_to_target_error_sec: float | None = None
    best_error = float("inf")
    final_error = float("inf")
    output = compiled_fn(params, sample_points)
    for step in range(1, config.outer_steps + 1):
        points = step_points[step - 1]
        reference = _block_tree(compiled_reference(params, points))
        start = time.perf_counter()
        output = _block_tree(compiled_fn(params, points))
        step_time_sec = time.perf_counter() - start
        timings.append(step_time_sec)
        cumulative_step_time_sec += step_time_sec

        if step % config.eval_every == 0 or step == config.outer_steps:
            error = _tree_max_abs_error(output, reference)
            final_error = error
            best_error = min(best_error, error)
            eval_history.append((step, error))
            if time_to_target_error_sec is None and error <= config.target_max_abs_error:
                time_to_target_error_sec = cumulative_step_time_sec

    timings_arr = np.asarray(timings, dtype=np.float64)
    iterations_to_target = next(
        (
            step
            for step, error in eval_history
            if error <= config.target_max_abs_error
        ),
        None,
    )
    loss = float(np.asarray(jtu.tree_leaves(output)[0]))
    return PINNScheduleBenchmarkResult(
        schedule_name=report_schedule_name,
        selected_schedule_name=schedule_name,
        compile_overhead_sec=compile_overhead_sec,
        avg_step_time_sec=float(np.mean(timings_arr)),
        p50_step_time_sec=float(np.percentile(timings_arr, 50)),
        p90_step_time_sec=float(np.percentile(timings_arr, 90)),
        peak_host_memory_mb=_current_peak_host_memory_mb(),
        peak_device_memory_mb=_current_peak_device_memory_mb(),
        final_max_abs_error=final_error,
        best_max_abs_error=best_error,
        iterations_to_target_error=iterations_to_target,
        time_to_target_error_sec=time_to_target_error_sec,
        time_to_target_error_with_compile_sec=(
            compile_overhead_sec + time_to_target_error_sec
            if time_to_target_error_sec is not None
            else None
        ),
        eval_history=eval_history,
        loss=loss,
        output_summary=_summarize_output(output),
        ir_summary=ir_summary,
        warmup_overhead_sec=warmup_overhead_sec,
        used_warmup_cache=used_warmup_cache,
    )


def _build_pinn_schedule_fn(schedule_name: str) -> Callable[[PINNParams, Array], Any]:
    normalize_pinn_schedule_names((schedule_name,))
    remat = schedule_name.endswith("_remat")
    base_schedule = schedule_name.removesuffix("_remat")

    def nn_scalar(params: PINNParams, coord: Array) -> Array:
        activations = coord
        for weights, bias in params[:-1]:
            activations = jnp.tanh(activations @ weights + bias)
        final_weights, final_bias = params[-1]
        return jnp.squeeze(activations @ final_weights + final_bias)

    def trial_solution(params: PINNParams, coord: Array) -> Array:
        x, y = coord
        boundary_factor = x * (1.0 - x) * y * (1.0 - y)
        return boundary_factor * nn_scalar(params, coord)

    remat_trial_solution = jax.checkpoint(trial_solution)

    def poisson_residual_at_point(params: PINNParams, coord: Array) -> Array:
        solution_fn = remat_trial_solution if remat else trial_solution
        scalar_fn = lambda z: solution_fn(params, z)

        if base_schedule == SCHEDULE_ROR:
            hessian = jax.jacrev(jax.grad(scalar_fn))(coord)
            laplacian = jnp.trace(hessian)
        elif base_schedule == SCHEDULE_FOR:
            grad_fn = jax.grad(scalar_fn)
            basis = jnp.eye(coord.shape[0], dtype=coord.dtype)

            def diagonal_entry(direction: Array) -> Array:
                _, hvp = jax.jvp(grad_fn, (coord,), (direction,))
                return jnp.dot(direction, hvp)

            laplacian = jnp.sum(jax.vmap(diagonal_entry)(basis))
        elif base_schedule == SCHEDULE_JACREV:
            hessian = jax.jacrev(jax.jacrev(scalar_fn))(coord)
            laplacian = jnp.trace(hessian)
        elif base_schedule == SCHEDULE_JACFWD:
            hessian = jax.jacfwd(jax.jacfwd(scalar_fn))(coord)
            laplacian = jnp.trace(hessian)
        else:
            raise ValueError(f"Unknown PINN schedule: {schedule_name}")

        forcing = jnp.sin(jnp.pi * coord[0]) * jnp.sin(jnp.pi * coord[1])
        return laplacian + forcing

    def pinn_loss(params: PINNParams, points: Array) -> Array:
        residuals = jax.vmap(lambda coord: poisson_residual_at_point(params, coord))(points)
        return jnp.mean(residuals**2)

    def scheduled_pinn_task(params: PINNParams, points: Array):
        return jax.value_and_grad(pinn_loss)(params, points)

    return scheduled_pinn_task


def _initialize_pinn_state(
    config: PINNBenchmarkConfig,
) -> tuple[PINNParams, Array, Array]:
    params = _init_pinn_params(config)
    step_points = jnp.stack(
        [_make_collocation_points(config, fold=step) for step in range(config.outer_steps)],
        axis=0,
    )
    sample_points = _make_collocation_points(config, fold=0)
    return params, step_points, sample_points


def _init_pinn_params(config: PINNBenchmarkConfig) -> PINNParams:
    key = jax.random.PRNGKey(config.seed)
    layer_dims = (
        [config.input_dim]
        + [config.hidden_dim] * config.hidden_layers
        + [config.output_dim]
    )
    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    for layer_key, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:]):
        weights = jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32)
        weights = weights / jnp.sqrt(jnp.asarray(in_dim, dtype=jnp.float32))
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        layers.append((weights, bias))
    return tuple(layers)


def _make_collocation_points(config: PINNBenchmarkConfig, *, fold: int) -> Array:
    if config.input_dim != 2:
        raise ValueError("Poisson PINN collocation currently requires input_dim=2")
    base_grid = jnp.linspace(0.1, 0.9, config.grid_size, dtype=jnp.float32)
    if fold == 0:
        grid = base_grid
    else:
        offset = 0.01 * jnp.sin(jnp.asarray(fold, dtype=jnp.float32))
        grid = jnp.clip(base_grid + offset, 0.05, 0.95)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid, indexing="ij")
    return jnp.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)


def _compile_jitted_schedule(jitted_fn, params: PINNParams, points: Array):
    compile_start = time.perf_counter()
    try:
        compiled_fn = jitted_fn.lower(params, points).compile()
    except AttributeError:
        compiled_fn = jitted_fn
    output = compiled_fn(params, points)
    _block_tree(output)
    return compiled_fn, time.perf_counter() - compile_start


def _ir_summary_for_schedule(
    schedule_name: str,
    params: PINNParams,
    sample_points: Array,
) -> PINNIRScheduleSummary:
    schedule_fn = _build_pinn_schedule_fn(schedule_name)
    closed_jaxpr = jax.make_jaxpr(schedule_fn)(params, sample_points)
    features = analyze_closed_jaxpr(closed_jaxpr, assume_outer_grad=True)
    return PINNIRScheduleSummary(
        total_equations=features.total_equations,
        max_loop_nesting=features.max_loop_nesting,
        num_higher_order_sites=len(features.higher_order_sites),
    )


def _block_tree(tree):
    return jtu.tree_map(lambda x: x.block_until_ready(), tree)


def _summarize_output(output) -> str:
    leaves = jtu.tree_leaves(output)
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


def _tree_max_abs(tree) -> float:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return 0.0
    return max(float(np.max(np.abs(np.asarray(leaf)))) for leaf in leaves)


def _tree_max_abs_error(lhs, rhs) -> float:
    lhs_leaves = jtu.tree_leaves(lhs)
    rhs_leaves = jtu.tree_leaves(rhs)
    if len(lhs_leaves) != len(rhs_leaves):
        return float("inf")

    max_error = 0.0
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        lhs_arr = np.asarray(lhs_leaf)
        rhs_arr = np.asarray(rhs_leaf)
        if lhs_arr.shape != rhs_arr.shape:
            return float("inf")
        max_error = max(max_error, float(np.max(np.abs(lhs_arr - rhs_arr))))
    return max_error


def _tree_num_parameters(tree) -> int:
    return sum(int(np.size(leaf)) for leaf in jtu.tree_leaves(tree))


def _pick_best_pinn_warmup_candidate(
    profiles: Sequence[PINNWarmupCandidateProfile],
) -> str:
    feasible = [
        profile
        for profile in profiles
        if profile.rejected_reason is None and np.isfinite(profile.score)
    ]
    if feasible:
        return min(feasible, key=lambda profile: profile.score).schedule_name

    finite = [profile for profile in profiles if np.isfinite(profile.score)]
    if finite:
        return min(finite, key=lambda profile: profile.score).schedule_name

    raise ValueError("All PINN warmup candidates failed or were rejected")


def _current_peak_host_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _current_peak_device_memory_mb() -> float | None:
    try:
        backend = jax.default_backend()
        if backend == "cpu":
            return None
        devices = jax.devices()
        memory_stats = getattr(devices[0], "memory_stats", None)
        if memory_stats is None:
            return None
        stats = memory_stats()
        if not stats:
            return None
        peak_bytes = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        if peak_bytes is None:
            return None
        return float(peak_bytes) / (1024 * 1024)
    except Exception:
        return None


def _make_warmup_signature(
    *,
    config: PINNBenchmarkConfig,
    candidate_schedules: Sequence[str],
    warmup_steps: int,
    memory_budget_mb: float | None,
    param_count: int,
) -> str:
    payload = {
        "config": asdict(config),
        "candidate_schedules": list(candidate_schedules),
        "warmup_steps": warmup_steps,
        "memory_budget_mb": memory_budget_mb,
        "param_count": param_count,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_warmup_cache(cache_path: str) -> dict[str, str]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_warmup_cache(cache_path: str, cache: dict[str, str]) -> None:
    path = Path(cache_path)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _validate_config(config: PINNBenchmarkConfig) -> None:
    if config.outer_steps < 1:
        raise ValueError("outer_steps must be >= 1")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if config.input_dim != 2:
        raise ValueError("input_dim must be 2 for the Poisson PINN workload")
    if config.hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if config.hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    if config.activation != "tanh":
        raise ValueError("Only activation='tanh' is currently supported")
    if config.output_dim != 1:
        raise ValueError("output_dim must be 1 for the Poisson PINN workload")
    if config.target_max_abs_error < 0:
        raise ValueError("target_max_abs_error must be >= 0")


def _validate_warmup_args(warmup_steps: int) -> None:
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
