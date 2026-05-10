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
MLPParams = tuple[tuple[Array, Array], ...]

SCHEDULE_ROR = "ror"
SCHEDULE_FOR = "for"
SCHEDULE_JACREV = "jacrev"
SCHEDULE_JACFWD = "jacfwd"
SCHEDULE_ROR_REMAT = "ror_remat"
SCHEDULE_FOR_REMAT = "for_remat"
SCHEDULE_JACREV_REMAT = "jacrev_remat"
SCHEDULE_JACFWD_REMAT = "jacfwd_remat"

LAPLACIAN_SCHEDULES = (
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
class LaplacianBenchmarkConfig:
    seed: int = 0
    outer_steps: int = 100
    eval_every: int = 10
    num_points: int = 64
    input_dim: int = 3
    hidden_dim: int = 128
    hidden_layers: int = 64
    target_max_abs_error: float = 1e-4


@dataclass(frozen=True)
class LaplacianIRScheduleSummary:
    total_equations: int
    max_loop_nesting: int
    num_higher_order_sites: int


@dataclass(frozen=True)
class LaplacianScheduleBenchmarkResult:
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
    eval_history: list[tuple[int, float]]
    output_mean: float
    output_min: float
    output_max: float
    ir_summary: LaplacianIRScheduleSummary
    warmup_overhead_sec: float = 0.0
    used_warmup_cache: bool = False


@dataclass(frozen=True)
class LaplacianWarmupCandidateProfile:
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
class LaplacianWarmupSelection:
    chosen_schedule_name: str
    profiles: list[LaplacianWarmupCandidateProfile]
    total_warmup_overhead_sec: float
    from_cache: bool
    cache_signature: str | None


@dataclass(frozen=True)
class LaplacianSelectionQuality:
    oracle_schedule_name: str
    oracle_estimated_runtime_sec: float
    auto_selected_schedule_name: str
    auto_estimated_runtime_sec: float
    runtime_regret_sec: float
    runtime_regret_pct: float


@dataclass(frozen=True)
class LaplacianEvaluationReport:
    config: LaplacianBenchmarkConfig
    fixed_results: list[LaplacianScheduleBenchmarkResult]
    auto_result: LaplacianScheduleBenchmarkResult | None
    warmup_selection: LaplacianWarmupSelection | None
    selection_quality: LaplacianSelectionQuality | None


def available_laplacian_schedule_names() -> tuple[str, ...]:
    return LAPLACIAN_SCHEDULES


def normalize_laplacian_schedule_names(
    schedule_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if schedule_names is None:
        return available_laplacian_schedule_names()

    normalized = tuple(dict.fromkeys(schedule_names))
    unknown = sorted(set(normalized) - set(LAPLACIAN_SCHEDULES))
    if unknown:
        known = ", ".join(available_laplacian_schedule_names())
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown schedules: {unknown_names}. Known schedules: {known}")
    return normalized


def run_laplacian_evaluation_protocol(
    config: LaplacianBenchmarkConfig,
    *,
    baseline_schedules: Sequence[str] | None = None,
    include_auto: bool = False,
    auto_candidate_schedules: Sequence[str] | None = None,
    warmup_steps: int = 3,
    memory_budget_mb: float | None = None,
    error_guard_tolerance: float = 0.10,
    max_params_for_forward_like: int = 50_000,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_laplacian_warmup_cache.json",
) -> LaplacianEvaluationReport:
    _validate_config(config)
    _validate_warmup_args(warmup_steps, max_params_for_forward_like)
    selected_baselines = normalize_laplacian_schedule_names(baseline_schedules)
    params, step_points, sample_points = _initialize_laplacian_state(config)
    fixed_results = [
        _run_laplacian_schedule_with_state(
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

    auto_result: LaplacianScheduleBenchmarkResult | None = None
    warmup_selection: LaplacianWarmupSelection | None = None
    selection_quality: LaplacianSelectionQuality | None = None
    if include_auto:
        selected_candidates = normalize_laplacian_schedule_names(
            auto_candidate_schedules or selected_baselines
        )
        selected_schedule, warmup_selection = choose_laplacian_schedule_via_warmup(
            config,
            params=params,
            candidate_schedules=selected_candidates,
            warmup_steps=warmup_steps,
            memory_budget_mb=memory_budget_mb,
            error_guard_tolerance=error_guard_tolerance,
            max_params_for_forward_like=max_params_for_forward_like,
            use_cache=use_cache,
            cache_path=cache_path,
        )
        auto_result = _run_laplacian_schedule_with_state(
            config,
            report_schedule_name="auto",
            schedule_name=selected_schedule,
            params=params,
            step_points=step_points,
            sample_points=sample_points,
            warmup_overhead_sec=warmup_selection.total_warmup_overhead_sec,
            used_warmup_cache=warmup_selection.from_cache,
        )
        selection_quality = compute_laplacian_selection_quality(
            config,
            fixed_results=fixed_results,
            auto_result=auto_result,
        )

    return LaplacianEvaluationReport(
        config=config,
        fixed_results=fixed_results,
        auto_result=auto_result,
        warmup_selection=warmup_selection,
        selection_quality=selection_quality,
    )


def choose_laplacian_schedule_via_warmup(
    config: LaplacianBenchmarkConfig,
    *,
    params: MLPParams,
    candidate_schedules: Sequence[str],
    warmup_steps: int,
    memory_budget_mb: float | None,
    error_guard_tolerance: float = 0.10,
    max_params_for_forward_like: int = 50_000,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_laplacian_warmup_cache.json",
) -> tuple[str, LaplacianWarmupSelection]:
    normalized_candidates = normalize_laplacian_schedule_names(candidate_schedules)
    warmup_points = _make_laplacian_step_points(config, num_steps=warmup_steps, fold=17_241)
    param_count = _tree_num_parameters(params)
    cache_signature = _make_warmup_signature(
        config=config,
        candidate_schedules=normalized_candidates,
        warmup_steps=warmup_steps,
        memory_budget_mb=memory_budget_mb,
        max_params_for_forward_like=max_params_for_forward_like,
        param_count=param_count,
    )

    cached_choice = None
    if use_cache:
        cache = _load_warmup_cache(cache_path)
        cached_choice = cache.get(cache_signature)
        if cached_choice not in normalized_candidates:
            cached_choice = None

    if cached_choice is not None:
        return cached_choice, LaplacianWarmupSelection(
            chosen_schedule_name=cached_choice,
            profiles=[],
            total_warmup_overhead_sec=0.0,
            from_cache=True,
            cache_signature=cache_signature,
        )

    reference_start = time.perf_counter()
    reference_fn = jax.jit(_build_laplacian_schedule_fn(SCHEDULE_ROR))
    reference_outputs = [_block_tree(reference_fn(params, points)) for points in warmup_points]
    reference_overhead = time.perf_counter() - reference_start
    raw_profiles: list[dict[str, Any]] = []

    for schedule_name in normalized_candidates:
        rejected_reason: str | None = None
        if (
            max_params_for_forward_like > 0
            and schedule_name in {SCHEDULE_FOR, SCHEDULE_FOR_REMAT}
            and param_count > max_params_for_forward_like
        ):
            rejected_reason = (
                f"skipped_by_guard:param_count={param_count} exceeds "
                f"{max_params_for_forward_like}"
            )

        compile_overhead = 0.0
        total_step_time = 0.0
        median_step_time = float("inf")
        p90_step_time = float("inf")
        peak_host_memory_mb = _read_peak_host_memory_mb()
        peak_device_memory_mb = _read_peak_device_memory_mb()
        errors: list[float] = []
        stable = True

        if rejected_reason is None:
            try:
                schedule_fn = jax.jit(_build_laplacian_schedule_fn(schedule_name))
                compiled_fn, compile_overhead = _compile_jitted_schedule(
                    schedule_fn,
                    params,
                    warmup_points[0],
                )

                step_times: list[float] = []
                for points, reference in zip(warmup_points, reference_outputs):
                    start = time.perf_counter()
                    output = compiled_fn(params, points)
                    output = _block_tree(output)
                    step_times.append(time.perf_counter() - start)
                    errors.append(_max_abs_error(output, reference))
                    peak_host_memory_mb = max(peak_host_memory_mb, _read_peak_host_memory_mb())
                    current_device = _read_peak_device_memory_mb()
                    if current_device is not None:
                        peak_device_memory_mb = (
                            current_device
                            if peak_device_memory_mb is None
                            else max(peak_device_memory_mb, current_device)
                        )

                total_step_time = float(np.sum(step_times))
                median_step_time = float(np.median(step_times))
                p90_step_time = float(np.percentile(step_times, 90))
                stable = bool(np.all(np.isfinite(np.asarray(errors, dtype=np.float64))))
                if not stable:
                    rejected_reason = "unstable_output"
            except Exception as exc:
                stable = False
                rejected_reason = f"runtime_error:{type(exc).__name__}"

        mean_error = float(np.mean(errors)) if errors else float("nan")
        final_error = errors[-1] if errors else float("nan")
        score = compile_overhead + median_step_time * config.outer_steps

        if memory_budget_mb is not None:
            memory_observed = peak_device_memory_mb or peak_host_memory_mb
            if memory_observed > memory_budget_mb:
                rejected_reason = (
                    f"memory_budget_exceeded:{memory_observed:.2f}MB>{memory_budget_mb:.2f}MB"
                )

        if rejected_reason is not None:
            score = float("inf")

        raw_profiles.append(
            {
                "schedule_name": schedule_name,
                "compile_overhead_sec": compile_overhead,
                "total_step_time_sec": total_step_time,
                "median_step_time_sec": median_step_time,
                "p90_step_time_sec": p90_step_time,
                "peak_host_memory_mb": peak_host_memory_mb,
                "peak_device_memory_mb": peak_device_memory_mb,
                "mean_max_abs_error": mean_error,
                "final_max_abs_error": final_error,
                "stable": stable,
                "rejected_reason": rejected_reason,
                "score": score,
            }
        )

    _apply_error_guard(raw_profiles, tolerance=error_guard_tolerance)
    profiles = [LaplacianWarmupCandidateProfile(**profile) for profile in raw_profiles]
    chosen_schedule = _pick_best_laplacian_warmup_candidate(profiles)
    total_warmup_overhead = reference_overhead + float(
        sum(profile.compile_overhead_sec + profile.total_step_time_sec for profile in profiles)
    )
    selection = LaplacianWarmupSelection(
        chosen_schedule_name=chosen_schedule,
        profiles=profiles,
        total_warmup_overhead_sec=total_warmup_overhead,
        from_cache=False,
        cache_signature=cache_signature if use_cache else None,
    )
    if use_cache:
        cache = _load_warmup_cache(cache_path)
        cache[cache_signature] = chosen_schedule
        _save_warmup_cache(cache_path, cache)
    return chosen_schedule, selection


def compute_laplacian_selection_quality(
    config: LaplacianBenchmarkConfig,
    *,
    fixed_results: Sequence[LaplacianScheduleBenchmarkResult],
    auto_result: LaplacianScheduleBenchmarkResult,
) -> LaplacianSelectionQuality:
    if not fixed_results:
        raise ValueError("fixed_results must contain at least one baseline result")

    oracle = min(
        fixed_results,
        key=lambda result: _estimated_runtime_sec(result, config.outer_steps),
    )
    oracle_runtime = _estimated_runtime_sec(oracle, config.outer_steps)
    auto_runtime = _estimated_runtime_sec(
        auto_result,
        config.outer_steps,
        include_warmup=True,
    )
    regret_sec = auto_runtime - oracle_runtime
    regret_pct = 100.0 * regret_sec / oracle_runtime if oracle_runtime > 0 else 0.0
    return LaplacianSelectionQuality(
        oracle_schedule_name=oracle.schedule_name,
        oracle_estimated_runtime_sec=oracle_runtime,
        auto_selected_schedule_name=auto_result.selected_schedule_name,
        auto_estimated_runtime_sec=auto_runtime,
        runtime_regret_sec=regret_sec,
        runtime_regret_pct=regret_pct,
    )


def laplacian_evaluation_report_to_dict(
    report: LaplacianEvaluationReport,
) -> dict[str, Any]:
    return asdict(report)


def build_laplacian_schedule_fn(
    schedule_name: str,
) -> Callable[[MLPParams, Array], Array]:
    return _build_laplacian_schedule_fn(schedule_name)


def _run_laplacian_schedule_with_state(
    config: LaplacianBenchmarkConfig,
    *,
    report_schedule_name: str,
    schedule_name: str,
    params: MLPParams,
    step_points: Array,
    sample_points: Array,
    warmup_overhead_sec: float,
    used_warmup_cache: bool,
) -> LaplacianScheduleBenchmarkResult:
    ir_summary = _ir_summary_for_schedule(schedule_name, params, sample_points)
    schedule_fn = jax.jit(_build_laplacian_schedule_fn(schedule_name))
    reference_fn = jax.jit(_build_laplacian_schedule_fn(SCHEDULE_ROR))
    compiled_fn, compile_overhead_sec = _compile_jitted_schedule(
        schedule_fn,
        params,
        sample_points,
    )
    reference_output = _block_tree(reference_fn(params, sample_points))

    step_times: list[float] = []
    eval_history: list[tuple[int, float]] = []
    final_output = compiled_fn(params, sample_points)
    final_output = _block_tree(final_output)
    peak_host_memory_mb = _read_peak_host_memory_mb()
    peak_device_memory_mb = _read_peak_device_memory_mb()

    for step_idx, points in enumerate(step_points, start=1):
        start = time.perf_counter()
        output = compiled_fn(params, points)
        output = _block_tree(output)
        step_times.append(time.perf_counter() - start)
        final_output = output
        peak_host_memory_mb = max(peak_host_memory_mb, _read_peak_host_memory_mb())
        current_device = _read_peak_device_memory_mb()
        if current_device is not None:
            peak_device_memory_mb = (
                current_device
                if peak_device_memory_mb is None
                else max(peak_device_memory_mb, current_device)
            )

        if step_idx % config.eval_every == 0 or step_idx == config.outer_steps:
            ref = _block_tree(reference_fn(params, points))
            eval_history.append((step_idx, _max_abs_error(output, ref)))

    step_times_arr = np.asarray(step_times, dtype=np.float64)
    final_error = eval_history[-1][1]
    best_error = min(error for _, error in eval_history)
    iterations_to_target = next(
        (
            iteration
            for iteration, error in eval_history
            if error <= config.target_max_abs_error
        ),
        None,
    )
    final_arr = np.asarray(final_output)
    return LaplacianScheduleBenchmarkResult(
        schedule_name=report_schedule_name,
        selected_schedule_name=schedule_name,
        compile_overhead_sec=compile_overhead_sec,
        avg_step_time_sec=float(np.mean(step_times_arr)),
        p50_step_time_sec=float(np.percentile(step_times_arr, 50)),
        p90_step_time_sec=float(np.percentile(step_times_arr, 90)),
        peak_host_memory_mb=peak_host_memory_mb,
        peak_device_memory_mb=peak_device_memory_mb,
        final_max_abs_error=final_error,
        best_max_abs_error=best_error,
        iterations_to_target_error=iterations_to_target,
        eval_history=eval_history,
        output_mean=float(np.mean(final_arr)),
        output_min=float(np.min(final_arr)),
        output_max=float(np.max(final_arr)),
        ir_summary=ir_summary,
        warmup_overhead_sec=warmup_overhead_sec,
        used_warmup_cache=used_warmup_cache,
    )


def _build_laplacian_schedule_fn(
    schedule_name: str,
) -> Callable[[MLPParams, Array], Array]:
    normalize_laplacian_schedule_names((schedule_name,))
    remat = schedule_name.endswith("_remat")
    base_schedule = schedule_name.removesuffix("_remat")

    def scalar_field(params: MLPParams, x: Array) -> Array:
        return _mlp_scalar_remat(params, x) if remat else _mlp_scalar(params, x)

    def laplacian_at_point(params: MLPParams, x: Array) -> Array:
        scalar_fn = lambda z: scalar_field(params, z)

        if base_schedule == SCHEDULE_ROR:
            hessian = jax.jacrev(jax.grad(scalar_fn))(x)
            return jnp.trace(hessian)

        if base_schedule == SCHEDULE_FOR:
            grad_fn = jax.grad(scalar_fn)
            basis = jnp.eye(x.shape[0], dtype=x.dtype)

            def diagonal_entry(direction: Array) -> Array:
                _, hvp = jax.jvp(grad_fn, (x,), (direction,))
                return jnp.dot(direction, hvp)

            return jnp.sum(jax.vmap(diagonal_entry)(basis))

        if base_schedule == SCHEDULE_JACREV:
            hessian = jax.jacrev(jax.jacrev(scalar_fn))(x)
            return jnp.trace(hessian)

        if base_schedule == SCHEDULE_JACFWD:
            hessian = jax.jacfwd(jax.jacfwd(scalar_fn))(x)
            return jnp.trace(hessian)

        raise ValueError(f"Unknown Laplacian schedule: {schedule_name}")

    def scheduled_laplacian(params: MLPParams, points: Array) -> Array:
        return jax.vmap(lambda point: laplacian_at_point(params, point))(points)

    return scheduled_laplacian


def _mlp_scalar(params: MLPParams, x: Array) -> Array:
    activations = x
    for weights, bias in params[:-1]:
        activations = jnp.tanh(activations @ weights + bias)
    final_weights, final_bias = params[-1]
    return jnp.squeeze(activations @ final_weights + final_bias)


_mlp_scalar_remat = jax.checkpoint(_mlp_scalar)


def _initialize_laplacian_state(
    config: LaplacianBenchmarkConfig,
) -> tuple[MLPParams, Array, Array]:
    params = _init_mlp_params(config)
    step_points = _make_laplacian_step_points(config, num_steps=config.outer_steps, fold=0)
    sample_points = step_points[0]
    return params, step_points, sample_points


def _init_mlp_params(config: LaplacianBenchmarkConfig) -> MLPParams:
    key = jax.random.PRNGKey(config.seed)
    layer_dims = (
        [config.input_dim]
        + [config.hidden_dim] * config.hidden_layers
        + [1]
    )
    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    for layer_key, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:]):
        weights = jax.random.normal(layer_key, (in_dim, out_dim)) / jnp.sqrt(float(in_dim))
        bias = jnp.zeros((out_dim,), dtype=weights.dtype)
        layers.append((weights.astype(jnp.float32), bias))
    return tuple(layers)


def _make_laplacian_step_points(
    config: LaplacianBenchmarkConfig,
    *,
    num_steps: int,
    fold: int,
) -> Array:
    key = jax.random.fold_in(jax.random.PRNGKey(config.seed), fold)
    return jax.random.normal(
        key,
        (num_steps, config.num_points, config.input_dim),
        dtype=jnp.float32,
    )


def _compile_jitted_schedule(jitted_fn, params: MLPParams, points: Array):
    try:
        start = time.perf_counter()
        compiled = jitted_fn.lower(params, points).compile()
        compile_overhead = time.perf_counter() - start
        return compiled, compile_overhead
    except AttributeError:
        start = time.perf_counter()
        output = jitted_fn(params, points)
        _ = _block_tree(output)
        compile_overhead = time.perf_counter() - start
        return jitted_fn, compile_overhead


def _ir_summary_for_schedule(
    schedule_name: str,
    params: MLPParams,
    sample_points: Array,
) -> LaplacianIRScheduleSummary:
    schedule_fn = _build_laplacian_schedule_fn(schedule_name)
    closed_jaxpr = jax.make_jaxpr(schedule_fn)(params, sample_points)
    features = analyze_closed_jaxpr(closed_jaxpr, assume_outer_grad=True)
    return LaplacianIRScheduleSummary(
        total_equations=features.total_equations,
        max_loop_nesting=features.max_loop_nesting,
        num_higher_order_sites=len(features.higher_order_sites),
    )


def _apply_error_guard(raw_profiles: list[dict[str, Any]], *, tolerance: float) -> None:
    stable_errors = [
        profile["final_max_abs_error"]
        for profile in raw_profiles
        if profile["rejected_reason"] is None
        and np.isfinite(profile["final_max_abs_error"])
    ]
    if not stable_errors:
        return

    best_error = min(stable_errors)
    max_allowed = max(1e-5, best_error * (1.0 + max(tolerance, 0.0)))
    for profile in raw_profiles:
        if profile["rejected_reason"] is not None:
            continue
        final_error = profile["final_max_abs_error"]
        if not np.isfinite(final_error):
            profile["rejected_reason"] = "invalid_final_error"
            profile["score"] = float("inf")
            continue
        if final_error > max_allowed:
            profile["rejected_reason"] = (
                f"error_guard:final_error={final_error:.6e}>{max_allowed:.6e}"
            )
            profile["score"] = float("inf")


def _pick_best_laplacian_warmup_candidate(
    profiles: Sequence[LaplacianWarmupCandidateProfile],
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
    if not profiles:
        raise ValueError("No warmup profiles provided")
    raise ValueError("All warmup candidates were rejected")


def _estimated_runtime_sec(
    result: LaplacianScheduleBenchmarkResult,
    outer_steps: int,
    *,
    include_warmup: bool = False,
) -> float:
    base = result.compile_overhead_sec + result.avg_step_time_sec * outer_steps
    if include_warmup:
        base += result.warmup_overhead_sec
    return base


def _max_abs_error(output: Array, reference: Array) -> float:
    return float(jnp.max(jnp.abs(output - reference)).block_until_ready())


def _block_tree(tree):
    return jtu.tree_map(lambda x: x.block_until_ready(), tree)


def _tree_num_parameters(tree: MLPParams) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jtu.tree_leaves(tree)))


def _read_peak_host_memory_mb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak_rss) / (1024 * 1024)
    return float(peak_rss) / 1024


def _read_peak_device_memory_mb() -> float | None:
    peaks: list[float] = []
    for device in jax.devices():
        memory_stats_fn = getattr(device, "memory_stats", None)
        if not callable(memory_stats_fn):
            continue
        try:
            stats = memory_stats_fn()
        except RuntimeError:
            continue
        if not isinstance(stats, dict):
            continue
        for key in (
            "peak_bytes_in_use",
            "peak_bytes_reserved",
            "bytes_in_use",
            "bytes_reserved",
        ):
            value = stats.get(key)
            if isinstance(value, (int, float)):
                peaks.append(float(value))
                break
    if not peaks:
        return None
    return max(peaks) / (1024 * 1024)


def _make_warmup_signature(
    *,
    config: LaplacianBenchmarkConfig,
    candidate_schedules: Sequence[str],
    warmup_steps: int,
    memory_budget_mb: float | None,
    max_params_for_forward_like: int,
    param_count: int,
) -> str:
    payload = {
        "seed": config.seed,
        "outer_steps": config.outer_steps,
        "eval_every": config.eval_every,
        "num_points": config.num_points,
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "hidden_layers": config.hidden_layers,
        "target_max_abs_error": config.target_max_abs_error,
        "candidate_schedules": list(candidate_schedules),
        "warmup_steps": warmup_steps,
        "memory_budget_mb": memory_budget_mb,
        "max_params_for_forward_like": max_params_for_forward_like,
        "param_count": param_count,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _load_warmup_cache(path: str) -> dict[str, str]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}
    entries = payload.get("entries", {})
    if isinstance(entries, dict):
        return {str(k): str(v) for k, v in entries.items()}
    return {}


def _save_warmup_cache(path: str, cache: dict[str, str]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "entries": cache}
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _validate_config(config: LaplacianBenchmarkConfig) -> None:
    if config.outer_steps < 1:
        raise ValueError("outer_steps must be >= 1")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.num_points < 1:
        raise ValueError("num_points must be >= 1")
    if config.input_dim < 1:
        raise ValueError("input_dim must be >= 1")
    if config.hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if config.hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    if config.target_max_abs_error < 0:
        raise ValueError("target_max_abs_error must be >= 0")


def _validate_warmup_args(
    warmup_steps: int,
    max_params_for_forward_like: int,
) -> None:
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    if max_params_for_forward_like < 0:
        raise ValueError("max_params_for_forward_like must be >= 0")
