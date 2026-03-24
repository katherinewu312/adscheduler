from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import resource
import sys
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from adscheduler.ir_analysis import analyze_closed_jaxpr
from adscheduler.maml import (
    ExampleConfig,
    TaskBatches,
    adapt_params,
    available_schedule_names,
    get_meta_grad_schedule,
    init_mlp_params,
    make_synthetic_task_batches,
    mlp,
    normalize_schedule_names,
    trace_maml_programs,
)

Array = jax.Array
TaskBatchArrays = tuple[Array, Array, Array, Array]


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 0
    outer_steps: int = 100
    eval_every: int = 10
    meta_batch_size: int = 4
    meta_test_tasks: int = 64
    target_meta_test_accuracy: float = 0.85
    in_dim: int = 8
    hidden_dim: int = 32
    out_dim: int = 1
    n_support: int = 16
    n_query: int = 16
    inner_steps: int = 3
    inner_lr: float = 0.1
    outer_lr: float = 0.01

    def to_example_config(self) -> ExampleConfig:
        return ExampleConfig(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            n_support=self.n_support,
            n_query=self.n_query,
            inner_steps=self.inner_steps,
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
        )


@dataclass(frozen=True)
class IRScheduleSummary:
    total_equations: int
    max_loop_nesting: int
    num_higher_order_sites: int


@dataclass(frozen=True)
class ScheduleBenchmarkResult:
    schedule_name: str
    selected_schedule_name: str
    compile_overhead_sec: float
    avg_outer_step_time_sec: float
    p50_outer_step_time_sec: float
    p90_outer_step_time_sec: float
    peak_host_memory_mb: float
    peak_device_memory_mb: float | None
    final_meta_train_loss: float
    final_meta_test_accuracy: float
    best_meta_test_accuracy: float
    outer_iterations_to_target: int | None
    eval_history: list[tuple[int, float]]
    ir_summary: IRScheduleSummary
    warmup_overhead_sec: float = 0.0
    used_warmup_cache: bool = False


@dataclass(frozen=True)
class BenchmarkReport:
    config: BenchmarkConfig
    schedule_results: list[ScheduleBenchmarkResult]


@dataclass(frozen=True)
class WarmupCandidateProfile:
    schedule_name: str
    compile_overhead_sec: float
    total_step_time_sec: float
    median_step_time_sec: float
    p90_step_time_sec: float
    peak_host_memory_mb: float
    peak_device_memory_mb: float | None
    mean_meta_loss: float
    final_meta_loss: float
    stable: bool
    rejected_reason: str | None
    score: float


@dataclass(frozen=True)
class WarmupSelection:
    chosen_schedule_name: str
    profiles: list[WarmupCandidateProfile]
    total_warmup_overhead_sec: float
    from_cache: bool
    cache_signature: str | None


@dataclass(frozen=True)
class SelectionQuality:
    oracle_schedule_name: str
    oracle_estimated_runtime_sec: float
    auto_selected_schedule_name: str
    auto_estimated_runtime_sec: float
    runtime_regret_sec: float
    runtime_regret_pct: float


@dataclass(frozen=True)
class EvaluationReport:
    config: BenchmarkConfig
    fixed_results: list[ScheduleBenchmarkResult]
    auto_result: ScheduleBenchmarkResult | None
    warmup_selection: WarmupSelection | None
    selection_quality: SelectionQuality | None


def run_benchmark_suite(
    config: BenchmarkConfig,
    *,
    schedule_names: Sequence[str] | None = None,
) -> BenchmarkReport:
    selected = normalize_schedule_names(schedule_names)
    results = [run_schedule_benchmark(config, schedule_name=s) for s in selected]
    return BenchmarkReport(config=config, schedule_results=results)


def run_schedule_benchmark(
    config: BenchmarkConfig,
    *,
    schedule_name: str,
) -> ScheduleBenchmarkResult:
    _validate_benchmark_config(config)
    normalize_schedule_names((schedule_name,))

    (
        params0,
        train_keys,
        test_tasks,
        sample_train_tasks,
    ) = _initialize_benchmark_state(config)
    return _run_schedule_with_state(
        config,
        report_schedule_name=schedule_name,
        schedule_name=schedule_name,
        params0=params0,
        train_keys=train_keys,
        test_tasks=test_tasks,
        sample_train_tasks=sample_train_tasks,
        warmup_overhead_sec=0.0,
        used_warmup_cache=False,
    )


def run_auto_schedule_benchmark(
    config: BenchmarkConfig,
    *,
    candidate_schedules: Sequence[str] | None = None,
    warmup_steps: int = 3,
    memory_budget_mb: float | None = None,
    loss_guard_tolerance: float = 0.10,
    max_params_for_forward_like: int = 50_000,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_warmup_cache.json",
) -> tuple[ScheduleBenchmarkResult, WarmupSelection]:
    _validate_benchmark_config(config)
    _validate_warmup_args(warmup_steps, max_params_for_forward_like)

    normalized_candidates = normalize_schedule_names(
        candidate_schedules or available_schedule_names()
    )
    (
        params0,
        train_keys,
        test_tasks,
        sample_train_tasks,
    ) = _initialize_benchmark_state(config)
    warmup_tasks = _make_warmup_tasks(config, warmup_steps)

    cache_signature = _make_warmup_signature(
        config=config,
        candidate_schedules=normalized_candidates,
        warmup_steps=warmup_steps,
        memory_budget_mb=memory_budget_mb,
        max_params_for_forward_like=max_params_for_forward_like,
        param_count=_tree_num_parameters(params0),
    )

    cached_choice = None
    if use_cache:
        cache = _load_warmup_cache(cache_path)
        cached_choice = cache.get(cache_signature)
        if cached_choice not in normalized_candidates:
            cached_choice = None

    if cached_choice is None:
        selected_schedule, profiles = choose_schedule_via_warmup(
            params0,
            warmup_tasks,
            normalized_candidates,
            remaining_outer_steps=config.outer_steps,
            memory_budget_mb=memory_budget_mb,
            inner_lr=config.inner_lr,
            outer_lr=config.outer_lr,
            inner_steps=config.inner_steps,
            loss_guard_tolerance=loss_guard_tolerance,
            max_params_for_forward_like=max_params_for_forward_like,
        )
        total_warmup_overhead_sec = _warmup_overhead_from_profiles(profiles)
        warmup_selection = WarmupSelection(
            chosen_schedule_name=selected_schedule,
            profiles=profiles,
            total_warmup_overhead_sec=total_warmup_overhead_sec,
            from_cache=False,
            cache_signature=cache_signature if use_cache else None,
        )
        if use_cache:
            cache = _load_warmup_cache(cache_path)
            cache[cache_signature] = selected_schedule
            _save_warmup_cache(cache_path, cache)
    else:
        selected_schedule = cached_choice
        warmup_selection = WarmupSelection(
            chosen_schedule_name=selected_schedule,
            profiles=[],
            total_warmup_overhead_sec=0.0,
            from_cache=True,
            cache_signature=cache_signature,
        )

    auto_result = _run_schedule_with_state(
        config,
        report_schedule_name="auto",
        schedule_name=selected_schedule,
        params0=params0,
        train_keys=train_keys,
        test_tasks=test_tasks,
        sample_train_tasks=sample_train_tasks,
        warmup_overhead_sec=warmup_selection.total_warmup_overhead_sec,
        used_warmup_cache=warmup_selection.from_cache,
    )
    return auto_result, warmup_selection


def run_evaluation_protocol(
    config: BenchmarkConfig,
    *,
    baseline_schedules: Sequence[str] | None = None,
    include_auto: bool = False,
    auto_candidate_schedules: Sequence[str] | None = None,
    warmup_steps: int = 3,
    memory_budget_mb: float | None = None,
    loss_guard_tolerance: float = 0.10,
    max_params_for_forward_like: int = 50_000,
    use_cache: bool = True,
    cache_path: str = ".adscheduler_warmup_cache.json",
) -> EvaluationReport:
    selected_baselines = normalize_schedule_names(
        baseline_schedules or available_schedule_names()
    )
    fixed_results = [
        run_schedule_benchmark(config, schedule_name=schedule_name)
        for schedule_name in selected_baselines
    ]

    auto_result: ScheduleBenchmarkResult | None = None
    warmup_selection: WarmupSelection | None = None
    selection_quality: SelectionQuality | None = None
    if include_auto:
        auto_result, warmup_selection = run_auto_schedule_benchmark(
            config,
            candidate_schedules=auto_candidate_schedules or selected_baselines,
            warmup_steps=warmup_steps,
            memory_budget_mb=memory_budget_mb,
            loss_guard_tolerance=loss_guard_tolerance,
            max_params_for_forward_like=max_params_for_forward_like,
            use_cache=use_cache,
            cache_path=cache_path,
        )
        selection_quality = compute_selection_quality(
            config,
            fixed_results=fixed_results,
            auto_result=auto_result,
        )

    return EvaluationReport(
        config=config,
        fixed_results=fixed_results,
        auto_result=auto_result,
        warmup_selection=warmup_selection,
        selection_quality=selection_quality,
    )


def compute_selection_quality(
    config: BenchmarkConfig,
    *,
    fixed_results: Sequence[ScheduleBenchmarkResult],
    auto_result: ScheduleBenchmarkResult,
) -> SelectionQuality:
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

    return SelectionQuality(
        oracle_schedule_name=oracle.schedule_name,
        oracle_estimated_runtime_sec=oracle_runtime,
        auto_selected_schedule_name=auto_result.selected_schedule_name,
        auto_estimated_runtime_sec=auto_runtime,
        runtime_regret_sec=regret_sec,
        runtime_regret_pct=regret_pct,
    )


def choose_schedule_via_warmup(
    params0: dict[str, Array],
    warmup_tasks: Sequence[TaskBatchArrays],
    candidates: Sequence[str],
    remaining_outer_steps: int,
    memory_budget_mb: float | None,
    *,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
    loss_guard_tolerance: float = 0.10,
    max_params_for_forward_like: int = 50_000,
) -> tuple[str, list[WarmupCandidateProfile]]:
    normalized_candidates = normalize_schedule_names(candidates)
    if not warmup_tasks:
        raise ValueError("warmup_tasks must contain at least one warmup step")
    if remaining_outer_steps < 1:
        raise ValueError("remaining_outer_steps must be >= 1")

    param_count = _tree_num_parameters(params0)
    raw_profiles: list[dict[str, Any]] = []

    for schedule_name in normalized_candidates:
        rejected_reason: str | None = None
        if (
            max_params_for_forward_like > 0
            and schedule_name in {"for", "for_remat"}
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
        losses: list[float] = []
        stable = True

        if rejected_reason is None:
            try:
                train_step_fn = _build_meta_train_step_fn(
                    schedule_name=schedule_name,
                    inner_lr=inner_lr,
                    outer_lr=outer_lr,
                    inner_steps=inner_steps,
                )
                jitted_train_step = jax.jit(train_step_fn)
                compiled_train_step, compile_overhead = _compile_jitted_step(
                    jitted_train_step,
                    params0,
                    warmup_tasks[0],
                )

                params = params0
                step_times: list[float] = []
                for warmup_task in warmup_tasks:
                    step_start = time.perf_counter()
                    params, meta_loss = compiled_train_step(params, *warmup_task)
                    params = _block_tree(params)
                    meta_loss = meta_loss.block_until_ready()
                    step_times.append(time.perf_counter() - step_start)
                    losses.append(float(meta_loss))
                    peak_host_memory_mb = max(
                        peak_host_memory_mb,
                        _read_peak_host_memory_mb(),
                    )
                    current_device = _read_peak_device_memory_mb()
                    if current_device is not None:
                        if peak_device_memory_mb is None:
                            peak_device_memory_mb = current_device
                        else:
                            peak_device_memory_mb = max(
                                peak_device_memory_mb,
                                current_device,
                            )

                total_step_time = float(np.sum(step_times))
                median_step_time = float(np.median(step_times))
                p90_step_time = float(np.percentile(step_times, 90))
                stable = bool(np.all(np.isfinite(np.asarray(losses, dtype=np.float64))))
                if not stable:
                    rejected_reason = "unstable_loss"
            except Exception as exc:
                stable = False
                rejected_reason = f"runtime_error:{type(exc).__name__}"

        mean_meta_loss = float(np.mean(losses)) if losses else float("nan")
        final_meta_loss = losses[-1] if losses else float("nan")
        score = compile_overhead + median_step_time * remaining_outer_steps

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
                "mean_meta_loss": mean_meta_loss,
                "final_meta_loss": final_meta_loss,
                "stable": stable,
                "rejected_reason": rejected_reason,
                "score": score,
            }
        )

    _apply_loss_guard(raw_profiles, tolerance=loss_guard_tolerance)
    profiles = [WarmupCandidateProfile(**profile) for profile in raw_profiles]
    chosen_schedule = _pick_best_warmup_candidate(profiles)
    return chosen_schedule, profiles


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)


def evaluation_report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return asdict(report)


def _initialize_benchmark_state(
    config: BenchmarkConfig,
) -> tuple[dict[str, Array], Array, TaskBatchArrays, TaskBatchArrays]:
    example_config = config.to_example_config()
    key = jax.random.PRNGKey(config.seed)
    key_params, key_train, key_test = jax.random.split(key, 3)

    params0 = init_mlp_params(
        key_params,
        example_config.in_dim,
        example_config.hidden_dim,
        example_config.out_dim,
    )
    train_keys = jax.random.split(key_train, config.outer_steps)
    test_tasks = _make_task_batch_arrays(
        key_test,
        num_tasks=config.meta_test_tasks,
        config=example_config,
    )
    sample_train_tasks = _make_task_batch_arrays(
        train_keys[0],
        num_tasks=config.meta_batch_size,
        config=example_config,
    )
    return params0, train_keys, test_tasks, sample_train_tasks


def _run_schedule_with_state(
    config: BenchmarkConfig,
    *,
    report_schedule_name: str,
    schedule_name: str,
    params0: dict[str, Array],
    train_keys: Array,
    test_tasks: TaskBatchArrays,
    sample_train_tasks: TaskBatchArrays,
    warmup_overhead_sec: float,
    used_warmup_cache: bool,
) -> ScheduleBenchmarkResult:
    ir_summary = _ir_summary_for_schedule(
        schedule_name,
        params0,
        sample_train_tasks,
        config,
    )

    train_step_fn = _build_meta_train_step_fn(
        schedule_name=schedule_name,
        inner_lr=config.inner_lr,
        outer_lr=config.outer_lr,
        inner_steps=config.inner_steps,
    )
    eval_accuracy_fn = _build_meta_test_accuracy_fn(
        inner_lr=config.inner_lr,
        inner_steps=config.inner_steps,
    )

    jitted_train_step = jax.jit(train_step_fn)
    jitted_eval_accuracy = jax.jit(eval_accuracy_fn)
    compiled_train_step, compile_overhead_sec = _compile_jitted_step(
        jitted_train_step,
        params0,
        sample_train_tasks,
    )

    _ = jitted_eval_accuracy(params0, *test_tasks).block_until_ready()

    params = params0
    step_times: list[float] = []
    eval_history: list[tuple[int, float]] = []
    final_meta_train_loss = float("nan")

    peak_host_memory_mb = _read_peak_host_memory_mb()
    peak_device_memory_mb = _read_peak_device_memory_mb()

    for step_idx, step_key in enumerate(train_keys, start=1):
        train_tasks = _make_task_batch_arrays(
            step_key,
            num_tasks=config.meta_batch_size,
            config=config.to_example_config(),
        )

        step_start = time.perf_counter()
        params, meta_loss = compiled_train_step(params, *train_tasks)
        params = _block_tree(params)
        meta_loss = meta_loss.block_until_ready()
        step_times.append(time.perf_counter() - step_start)
        final_meta_train_loss = float(meta_loss)

        peak_host_memory_mb = max(peak_host_memory_mb, _read_peak_host_memory_mb())
        current_device = _read_peak_device_memory_mb()
        if current_device is not None:
            if peak_device_memory_mb is None:
                peak_device_memory_mb = current_device
            else:
                peak_device_memory_mb = max(peak_device_memory_mb, current_device)

        if step_idx % config.eval_every == 0 or step_idx == config.outer_steps:
            accuracy = jitted_eval_accuracy(params, *test_tasks).block_until_ready()
            eval_history.append((step_idx, float(accuracy)))

    step_times_arr = np.asarray(step_times, dtype=np.float64)
    avg_outer_step_time_sec = float(np.mean(step_times_arr))
    p50_outer_step_time_sec = float(np.percentile(step_times_arr, 50))
    p90_outer_step_time_sec = float(np.percentile(step_times_arr, 90))

    final_meta_test_accuracy = eval_history[-1][1]
    best_meta_test_accuracy = max(acc for _, acc in eval_history)
    outer_iterations_to_target = next(
        (
            iteration
            for iteration, accuracy in eval_history
            if accuracy >= config.target_meta_test_accuracy
        ),
        None,
    )

    return ScheduleBenchmarkResult(
        schedule_name=report_schedule_name,
        selected_schedule_name=schedule_name,
        compile_overhead_sec=compile_overhead_sec,
        avg_outer_step_time_sec=avg_outer_step_time_sec,
        p50_outer_step_time_sec=p50_outer_step_time_sec,
        p90_outer_step_time_sec=p90_outer_step_time_sec,
        peak_host_memory_mb=peak_host_memory_mb,
        peak_device_memory_mb=peak_device_memory_mb,
        final_meta_train_loss=final_meta_train_loss,
        final_meta_test_accuracy=final_meta_test_accuracy,
        best_meta_test_accuracy=best_meta_test_accuracy,
        outer_iterations_to_target=outer_iterations_to_target,
        eval_history=eval_history,
        ir_summary=ir_summary,
        warmup_overhead_sec=warmup_overhead_sec,
        used_warmup_cache=used_warmup_cache,
    )


def _make_warmup_tasks(
    config: BenchmarkConfig,
    warmup_steps: int,
) -> list[TaskBatchArrays]:
    warmup_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 17_241)
    warmup_keys = jax.random.split(warmup_key, warmup_steps)
    example_config = config.to_example_config()
    return [
        _make_task_batch_arrays(
            key,
            num_tasks=config.meta_batch_size,
            config=example_config,
        )
        for key in warmup_keys
    ]


def _apply_loss_guard(raw_profiles: list[dict[str, Any]], *, tolerance: float) -> None:
    stable_losses = [
        profile["final_meta_loss"]
        for profile in raw_profiles
        if profile["rejected_reason"] is None and np.isfinite(profile["final_meta_loss"])
    ]
    if not stable_losses:
        return

    best_loss = min(stable_losses)
    max_allowed = best_loss * (1.0 + max(tolerance, 0.0))
    for profile in raw_profiles:
        if profile["rejected_reason"] is not None:
            continue
        if not np.isfinite(profile["final_meta_loss"]):
            profile["rejected_reason"] = "invalid_final_loss"
            profile["score"] = float("inf")
            continue
        if profile["final_meta_loss"] > max_allowed:
            profile["rejected_reason"] = (
                f"loss_guard:final_loss={profile['final_meta_loss']:.6f}>{max_allowed:.6f}"
            )
            profile["score"] = float("inf")


def _pick_best_warmup_candidate(profiles: Sequence[WarmupCandidateProfile]) -> str:
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


def _warmup_overhead_from_profiles(profiles: Sequence[WarmupCandidateProfile]) -> float:
    return float(
        sum(profile.compile_overhead_sec + profile.total_step_time_sec for profile in profiles)
    )


def _estimated_runtime_sec(
    result: ScheduleBenchmarkResult,
    outer_steps: int,
    *,
    include_warmup: bool = False,
) -> float:
    base = result.compile_overhead_sec + result.avg_outer_step_time_sec * outer_steps
    if include_warmup:
        base += result.warmup_overhead_sec
    return base


def _tree_num_parameters(tree: dict[str, Array]) -> int:
    leaves = jtu.tree_leaves(tree)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def _make_warmup_signature(
    *,
    config: BenchmarkConfig,
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
        "meta_batch_size": config.meta_batch_size,
        "meta_test_tasks": config.meta_test_tasks,
        "target_meta_test_accuracy": config.target_meta_test_accuracy,
        "in_dim": config.in_dim,
        "hidden_dim": config.hidden_dim,
        "out_dim": config.out_dim,
        "n_support": config.n_support,
        "n_query": config.n_query,
        "inner_steps": config.inner_steps,
        "inner_lr": config.inner_lr,
        "outer_lr": config.outer_lr,
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


def _validate_warmup_args(
    warmup_steps: int,
    max_params_for_forward_like: int,
) -> None:
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    if max_params_for_forward_like < 0:
        raise ValueError("max_params_for_forward_like must be >= 0")


def _build_meta_train_step_fn(
    *,
    schedule_name: str,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
):
    schedule_fn = get_meta_grad_schedule(schedule_name)

    def train_step(
        params: dict[str, Array],
        support_x: Array,
        support_y: Array,
        query_x: Array,
        query_y: Array,
    ) -> tuple[dict[str, Array], Array]:
        def single_task(task_support_x: Array, task_support_y: Array, task_query_x: Array, task_query_y: Array):
            batches = TaskBatches(
                support=(task_support_x, task_support_y),
                query=(task_query_x, task_query_y),
            )
            return schedule_fn(params, batches, inner_lr, inner_steps)

        losses, task_grads = jax.vmap(single_task)(support_x, support_y, query_x, query_y)
        mean_loss = jnp.mean(losses)
        mean_grads = jtu.tree_map(lambda x: jnp.mean(x, axis=0), task_grads)
        updated_params = jtu.tree_map(lambda p, g: p - outer_lr * g, params, mean_grads)
        return updated_params, mean_loss

    return train_step


def _build_meta_test_accuracy_fn(
    *,
    inner_lr: float,
    inner_steps: int,
):
    def meta_test_accuracy(
        params: dict[str, Array],
        support_x: Array,
        support_y: Array,
        query_x: Array,
        query_y: Array,
    ) -> Array:
        def single_task(task_support_x: Array, task_support_y: Array, task_query_x: Array, task_query_y: Array) -> Array:
            adapted = adapt_params(
                params,
                (task_support_x, task_support_y),
                inner_lr,
                inner_steps,
                stop_higher_order=False,
                remat_inner=False,
            )
            predictions = mlp(adapted, task_query_x)
            return _binary_accuracy(predictions, task_query_y)

        task_accuracies = jax.vmap(single_task)(support_x, support_y, query_x, query_y)
        return jnp.mean(task_accuracies)

    return meta_test_accuracy


def _binary_accuracy(predictions: Array, targets: Array) -> Array:
    pred_labels = predictions >= 0
    target_labels = targets >= 0
    return jnp.mean(pred_labels == target_labels)


def _make_task_batch_arrays(
    key: Array,
    *,
    num_tasks: int,
    config: ExampleConfig,
) -> TaskBatchArrays:
    task_keys = jax.random.split(key, num_tasks)

    def sample_task(task_key: Array) -> TaskBatchArrays:
        task = make_synthetic_task_batches(
            task_key,
            in_dim=config.in_dim,
            out_dim=config.out_dim,
            n_support=config.n_support,
            n_query=config.n_query,
        )
        support_x, support_y = task.support
        query_x, query_y = task.query
        return support_x, support_y, query_x, query_y

    return jax.vmap(sample_task)(task_keys)


def _compile_jitted_step(jitted_step, params: dict[str, Array], sample_tasks: TaskBatchArrays):
    support_x, support_y, query_x, query_y = sample_tasks

    try:
        start = time.perf_counter()
        compiled = jitted_step.lower(params, support_x, support_y, query_x, query_y).compile()
        compile_overhead = time.perf_counter() - start
        return compiled, compile_overhead
    except AttributeError:
        start = time.perf_counter()
        params, loss = jitted_step(params, support_x, support_y, query_x, query_y)
        _ = _block_tree(params)
        _ = loss.block_until_ready()
        compile_overhead = time.perf_counter() - start
        return jitted_step, compile_overhead


def _ir_summary_for_schedule(
    schedule_name: str,
    params: dict[str, Array],
    sample_train_tasks: TaskBatchArrays,
    config: BenchmarkConfig,
) -> IRScheduleSummary:
    support_x, support_y, query_x, query_y = sample_train_tasks
    sample_task = TaskBatches(
        support=(support_x[0], support_y[0]),
        query=(query_x[0], query_y[0]),
    )

    traced = trace_maml_programs(
        params,
        sample_task,
        inner_lr=config.inner_lr,
        outer_lr=config.outer_lr,
        inner_steps=config.inner_steps,
        schedule_names=(schedule_name,),
    )

    features = analyze_closed_jaxpr(
        traced.schedule_meta_grads[schedule_name],
        known_inner_steps=config.inner_steps,
        assume_outer_grad=True,
    )
    return IRScheduleSummary(
        total_equations=features.total_equations,
        max_loop_nesting=features.max_loop_nesting,
        num_higher_order_sites=len(features.higher_order_sites),
    )


def _block_tree(tree):
    return jtu.tree_map(lambda x: x.block_until_ready(), tree)


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


def _validate_benchmark_config(config: BenchmarkConfig) -> None:
    if config.outer_steps < 1:
        raise ValueError("outer_steps must be >= 1")
    if config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.meta_batch_size < 1:
        raise ValueError("meta_batch_size must be >= 1")
    if config.meta_test_tasks < 1:
        raise ValueError("meta_test_tasks must be >= 1")
