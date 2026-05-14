#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import jet

from adscheduler.stablehlo_execution import (
    StableHLOExecutionUnavailable,
    compile_stablehlo_with_xla,
)
from adscheduler.stablehlo_passes import (
    StableHLOProgram,
    lower_to_stablehlo,
    run_stablehlo_transform_pipeline,
)

Array = jax.Array
BenchmarkName = Literal["mlp_laplacian", "poisson_pinn"]
StrategyName = Literal["hessian", "jvp_grad", "jet"]


@dataclass(frozen=True)
class RuntimeStats:
    compile_overhead_ms: float | None
    avg_runtime_ms: float | None
    p50_runtime_ms: float | None
    p90_runtime_ms: float | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class SweepResult:
    benchmark: str
    hidden_width: int
    method: str
    before: RuntimeStats
    after: RuntimeStats
    speedup: float | None


@dataclass(frozen=True)
class Workload:
    name: str
    task: Callable[..., Any]
    args: tuple[Any, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ICML paper hidden-width sweep for fixed AD methods.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--width",
        action="append",
        type=int,
        dest="widths",
        default=None,
        help="Hidden width to benchmark. Repeat for multiple. Defaults to 128,256,384,512.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("mlp_laplacian", "poisson_pinn"),
        dest="benchmarks",
        default=None,
        help="Benchmark to run. Repeat for multiple. Defaults to both.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=("hessian", "jvp_grad", "jet"),
        dest="methods",
        default=None,
        help="AD method to run. Repeat for multiple. Defaults to all three.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "icml2025" / "paper_width_sweep_results.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    widths = tuple(args.widths or (128, 256, 384, 512))
    benchmarks = tuple(args.benchmarks or ("mlp_laplacian", "poisson_pinn"))
    methods = tuple(args.methods or ("hessian", "jvp_grad", "jet"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for benchmark in benchmarks:
            for width in widths:
                for method in methods:
                    result = run_one(
                        benchmark=benchmark,
                        hidden_width=width,
                        method=method,
                        seed=args.seed,
                        warmup_runs=args.warmup_runs,
                        runs=args.runs,
                    )
                    output.write(json.dumps(asdict(result)) + "\n")
                    output.flush()
                    print(json.dumps(asdict(result), indent=2), flush=True)


def run_one(
    *,
    benchmark: BenchmarkName,
    hidden_width: int,
    method: StrategyName,
    seed: int,
    warmup_runs: int,
    runs: int,
) -> SweepResult:
    workload = make_workload(
        benchmark=benchmark,
        hidden_width=hidden_width,
        method=method,
        seed=seed,
    )
    program = lower_to_stablehlo(workload.name, workload.task, workload.args)
    before = benchmark_program(program, workload.args, warmup_runs=warmup_runs, runs=runs)

    transform_result = run_stablehlo_transform_pipeline(program, source_args=workload.args)
    after = benchmark_program(
        transform_result.transformed_program,
        workload.args,
        warmup_runs=warmup_runs,
        runs=runs,
    )
    speedup = None
    if before.avg_runtime_ms is not None and after.avg_runtime_ms is not None:
        speedup = before.avg_runtime_ms / after.avg_runtime_ms
    return SweepResult(
        benchmark=benchmark,
        hidden_width=hidden_width,
        method=method,
        before=before,
        after=after,
        speedup=speedup,
    )


def benchmark_program(
    program: StableHLOProgram,
    args: tuple[Any, ...],
    *,
    warmup_runs: int,
    runs: int,
) -> RuntimeStats:
    try:
        executable = compile_stablehlo_with_xla(program)
    except StableHLOExecutionUnavailable as exc:
        return RuntimeStats(None, None, None, None, str(exc))

    try:
        if executable.prepare_args is not None and executable.call_prepared is not None:
            prepared_args = executable.prepare_args(*args)

            def run_once():
                return executable.call_prepared(prepared_args)
        else:

            def run_once():
                return executable.callable(*args)

        output = run_once()
        block_tree(output)
        for _ in range(warmup_runs):
            output = run_once()
            block_tree(output)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            output = run_once()
            block_tree(output)
            timings.append(time.perf_counter() - start)
    except Exception as exc:
        return RuntimeStats(None, None, None, None, f"execution failed: {exc}")

    timings_ms = np.asarray(timings, dtype=np.float64) * 1e3
    return RuntimeStats(
        compile_overhead_ms=executable.compile_overhead_sec * 1e3,
        avg_runtime_ms=float(np.mean(timings_ms)),
        p50_runtime_ms=float(np.percentile(timings_ms, 50)),
        p90_runtime_ms=float(np.percentile(timings_ms, 90)),
    )


def make_workload(
    *,
    benchmark: BenchmarkName,
    hidden_width: int,
    method: StrategyName,
    seed: int,
) -> Workload:
    if benchmark == "mlp_laplacian":
        return make_mlp_laplacian_workload(hidden_width=hidden_width, method=method, seed=seed)
    return make_poisson_pinn_workload(hidden_width=hidden_width, method=method, seed=seed)


def make_mlp_laplacian_workload(
    *,
    hidden_width: int,
    method: StrategyName,
    seed: int,
) -> Workload:
    input_dim = 3
    hidden_layers = 128
    num_points = 256
    params = init_mlp_params(
        seed=seed,
        layer_dims=tuple([input_dim] + [hidden_width] * hidden_layers + [1]),
    )
    points = jnp.linspace(
        -1.0,
        1.0,
        num_points * input_dim,
        dtype=jnp.float32,
    ).reshape(num_points, input_dim)

    def mlp_scalar(mlp_params, x):
        activations = x
        for weights, bias in mlp_params[:-1]:
            activations = jnp.tanh(activations @ weights + bias)
        final_weights, final_bias = mlp_params[-1]
        return jnp.squeeze(activations @ final_weights + final_bias)

    def laplacian_at_point(mlp_params, x):
        scalar_fn = lambda z: mlp_scalar(mlp_params, z)
        basis = jnp.eye(x.shape[0], dtype=x.dtype)
        return laplacian_by_method(scalar_fn, x, basis, method)

    def task(mlp_params, xs):
        return jax.vmap(partial(laplacian_at_point, mlp_params))(xs)

    return Workload(
        name=f"mlp_laplacian_{method}_h{hidden_width}",
        task=task,
        args=(params, points),
    )


def make_poisson_pinn_workload(
    *,
    hidden_width: int,
    method: StrategyName,
    seed: int,
) -> Workload:
    input_dim = 2
    hidden_layers = 128
    output_dim = 1
    grid_size = 16
    params = init_mlp_params(
        seed=seed,
        layer_dims=tuple([input_dim] + [hidden_width] * hidden_layers + [output_dim]),
    )
    grid = jnp.linspace(0.1, 0.9, grid_size, dtype=jnp.float32)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid, indexing="ij")
    points = jnp.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)

    def nn_scalar(nn_params, coord):
        activations = coord
        for weights, bias in nn_params[:-1]:
            activations = jnp.tanh(activations @ weights + bias)
        final_weights, final_bias = nn_params[-1]
        return jnp.squeeze(activations @ final_weights + final_bias)

    def trial_solution(nn_params, coord):
        x, y = coord
        boundary_factor = x * (1.0 - x) * y * (1.0 - y)
        return boundary_factor * nn_scalar(nn_params, coord)

    def poisson_residual_at_point(nn_params, coord):
        scalar_fn = lambda z: trial_solution(nn_params, z)
        basis = jnp.eye(coord.shape[0], dtype=coord.dtype)
        laplacian = laplacian_by_method(scalar_fn, coord, basis, method)
        forcing = jnp.sin(jnp.pi * coord[0]) * jnp.sin(jnp.pi * coord[1])
        return laplacian + forcing

    def pinn_loss(nn_params, collocation_points):
        residuals = jax.vmap(partial(poisson_residual_at_point, nn_params))(
            collocation_points
        )
        return jnp.mean(residuals**2)

    def task(nn_params, collocation_points):
        return jax.value_and_grad(pinn_loss)(nn_params, collocation_points)

    return Workload(
        name=f"poisson_pinn_{method}_h{hidden_width}",
        task=task,
        args=(params, points),
    )


def laplacian_by_method(
    scalar_fn: Callable[[Array], Array],
    x: Array,
    basis: Array,
    method: StrategyName,
) -> Array:
    if method == "hessian":
        return jnp.trace(jax.hessian(scalar_fn)(x))

    if method == "jvp_grad":
        grad_fn = jax.grad(scalar_fn)

        def diagonal_entry(direction):
            _, hvp = jax.jvp(grad_fn, (x,), (direction,))
            return jnp.dot(direction, hvp)

        return jnp.sum(jax.vmap(diagonal_entry)(basis))

    zero_direction = jnp.zeros_like(x)

    def second_directional_derivative(direction):
        _, series_out = jet.jet(scalar_fn, (x,), ((direction, zero_direction),))
        return series_out[1]

    return jnp.sum(jax.vmap(second_directional_derivative)(basis))


def init_mlp_params(
    *,
    seed: int,
    layer_dims: tuple[int, ...],
) -> tuple[tuple[Array, Array], ...]:
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    for layer_key, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:]):
        weights = jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32)
        weights = weights / jnp.sqrt(jnp.asarray(in_dim, dtype=jnp.float32))
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        layers.append((weights, bias))
    return tuple(layers)


def block_tree(tree):
    return jtu.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


if __name__ == "__main__":
    main()
