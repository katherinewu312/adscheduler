#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import jet

Array = jax.Array
PINNParams = tuple[tuple[Array, Array], ...]
MethodName = Literal["hessian", "jvp_grad", "jet", "recurrence"]


@dataclass(frozen=True)
class ConvergenceConfig:
    seed: int
    grid_size: int
    hidden_layers: int
    hidden_dim: int
    max_steps: int
    eval_every: int
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    target_loss: float
    target_source: str


@dataclass(frozen=True)
class ConvergenceResult:
    method: str
    compile_overhead_sec: float
    initial_loss: float
    final_loss: float
    best_loss: float
    iterations_to_target_loss: int | None
    time_to_target_loss_sec: float | None
    time_to_target_loss_with_compile_sec: float | None
    total_train_time_sec: float
    total_wall_time_sec: float
    total_wall_time_with_compile_sec: float
    avg_step_time_ms: float
    p50_step_time_ms: float
    p90_step_time_ms: float
    loss_history: list[tuple[int, float]]


@dataclass(frozen=True)
class ConvergenceReport:
    config: ConvergenceConfig
    results: list[ConvergenceResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Poisson PINN benchmark and report statistical convergence "
            "as iterations to a loss threshold."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        action="append",
        choices=("hessian", "jvp_grad", "jet", "recurrence"),
        dest="methods",
        default=None,
        help="PINN derivative method to train. Repeat for multiple.",
    )
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="Absolute loss threshold. If omitted, use baseline initial loss times --relative-target.",
    )
    parser.add_argument(
        "--relative-target",
        type=float,
        default=0.5,
        help="Target as a fraction of the first method's initial loss when --target-loss is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "icml2025" / "pinn_convergence_results.json",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    methods: tuple[MethodName, ...] = tuple(args.methods or ("hessian", "jvp_grad", "jet"))
    points = make_collocation_points(args.grid_size)
    initial_params = init_pinn_params(
        seed=args.seed,
        input_dim=2,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        output_dim=1,
    )

    first_loss = float(
        np.asarray(make_pinn_loss(methods[0])(initial_params, points))
    )
    if args.target_loss is None:
        target_loss = first_loss * args.relative_target
        target_source = f"{methods[0]} initial loss * {args.relative_target:g}"
    else:
        target_loss = args.target_loss
        target_source = "absolute target"

    config = ConvergenceConfig(
        seed=args.seed,
        grid_size=args.grid_size,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        target_loss=target_loss,
        target_source=target_source,
    )

    results = [
        run_method(
            method=method,
            initial_params=initial_params,
            points=points,
            config=config,
        )
        for method in methods
    ]
    report = ConvergenceReport(config=config, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print_summary(report, args.output)
    if args.json:
        print("=== JSON ===")
        print(json.dumps(asdict(report), indent=2))


def run_method(
    *,
    method: MethodName,
    initial_params: PINNParams,
    points: Array,
    config: ConvergenceConfig,
) -> ConvergenceResult:
    loss_fn = make_pinn_loss(method)
    eval_loss = jax.jit(loss_fn)
    train_step = jax.jit(
        partial(
            adam_train_step,
            loss_fn=loss_fn,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        )
    )

    params = initial_params
    opt_state = init_adam_state(params)

    compile_start = time.perf_counter()
    compiled_eval_loss = eval_loss.lower(params, points).compile()
    compiled_train_step = train_step.lower(
        params,
        opt_state,
        points,
        jnp.asarray(1, dtype=jnp.int32),
    ).compile()
    initial_loss_array = compiled_eval_loss(params, points)
    initial_loss_array.block_until_ready()
    compile_overhead_sec = time.perf_counter() - compile_start

    initial_loss = float(np.asarray(initial_loss_array))
    best_loss = initial_loss
    final_loss = initial_loss
    loss_history: list[tuple[int, float]] = [(0, initial_loss)]
    iterations_to_target: int | None = 0 if initial_loss <= config.target_loss else None
    time_to_target_sec: float | None = 0.0 if iterations_to_target == 0 else None
    wall_time_to_target_sec: float | None = 0.0 if iterations_to_target == 0 else None
    step_times: list[float] = []
    cumulative_step_time_sec = 0.0
    cumulative_wall_time_sec = 0.0

    for step in range(1, config.max_steps + 1):
        start = time.perf_counter()
        params, opt_state, _ = compiled_train_step(
            params,
            opt_state,
            points,
            jnp.asarray(step, dtype=jnp.int32),
        )
        params = block_tree(params)
        step_time = time.perf_counter() - start
        step_times.append(step_time)
        cumulative_step_time_sec += step_time
        cumulative_wall_time_sec += step_time

        if step % config.eval_every == 0 or step == config.max_steps:
            eval_start = time.perf_counter()
            loss_array = compiled_eval_loss(params, points)
            loss_array.block_until_ready()
            cumulative_wall_time_sec += time.perf_counter() - eval_start
            loss = float(np.asarray(loss_array))
            final_loss = loss
            best_loss = min(best_loss, loss)
            loss_history.append((step, loss))
            if iterations_to_target is None and loss <= config.target_loss:
                iterations_to_target = step
                time_to_target_sec = cumulative_step_time_sec
                wall_time_to_target_sec = cumulative_wall_time_sec

    timings = np.asarray(step_times, dtype=np.float64)
    return ConvergenceResult(
        method=method,
        compile_overhead_sec=compile_overhead_sec,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        iterations_to_target_loss=iterations_to_target,
        time_to_target_loss_sec=time_to_target_sec,
        time_to_target_loss_with_compile_sec=(
            compile_overhead_sec + wall_time_to_target_sec
            if wall_time_to_target_sec is not None
            else None
        ),
        total_train_time_sec=cumulative_step_time_sec,
        total_wall_time_sec=cumulative_wall_time_sec,
        total_wall_time_with_compile_sec=compile_overhead_sec + cumulative_wall_time_sec,
        avg_step_time_ms=float(np.mean(timings) * 1e3),
        p50_step_time_ms=float(np.percentile(timings, 50) * 1e3),
        p90_step_time_ms=float(np.percentile(timings, 90) * 1e3),
        loss_history=loss_history,
    )


def make_pinn_loss(method: MethodName) -> Callable[[PINNParams, Array], Array]:
    def poisson_residual_at_point(params: PINNParams, coord: Array) -> Array:
        forcing = jnp.sin(jnp.pi * coord[0]) * jnp.sin(jnp.pi * coord[1])
        if method == "recurrence":
            laplacian = trial_solution_laplacian_recurrence(params, coord)
        else:
            scalar_fn = lambda z: trial_solution(params, z)
            basis = jnp.eye(coord.shape[0], dtype=coord.dtype)
            laplacian = laplacian_by_method(scalar_fn, coord, basis, method)
        return laplacian + forcing

    def pinn_loss(params: PINNParams, collocation_points: Array) -> Array:
        residuals = jax.vmap(lambda coord: poisson_residual_at_point(params, coord))(
            collocation_points
        )
        return jnp.mean(residuals**2)

    return pinn_loss


def adam_train_step(
    params: PINNParams,
    opt_state,
    points: Array,
    step: Array,
    *,
    loss_fn: Callable[[PINNParams, Array], Array],
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
):
    loss, grads = jax.value_and_grad(loss_fn)(params, points)
    moments, velocities = opt_state
    moments = jtu.tree_map(
        lambda moment, grad: beta1 * moment + (1.0 - beta1) * grad,
        moments,
        grads,
    )
    velocities = jtu.tree_map(
        lambda velocity, grad: beta2 * velocity + (1.0 - beta2) * (grad * grad),
        velocities,
        grads,
    )
    step_float = step.astype(jnp.float32)
    moment_scale = 1.0 - beta1**step_float
    velocity_scale = 1.0 - beta2**step_float
    params = jtu.tree_map(
        lambda param, moment, velocity: param
        - learning_rate
        * (moment / moment_scale)
        / (jnp.sqrt(velocity / velocity_scale) + eps),
        params,
        moments,
        velocities,
    )
    return params, (moments, velocities), loss


def laplacian_by_method(
    scalar_fn: Callable[[Array], Array],
    coord: Array,
    basis: Array,
    method: str,
) -> Array:
    if method == "hessian":
        return jnp.trace(jax.hessian(scalar_fn)(coord))

    if method == "jvp_grad":
        grad_fn = jax.grad(scalar_fn)

        def diagonal_entry(direction: Array) -> Array:
            _, hvp = jax.jvp(grad_fn, (coord,), (direction,))
            return jnp.dot(direction, hvp)

        return jnp.sum(jax.vmap(diagonal_entry)(basis))

    if method == "jet":
        zero_direction = jnp.zeros_like(coord)

        def second_directional_derivative(direction: Array) -> Array:
            _, series_out = jet.jet(scalar_fn, (coord,), ((direction, zero_direction),))
            return series_out[1]

        return jnp.sum(jax.vmap(second_directional_derivative)(basis))

    raise ValueError(f"Unknown PINN method: {method}")


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


def trial_solution_laplacian_recurrence(params: PINNParams, coord: Array) -> Array:
    nn_value, nn_grad, nn_laplacian = tanh_mlp_value_grad_laplacian(params, coord)
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


def tanh_mlp_value_grad_laplacian(params: PINNParams, coord: Array):
    activations = coord
    gradient = jnp.eye(coord.shape[0], dtype=coord.dtype)
    laplacian = jnp.zeros((coord.shape[0],), dtype=coord.dtype)

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


def init_pinn_params(
    *,
    seed: int,
    input_dim: int,
    hidden_layers: int,
    hidden_dim: int,
    output_dim: int,
) -> PINNParams:
    key = jax.random.PRNGKey(seed)
    layer_dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    for layer_key, in_dim, out_dim in zip(keys, layer_dims[:-1], layer_dims[1:]):
        weights = jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32)
        weights = weights / jnp.sqrt(jnp.asarray(in_dim, dtype=jnp.float32))
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        layers.append((weights, bias))
    return tuple(layers)


def make_collocation_points(grid_size: int) -> Array:
    grid = jnp.linspace(0.1, 0.9, grid_size, dtype=jnp.float32)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid, indexing="ij")
    return jnp.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)


def init_adam_state(params: PINNParams):
    zeros = jtu.tree_map(jnp.zeros_like, params)
    return zeros, zeros


def block_tree(tree):
    return jtu.tree_map(lambda x: x.block_until_ready(), tree)


def print_summary(report: ConvergenceReport, output_path: Path) -> None:
    cfg = report.config
    print("=== PINN Convergence Benchmark ===")
    print(
        "workload: "
        f"grid_size={cfg.grid_size} hidden_layers={cfg.hidden_layers} "
        f"hidden_dim={cfg.hidden_dim} seed={cfg.seed}"
    )
    print(
        "optimizer: "
        f"adam lr={cfg.learning_rate:g} beta1={cfg.beta1:g} "
        f"beta2={cfg.beta2:g} eps={cfg.eps:g}"
    )
    print(
        f"target_loss: {cfg.target_loss:.6e} ({cfg.target_source}); "
        f"max_steps={cfg.max_steps} eval_every={cfg.eval_every}"
    )
    print()
    for result in report.results:
        iterations = (
            str(result.iterations_to_target_loss)
            if result.iterations_to_target_loss is not None
            else "not reached"
        )
        time_to_target = (
            f"{result.time_to_target_loss_sec:.6f}"
            if result.time_to_target_loss_sec is not None
            else "not reached"
        )
        time_to_target_with_compile = (
            f"{result.time_to_target_loss_with_compile_sec:.6f}"
            if result.time_to_target_loss_with_compile_sec is not None
            else "not reached"
        )
        print(f"[{result.method}]")
        print(f"  compile_overhead_sec: {result.compile_overhead_sec:.6f}")
        print(f"  initial_loss: {result.initial_loss:.6e}")
        print(f"  final_loss: {result.final_loss:.6e}")
        print(f"  best_loss: {result.best_loss:.6e}")
        print(f"  iterations_to_target_loss: {iterations}")
        print(f"  time_to_target_loss_sec: {time_to_target}")
        print(f"  time_to_target_loss_with_compile_sec: {time_to_target_with_compile}")
        print(f"  total_train_time_sec: {result.total_train_time_sec:.6f}")
        print(f"  total_wall_time_sec: {result.total_wall_time_sec:.6f}")
        print(
            "  total_wall_time_with_compile_sec: "
            f"{result.total_wall_time_with_compile_sec:.6f}"
        )
        print(f"  avg_step_time_ms: {result.avg_step_time_ms:.3f}")
        print(f"  p50_step_time_ms: {result.p50_step_time_ms:.3f}")
        print(f"  p90_step_time_ms: {result.p90_step_time_ms:.3f}")
        print()
    print(f"wrote: {output_path}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.grid_size < 2:
        raise ValueError("--grid-size must be >= 2")
    if args.hidden_layers < 1:
        raise ValueError("--hidden-layers must be >= 1")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.target_loss is not None and args.target_loss < 0:
        raise ValueError("--target-loss must be non-negative")
    if args.relative_target <= 0:
        raise ValueError("--relative-target must be positive")


if __name__ == "__main__":
    main()
