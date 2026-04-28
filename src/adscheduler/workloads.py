from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import jet

Array = jax.Array


@dataclass(frozen=True)
class DerivativeWorkload:
    """A concrete JAX program paired with a derivative task to trace."""

    name: str
    description: str
    derivative_task: Callable[..., Any]
    args: tuple[Any, ...]
    known_loop_steps: int | None = None
    assume_outer_grad: bool = True


@dataclass(frozen=True)
class TracedDerivativeWorkload:
    name: str
    description: str
    closed_jaxpr: Any
    known_loop_steps: int | None
    assume_outer_grad: bool


def available_derivative_workload_names() -> tuple[str, ...]:
    return tuple(_WORKLOAD_BUILDERS)


def normalize_derivative_workload_names(
    workload_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if workload_names is None:
        return available_derivative_workload_names()

    normalized = tuple(dict.fromkeys(workload_names))
    unknown = sorted(set(normalized) - set(_WORKLOAD_BUILDERS))
    if unknown:
        known = ", ".join(available_derivative_workload_names())
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown workloads: {unknown_names}. Known workloads: {known}")
    return normalized


def make_derivative_workload(name: str, *, seed: int = 0) -> DerivativeWorkload:
    try:
        builder = _WORKLOAD_BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(available_derivative_workload_names())
        raise ValueError(f"Unknown workload '{name}'. Known workloads: {known}") from exc
    return builder(seed)


def trace_derivative_workload(workload: DerivativeWorkload) -> TracedDerivativeWorkload:
    closed_jaxpr = jax.make_jaxpr(workload.derivative_task)(*workload.args)
    return TracedDerivativeWorkload(
        name=workload.name,
        description=workload.description,
        closed_jaxpr=closed_jaxpr,
        known_loop_steps=workload.known_loop_steps,
        assume_outer_grad=workload.assume_outer_grad,
    )


def trace_derivative_workloads(
    workload_names: Sequence[str] | None = None,
    *,
    seed: int = 0,
) -> dict[str, TracedDerivativeWorkload]:
    selected = normalize_derivative_workload_names(workload_names)
    return {
        name: trace_derivative_workload(make_derivative_workload(name, seed=seed))
        for name in selected
    }


def _make_mlp_laplacian_workload(seed: int) -> DerivativeWorkload:
    return _make_mlp_laplacian_strategy_workload(
        seed,
        name="mlp_laplacian",
        strategy="hessian",
    )


def _make_mlp_laplacian_hessian_workload(seed: int) -> DerivativeWorkload:
    return _make_mlp_laplacian_strategy_workload(
        seed,
        name="mlp_laplacian_hessian",
        strategy="hessian",
    )


def _make_mlp_laplacian_jvp_grad_workload(seed: int) -> DerivativeWorkload:
    return _make_mlp_laplacian_strategy_workload(
        seed,
        name="mlp_laplacian_jvp_grad",
        strategy="jvp_grad",
    )


def _make_mlp_laplacian_jet_workload(seed: int) -> DerivativeWorkload:
    return _make_mlp_laplacian_strategy_workload(
        seed,
        name="mlp_laplacian_jet",
        strategy="jet",
    )


def _make_poisson_pinn_workload(seed: int) -> DerivativeWorkload:
    return _make_poisson_pinn_strategy_workload(
        seed,
        name="poisson_pinn",
        strategy="hessian",
    )


def _make_poisson_pinn_hessian_workload(seed: int) -> DerivativeWorkload:
    return _make_poisson_pinn_strategy_workload(
        seed,
        name="poisson_pinn_hessian",
        strategy="hessian",
    )


def _make_poisson_pinn_jvp_grad_workload(seed: int) -> DerivativeWorkload:
    return _make_poisson_pinn_strategy_workload(
        seed,
        name="poisson_pinn_jvp_grad",
        strategy="jvp_grad",
    )


def _make_poisson_pinn_jet_workload(seed: int) -> DerivativeWorkload:
    return _make_poisson_pinn_strategy_workload(
        seed,
        name="poisson_pinn_jet",
        strategy="jet",
    )


def _make_poisson_pinn_strategy_workload(
    seed: int,
    *,
    name: str,
    strategy: str,
) -> DerivativeWorkload:
    params = _init_mlp_params(
        seed=seed,
        layer_dims=(2, 16, 16, 1),
    )
    grid = jnp.linspace(0.1, 0.9, 5, dtype=jnp.float32)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid, indexing="ij")
    collocation_points = jnp.stack(
        [mesh_x.reshape(-1), mesh_y.reshape(-1)],
        axis=1,
    )

    def nn_scalar(nn_params: tuple[tuple[Array, Array], ...], coord: Array) -> Array:
        activations = coord
        for weights, bias in nn_params[:-1]:
            activations = jnp.tanh(activations @ weights + bias)
        final_weights, final_bias = nn_params[-1]
        return jnp.squeeze(activations @ final_weights + final_bias)

    def trial_solution(nn_params: tuple[tuple[Array, Array], ...], coord: Array) -> Array:
        x, y = coord
        boundary_factor = x * (1.0 - x) * y * (1.0 - y)
        return boundary_factor * nn_scalar(nn_params, coord)

    def poisson_residual_at_point(
        nn_params: tuple[tuple[Array, Array], ...],
        coord: Array,
    ) -> Array:
        scalar_fn = lambda z: trial_solution(nn_params, z)
        basis = jnp.eye(coord.shape[0], dtype=coord.dtype)

        if strategy == "hessian":
            hessian = jax.hessian(scalar_fn)(coord)
            laplacian = hessian[0, 0] + hessian[1, 1]
        elif strategy == "jvp_grad":
            grad_fn = jax.grad(scalar_fn)

            def diagonal_entry(direction: Array) -> Array:
                _, hvp = jax.jvp(grad_fn, (coord,), (direction,))
                return jnp.dot(direction, hvp)

            laplacian = jnp.sum(jax.vmap(diagonal_entry)(basis))
        elif strategy == "jet":
            zero_direction = jnp.zeros_like(coord)

            def second_directional_derivative(direction: Array) -> Array:
                _, series_out = jet.jet(
                    scalar_fn,
                    (coord,),
                    ((direction, zero_direction),),
                )
                return series_out[1]

            laplacian = jnp.sum(jax.vmap(second_directional_derivative)(basis))
        else:
            raise ValueError(f"Unknown Poisson PINN derivative strategy: {strategy}")

        forcing = jnp.sin(jnp.pi * coord[0]) * jnp.sin(jnp.pi * coord[1])
        return laplacian + forcing

    def pinn_loss(nn_params: tuple[tuple[Array, Array], ...], points: Array) -> Array:
        residuals = jax.vmap(partial(poisson_residual_at_point, nn_params))(points)
        return jnp.mean(residuals**2)

    def derivative_task(
        nn_params: tuple[tuple[Array, Array], ...],
        points: Array,
    ):
        return jax.value_and_grad(pinn_loss)(nn_params, points)

    return DerivativeWorkload(
        name=name,
        description=(
            "Training loss and parameter gradient for a Poisson PINN with "
            "u=x(1-x)y(1-y)NN(x,y), where the loss differentiates through "
            f"second input derivatives via {_poisson_pinn_strategy_description(strategy)}"
            "."
        ),
        derivative_task=derivative_task,
        args=(params, collocation_points),
    )


def _make_mlp_laplacian_strategy_description(strategy: str) -> str:
    descriptions = {
        "hessian": "trace(jax.hessian(f)(x))",
        "jvp_grad": "sum_i jax.jvp(jax.grad(f), (x,), (e_i,))[1][i]",
        "jet": "sum_i second-order jax.experimental.jet coefficient along e_i",
    }
    return descriptions[strategy]


def _poisson_pinn_strategy_description(strategy: str) -> str:
    descriptions = {
        "hessian": "trace(jax.hessian)",
        "jvp_grad": "jax.jvp(jax.grad)",
        "jet": "jax.experimental.jet",
    }
    return descriptions[strategy]


def _make_mlp_laplacian_strategy_workload(
    seed: int,
    *,
    name: str,
    strategy: str,
) -> DerivativeWorkload:
    params = _init_mlp_laplacian_params(
        seed=seed,
        input_dim=3,
        hidden_dim=128,
        hidden_layers=64,
    )
    points = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float32).reshape(5, 3)

    def mlp_scalar(mlp_params: tuple[tuple[Array, Array], ...], x: Array) -> Array:
        activations = x
        for weights, bias in mlp_params[:-1]:
            activations = jnp.tanh(activations @ weights + bias)
        final_weights, final_bias = mlp_params[-1]
        output = activations @ final_weights + final_bias
        return jnp.squeeze(output)

    def laplacian_at_point(
        mlp_params: tuple[tuple[Array, Array], ...],
        x: Array,
    ) -> Array:
        basis = jnp.eye(x.shape[0], dtype=x.dtype)
        scalar_fn = lambda z: mlp_scalar(mlp_params, z)

        if strategy == "hessian":
            hessian = jax.hessian(scalar_fn)(x)
            return jnp.trace(hessian)

        if strategy == "jvp_grad":
            grad_fn = jax.grad(scalar_fn)

            def diagonal_entry(direction: Array) -> Array:
                _, hvp = jax.jvp(grad_fn, (x,), (direction,))
                return jnp.dot(direction, hvp)

            return jnp.sum(jax.vmap(diagonal_entry)(basis))

        if strategy == "jet":
            zero_direction = jnp.zeros_like(x)

            def second_directional_derivative(direction: Array) -> Array:
                _, series_out = jet.jet(scalar_fn, (x,), ((direction, zero_direction),))
                return series_out[1]

            return jnp.sum(jax.vmap(second_directional_derivative)(basis))

        raise ValueError(f"Unknown MLP Laplacian strategy: {strategy}")

    def derivative_task(
        mlp_params: tuple[tuple[Array, Array], ...],
        xs: Array,
    ) -> Array:
        return jax.vmap(partial(laplacian_at_point, mlp_params))(xs)

    return DerivativeWorkload(
        name=name,
        description=(
            "Input-space Laplacian of a 64-layer JAX MLP with hidden size 128 via "
            f"{_make_mlp_laplacian_strategy_description(strategy)}."
        ),
        derivative_task=derivative_task,
        args=(params, points),
    )


def _init_mlp_laplacian_params(
    *,
    seed: int,
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> tuple[tuple[Array, Array], ...]:
    return _init_mlp_params(
        seed=seed,
        layer_dims=tuple([input_dim] + [hidden_dim] * hidden_layers + [1]),
    )


def _init_mlp_params(
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


_WORKLOAD_BUILDERS: dict[str, Callable[[int], DerivativeWorkload]] = {
    "mlp_laplacian": _make_mlp_laplacian_workload,
    "mlp_laplacian_hessian": _make_mlp_laplacian_hessian_workload,
    "mlp_laplacian_jvp_grad": _make_mlp_laplacian_jvp_grad_workload,
    "mlp_laplacian_jet": _make_mlp_laplacian_jet_workload,
    "poisson_pinn": _make_poisson_pinn_workload,
    "poisson_pinn_hessian": _make_poisson_pinn_hessian_workload,
    "poisson_pinn_jvp_grad": _make_poisson_pinn_jvp_grad_workload,
    "poisson_pinn_jet": _make_poisson_pinn_jet_workload,
}
