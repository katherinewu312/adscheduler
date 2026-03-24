from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as jtu

Array = jax.Array
Batch = tuple[Array, Array]
Params = dict[str, Array]
MetaGradResult = tuple[Array, Params]

SCHEDULE_ROR = "ror"
SCHEDULE_FOR = "for"
SCHEDULE_JACREV = "jacrev"
SCHEDULE_JACFWD = "jacfwd"
SCHEDULE_ROR_REMAT = "ror_remat"
SCHEDULE_FOR_REMAT = "for_remat"
SCHEDULE_JACREV_REMAT = "jacrev_remat"
SCHEDULE_JACFWD_REMAT = "jacfwd_remat"


@dataclass(frozen=True)
class TaskBatches:
    """Support/query batches for one synthetic few-shot task."""

    support: Batch
    query: Batch


@dataclass(frozen=True)
class ExampleConfig:
    """Config used to instantiate a synthetic MAML tracing example."""

    in_dim: int = 8
    hidden_dim: int = 32
    out_dim: int = 1
    n_support: int = 16
    n_query: int = 16
    inner_steps: int = 3
    inner_lr: float = 0.1
    outer_lr: float = 0.01


@dataclass(frozen=True)
class TracedMAMLPrograms:
    """Closed jaxprs captured for baseline and alternative AD schedules."""

    objective: Any
    first_order_meta_grad: Any
    schedule_meta_grads: dict[str, Any]
    schedule_train_steps: dict[str, Any]

    def _meta_grad_for(self, schedule_name: str) -> Any:
        if schedule_name not in self.schedule_meta_grads:
            known = ", ".join(self.schedule_meta_grads)
            raise KeyError(f"Schedule '{schedule_name}' not traced. Available: {known}")
        return self.schedule_meta_grads[schedule_name]

    def _train_step_for(self, schedule_name: str) -> Any:
        if schedule_name not in self.schedule_train_steps:
            known = ", ".join(self.schedule_train_steps)
            raise KeyError(f"Schedule '{schedule_name}' not traced. Available: {known}")
        return self.schedule_train_steps[schedule_name]

    @property
    def meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_ROR)

    @property
    def train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_ROR)

    @property
    def forward_over_reverse_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_FOR)

    @property
    def forward_over_reverse_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_FOR)

    @property
    def jacrev_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_JACREV)

    @property
    def jacrev_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_JACREV)

    @property
    def jacfwd_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_JACFWD)

    @property
    def jacfwd_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_JACFWD)

    @property
    def ror_remat_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_ROR_REMAT)

    @property
    def ror_remat_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_ROR_REMAT)

    @property
    def for_remat_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_FOR_REMAT)

    @property
    def for_remat_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_FOR_REMAT)

    @property
    def jacrev_remat_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_JACREV_REMAT)

    @property
    def jacrev_remat_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_JACREV_REMAT)

    @property
    def jacfwd_remat_meta_grad(self) -> Any:
        return self._meta_grad_for(SCHEDULE_JACFWD_REMAT)

    @property
    def jacfwd_remat_train_step(self) -> Any:
        return self._train_step_for(SCHEDULE_JACFWD_REMAT)


def init_mlp_params(
    key: Array,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
) -> Params:
    k1, k2 = jax.random.split(key)
    return {
        "w1": 0.1 * jax.random.normal(k1, (in_dim, hidden_dim)),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": 0.1 * jax.random.normal(k2, (hidden_dim, out_dim)),
        "b2": jnp.zeros((out_dim,)),
    }


def mlp(params: Params, x: Array) -> Array:
    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def mse_loss(pred: Array, target: Array) -> Array:
    return jnp.mean((pred - target) ** 2)


def batch_loss(params: Params, batch: Batch) -> Array:
    x, y = batch
    return mse_loss(mlp(params, x), y)


def _tree_sub(lhs: Params, rhs: Params) -> Params:
    return jtu.tree_map(lambda x, y: x - y, lhs, rhs)


def _tree_scale(tree: Params, scalar: float) -> Params:
    return jtu.tree_map(lambda x: scalar * x, tree)


def inner_update(
    params: Params,
    support_batch: Batch,
    inner_lr: float,
    *,
    stop_higher_order: bool,
) -> Params:
    with jax.named_scope("inner_grad"):
        grads = jax.grad(batch_loss)(params, support_batch)

    if stop_higher_order:
        grads = jtu.tree_map(jax.lax.stop_gradient, grads)

    return _tree_sub(params, _tree_scale(grads, inner_lr))


def adapt_params(
    initial_params: Params,
    support_batch: Batch,
    inner_lr: float,
    inner_steps: int,
    *,
    stop_higher_order: bool,
    remat_inner: bool = False,
) -> Params:
    def inner_step_fn(params: Params) -> Params:
        return inner_update(
            params,
            support_batch,
            inner_lr,
            stop_higher_order=stop_higher_order,
        )

    if remat_inner:
        inner_step_fn = jax.checkpoint(inner_step_fn)

    def body_fn(_: int, params: Params) -> Params:
        return inner_step_fn(params)

    return jax.lax.fori_loop(0, inner_steps, body_fn, initial_params)


def maml_meta_objective(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    stop_higher_order: bool,
    remat_inner: bool = False,
) -> Array:
    adapted = adapt_params(
        initial_params,
        batches.support,
        inner_lr,
        inner_steps,
        stop_higher_order=stop_higher_order,
        remat_inner=remat_inner,
    )
    return batch_loss(adapted, batches.query)


def meta_train_step(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
    *,
    stop_higher_order: bool,
) -> tuple[Params, Array, Params]:
    if stop_higher_order:
        objective = partial(
            maml_meta_objective,
            batches=batches,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            stop_higher_order=True,
        )
        with jax.named_scope("meta_grad_first_order"):
            meta_loss, meta_grads = jax.value_and_grad(objective)(initial_params)
        updated_params = _tree_sub(initial_params, _tree_scale(meta_grads, outer_lr))
        return updated_params, meta_loss, meta_grads

    return meta_train_step_with_schedule(
        initial_params,
        batches,
        inner_lr,
        outer_lr,
        inner_steps,
        schedule_name=SCHEDULE_ROR,
    )


def _full_objective(
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    remat_inner: bool = False,
) -> Callable[[Params], Array]:
    return partial(
        maml_meta_objective,
        batches=batches,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        stop_higher_order=False,
        remat_inner=remat_inner,
    )


def meta_grad_reverse_over_reverse(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    remat_inner: bool = False,
) -> MetaGradResult:
    objective = _full_objective(
        batches,
        inner_lr,
        inner_steps,
        remat_inner=remat_inner,
    )
    with jax.named_scope("meta_grad_ror"):
        return jax.value_and_grad(objective)(initial_params)


def meta_grad_forward_over_reverse(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    remat_inner: bool = False,
) -> MetaGradResult:
    """Compute full MAML meta-grad with a forward-over-reverse schedule.

    This keeps reverse-mode for per-step support/query gradients while using
    forward-mode JVPs through the support gradient to propagate tangent vectors
    across the inner-loop update map.
    """

    support_loss = lambda p: batch_loss(p, batches.support)
    query_loss = lambda p: batch_loss(p, batches.query)
    support_grad = jax.grad(support_loss)
    if remat_inner:
        support_grad = jax.checkpoint(support_grad)

    params_vec, unravel = ravel_pytree(initial_params)
    n_params = params_vec.shape[0]
    jacobian_t = jnp.eye(n_params, dtype=params_vec.dtype)
    params = initial_params

    for _ in range(inner_steps):
        with jax.named_scope("inner_grad"):
            support_grads = support_grad(params)
        support_grad_vec, _ = ravel_pytree(support_grads)

        def propagate_tangent(tangent_vec: Array) -> Array:
            tangent_tree = unravel(tangent_vec)
            with jax.named_scope("inner_hvp"):
                _, hvp_tree = jax.jvp(support_grad, (params,), (tangent_tree,))
            hvp_vec, _ = ravel_pytree(hvp_tree)
            return tangent_vec - inner_lr * hvp_vec

        with jax.named_scope("forward_tangent_propagation"):
            jacobian_t = jax.vmap(propagate_tangent)(jacobian_t)

        params_vec = params_vec - inner_lr * support_grad_vec
        params = unravel(params_vec)

    with jax.named_scope("query_grad"):
        meta_loss, query_grads = jax.value_and_grad(query_loss)(params)

    query_grad_vec, _ = ravel_pytree(query_grads)
    meta_grad_vec = jacobian_t @ query_grad_vec
    meta_grads = unravel(meta_grad_vec)
    return meta_loss, meta_grads


def meta_grad_jacrev(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    remat_inner: bool = False,
) -> MetaGradResult:
    objective = _full_objective(
        batches,
        inner_lr,
        inner_steps,
        remat_inner=remat_inner,
    )
    with jax.named_scope("meta_grad_jacrev"):
        meta_loss = objective(initial_params)
        meta_grads = jax.jacrev(objective)(initial_params)
    return meta_loss, meta_grads


def meta_grad_jacfwd(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    inner_steps: int,
    *,
    remat_inner: bool = False,
) -> MetaGradResult:
    objective = _full_objective(
        batches,
        inner_lr,
        inner_steps,
        remat_inner=remat_inner,
    )
    with jax.named_scope("meta_grad_jacfwd"):
        meta_loss = objective(initial_params)
        meta_grads = jax.jacfwd(objective)(initial_params)
    return meta_loss, meta_grads


SCHEDULE_REGISTRY: dict[str, Callable[[Params, TaskBatches, float, int], MetaGradResult]] = {
    SCHEDULE_ROR: partial(meta_grad_reverse_over_reverse, remat_inner=False),
    SCHEDULE_FOR: partial(meta_grad_forward_over_reverse, remat_inner=False),
    SCHEDULE_JACREV: partial(meta_grad_jacrev, remat_inner=False),
    SCHEDULE_JACFWD: partial(meta_grad_jacfwd, remat_inner=False),
    SCHEDULE_ROR_REMAT: partial(meta_grad_reverse_over_reverse, remat_inner=True),
    SCHEDULE_FOR_REMAT: partial(meta_grad_forward_over_reverse, remat_inner=True),
    SCHEDULE_JACREV_REMAT: partial(meta_grad_jacrev, remat_inner=True),
    SCHEDULE_JACFWD_REMAT: partial(meta_grad_jacfwd, remat_inner=True),
}


def available_schedule_names() -> tuple[str, ...]:
    return tuple(SCHEDULE_REGISTRY)


def get_meta_grad_schedule(
    schedule_name: str,
) -> Callable[[Params, TaskBatches, float, int], MetaGradResult]:
    try:
        return SCHEDULE_REGISTRY[schedule_name]
    except KeyError as exc:
        known = ", ".join(available_schedule_names())
        raise ValueError(f"Unknown schedule '{schedule_name}'. Known schedules: {known}") from exc


def normalize_schedule_names(schedule_names: Sequence[str] | None) -> tuple[str, ...]:
    if schedule_names is None:
        return available_schedule_names()

    normalized = tuple(dict.fromkeys(schedule_names))
    unknown = sorted(set(normalized) - set(SCHEDULE_REGISTRY))
    if unknown:
        known = ", ".join(available_schedule_names())
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown schedules: {unknown_names}. Known schedules: {known}")
    return normalized


def meta_train_step_with_schedule(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
    *,
    schedule_name: str,
) -> tuple[Params, Array, Params]:
    schedule_fn = get_meta_grad_schedule(schedule_name)
    with jax.named_scope(f"meta_grad_{schedule_name}"):
        meta_loss, meta_grads = schedule_fn(
            initial_params,
            batches,
            inner_lr,
            inner_steps,
        )
    updated_params = _tree_sub(initial_params, _tree_scale(meta_grads, outer_lr))
    return updated_params, meta_loss, meta_grads


def meta_train_step_forward_over_reverse(
    initial_params: Params,
    batches: TaskBatches,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
) -> tuple[Params, Array, Params]:
    return meta_train_step_with_schedule(
        initial_params,
        batches,
        inner_lr,
        outer_lr,
        inner_steps,
        schedule_name=SCHEDULE_FOR,
    )


def make_synthetic_task_batches(
    key: Array,
    *,
    in_dim: int,
    out_dim: int,
    n_support: int,
    n_query: int,
) -> TaskBatches:
    key_w, key_b, key_support, key_query = jax.random.split(key, 4)

    task_w = jax.random.normal(key_w, (in_dim, out_dim))
    task_b = jax.random.normal(key_b, (out_dim,))

    def sample_data(sample_key: Array, n_examples: int) -> Batch:
        x = jax.random.normal(sample_key, (n_examples, in_dim))
        y = jnp.tanh(x @ task_w + task_b)
        return x, y

    return TaskBatches(
        support=sample_data(key_support, n_support),
        query=sample_data(key_query, n_query),
    )


def make_example_state(
    seed: int = 0,
    config: ExampleConfig | None = None,
) -> tuple[Params, TaskBatches, ExampleConfig]:
    config = config or ExampleConfig()
    key = jax.random.PRNGKey(seed)
    key_params, key_task = jax.random.split(key)

    params = init_mlp_params(
        key_params,
        config.in_dim,
        config.hidden_dim,
        config.out_dim,
    )
    batches = make_synthetic_task_batches(
        key_task,
        in_dim=config.in_dim,
        out_dim=config.out_dim,
        n_support=config.n_support,
        n_query=config.n_query,
    )
    return params, batches, config


def trace_maml_programs(
    params: Params,
    batches: TaskBatches,
    *,
    inner_lr: float,
    outer_lr: float,
    inner_steps: int,
    schedule_names: Sequence[str] | None = None,
) -> TracedMAMLPrograms:
    objective = _full_objective(batches, inner_lr, inner_steps)
    first_order_objective = partial(
        maml_meta_objective,
        batches=batches,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        stop_higher_order=True,
    )
    selected_schedules = normalize_schedule_names(schedule_names)

    schedule_meta_grads: dict[str, Any] = {}
    schedule_train_steps: dict[str, Any] = {}
    for schedule_name in selected_schedules:
        meta_grad_program = partial(
            get_meta_grad_schedule(schedule_name),
            batches=batches,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )
        train_step_program = partial(
            meta_train_step_with_schedule,
            batches=batches,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps,
            schedule_name=schedule_name,
        )

        schedule_meta_grads[schedule_name] = jax.make_jaxpr(meta_grad_program)(params)
        schedule_train_steps[schedule_name] = jax.make_jaxpr(train_step_program)(params)

    return TracedMAMLPrograms(
        objective=jax.make_jaxpr(objective)(params),
        first_order_meta_grad=jax.make_jaxpr(jax.grad(first_order_objective))(params),
        schedule_meta_grads=schedule_meta_grads,
        schedule_train_steps=schedule_train_steps,
    )
