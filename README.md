# adscheduler

`adscheduler` is a small JAX prototype for studying AD strategy implementations and
derivative-program structure. The core framing is: the input is a JAX program
plus a derivative task, and the system traces that workload to a closed `jaxpr`,
extracts IR features, and compares differentiation strategies when alternatives
are available.

Laplacian workload: Given a scalar MLP f(x) : R^d -> R, how fast can JAX/XLA compute the input space Laplacian at many input points?

The Laplacian is:

Delta f(x) = d²f/dx_0² + d²f/dx_1² + ... + d²f/dx_{d-1}²
           = trace(Hessian(f)(x))
where x is a vector.

In the current config, the Laplacian MLP uses:

num_points = 256
input_dim = 3
hidden_dim = 256
hidden_layers = 128
activation = tanh

The MLP itself is:

activations = x
for weights, bias in mlp_params[:-1]:
    activations = jnp.tanh(activations @ weights + bias)

final_weights, final_bias = mlp_params[-1]
output = activations @ final_weights + final_bias
return jnp.squeeze(output)

For each point x, the benchmark computes one scalar:

Delta f_theta(x)

Then jax.vmap(...) runs that over all num_points, so the output is a vector of shape:
(num_points,)

The Laplacian benchmarks compute the same mathematical object, but ask JAX to produce it in different AD forms.

Here are the different ways JAX produces the Laplacian in different AD forms:

- MLP_LAPLACIAN_HESSIAN

Uses:
```
hessian = jax.hessian(scalar_fn)(x)
return jnp.trace(hessian)
```

This builds the full Hessian matrix:
H[i, j] = d²f / dx_i dx_j
Then it sums the diagonal:
H[0,0] + H[1,1] + H[2,2].

For input_dim = 3, the Hessian is 3x3.

- MLP_LAPLACIAN_JVP_GRAD

Uses:
```
grad_fn = jax.grad(scalar_fn)

_, hvp = jax.jvp(grad_fn, (x,), (direction,))
return jnp.dot(direction, hvp)
```

Here jax.grad(scalar_fn) gives:

grad f(x)
Then jax.jvp(grad_fn, ..., direction) computes the directional derivative of the gradient:

Hessian(f)(x) @ direction
The code uses basis directions:

e_0 = [1,0,0]
e_1 = [0,1,0]
e_2 = [0,0,1]
So each JVP extracts one diagonal Hessian entry:

e_i dot (H @ e_i) = H[i,i]
Then it sums them.


- MLP_LAPLACIAN_JET
 Propagates Taylor coefficients through the program. For each basis direction e_i, it conceptually asks:

What is the second derivative of f(x + t e_i) with respect to t?
That gives:

d²/dt² f(x + t e_i) at t=0 = e_i^T H e_i = H[i,i]
Then the code sums over all basis directions, again giving the Laplacian.

The Poisson benchmarks are similar, but one level larger.  They define:

u(x,y) = x(1-x)y(1-y) NN(x,y)
Then compute a Poisson residual:

residual(x,y) = Delta u(x,y) + forcing(x,y)
Then compute:

jax.value_and_grad(pinn_loss)(params, points)
So the PINN benchmark output is not just the Laplacian values. It is:

(loss, gradient_of_loss_wrt_network_parameters)



compile_overhead_ms measures how long it takes to compile the StableHLO program with XLA before execution timing starts.


The current included workloads are an input-space Laplacian of a JAX MLP and a
Poisson PINN residual-gradient workload.

We claim that compiler optimizations can improve the performance of higher-order AD in tensor programs, since derivative code often contains explotable high-level structure that XLA cannot pick up. So far: tanh-MLP Laplacian recurrences, derivative-aware constant propagation, and structural zero elimination.

For example, jax.jet could produce cases where chlo.lgamma of constants are not folded into constants by XLA during compile time.

Generic XLA can eliminate unused values, but it will not usually redesign a dense derivative computation into a sparse derivative computation without that higher-level structure.

** XLA does generic CSE, DCE, constant folding, fusion, etc.
OpenXLA documents those generic HLO passes, but they operate on array ops and shapes, not on derivative meaning.

Good AD-Aware Pass Candidates

1. Laplacian / diagonal-Hessian specialization

If the user only needs trace(H), compute only diagonal Hessian entries.
Avoid materializing full Hessians or off-diagonal terms.
Especially relevant to your MLP Laplacian benchmark.

2. Sparse derivative propagation

Track known-zero Jacobian/Hessian blocks.
Remove tangent/cotangent computations that are structurally zero.
Useful for PINNs, PDE operators, boundary factors, separable coordinates.

3. Derivative-aware constant propagation

Fold constants created by AD rules, especially in Taylor/jet expansions.
The paper specifically notes chlo.lgamma(constant) not being folded by XLA in their setup.
XLA has generic constant folding, but may miss frontend/CHLO/AD-generated cases.

4. Zero tangent / zero cotangent elimination

AD often creates zero tangent lanes, zero cotangents, or zero Taylor coefficients.
A pass can propagate “this derivative component is identically zero,” not just “this tensor currently equals a literal zero.”

5. Basis-vector JVP sharing

Laplacian via repeated JVPs uses standard basis directions.
Many directions share the same primal computation.
A pass can batch/share primal work across basis-vector derivative evaluations.

6. Primal recomputation sharing

Higher-order AD can duplicate primal subexpressions across derivative paths.
XLA can CSE identical HLO, but may miss equivalences hidden behind different AD expansion structures.

7. Trace/Jacobian/Hessian use-site slicing

If downstream code only uses diag(H), rows of J, or a Hessian-vector product, push that selection backward into derivative generation.
This is stronger than DCE because it changes how derivatives are produced.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib numpy
```

## Quick Run

Trace the MLP Laplacian derivative workload:

```bash
python3 scripts/run_trace_analysis.py \
  --workload mlp_laplacian_hessian \
  --workload mlp_laplacian_jvp_grad \
  --workload mlp_laplacian_jet
```

Trace the Poisson PINN training-gradient workload:

```bash
python3 scripts/run_trace_analysis.py \
  --workload poisson_pinn_hessian \
  --workload poisson_pinn_jvp_grad \
  --workload poisson_pinn_jet
```

Add `--print-jaxpr` to dump full IR text.

## Runtime Benchmarks

Benchmark the MLP Laplacian derivative strategies:

```bash
python3 scripts/run_workload_benchmark.py \
  --workload mlp_laplacian_hessian \
  --workload mlp_laplacian_jvp_grad \
  --workload mlp_laplacian_jet \
  --warmup-runs 5 \
  --runs 100
```

The explicit `hessian`, `jvp_grad`, and `jet` workloads report runtime
measurements before and after the StableHLO pass pipeline.

Benchmark Poisson PINN derivative strategies:

```bash
python3 scripts/run_workload_benchmark.py \
  --workload poisson_pinn_hessian \
  --workload poisson_pinn_jvp_grad \
  --workload poisson_pinn_jet \
  --warmup-runs 5 \
  --runs 100
```

The explicit `hessian`, `jvp_grad`, and `jet` PINN workloads also report
runtime measurements before and after the StableHLO pass pipeline.

## StableHLO Passes

Lower workloads to StableHLO and run the first compiler-aware analysis passes:

```bash
python3 scripts/run_stablehlo_passes.py \
  --workload mlp_laplacian_hessian \
  --workload mlp_laplacian_jvp_grad \
  --workload mlp_laplacian_jet
```

The default StableHLO optimization passes are AD-aware passes that can produce
new optimized StableHLO for the benchmark workloads:

- `tanh_mlp_laplacian_recurrence`: rewrites known tanh-MLP Laplacian and Poisson PINN workloads from nested input AD to a direct value/gradient/Laplacian recurrence.
- `derivative_constant_propagation`: folds scalar/splat constants generated by AD lowering.
- `structural_zero_elimination`: removes AD-created zero tangent/cotangent operations such as zero broadcasts, add/subtract-by-zero aliases, zero multiplies, and zero dot-generals.

## What the Benchmarks Report

The workload benchmark reports JIT compile overhead and timed steady-state
execution for the explicit derivative strategies.

## Repository Map

- `src/adscheduler/workloads.py`: MLP Laplacian, Poisson PINN, and general workload tracing helpers.
- `src/adscheduler/stablehlo_passes.py`: StableHLO lowering and compiler-aware analysis passes.
- `src/adscheduler/ir_analysis.py`: recursive `jaxpr` feature extraction.
- `scripts/run_trace_analysis.py`: CLI for tracing derivative workloads and IR reports.
- `scripts/run_stablehlo_passes.py`: CLI for lowering to StableHLO and running analysis passes.
- `scripts/run_workload_benchmark.py`: CLI for benchmarking derivative workload runtimes.

## Notes

Got rid of argument flattening, NumPy conversion, device transfer, and output normalization on every run (hence lowering the runtimes for the after compiler passes.)
