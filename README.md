# adscheduler

`adscheduler` is a small JAX prototype for studying AD schedule selection and
derivative-program structure. The core framing is: the input is a JAX program
plus a derivative task, and the system traces that workload to a closed `jaxpr`,
extracts IR features, and compares differentiation strategies when alternatives
are available.

The current included workloads are an input-space Laplacian of a JAX MLP and a
Poisson PINN residual-gradient workload.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib numpy
```

## Quick Run

Trace the MLP Laplacian derivative workload:

```bash
python3 scripts/run_trace_analysis.py --workload mlp_laplacian
python3 scripts/run_trace_analysis.py \
  --workload mlp_laplacian_hessian \
  --workload mlp_laplacian_jvp_grad \
  --workload mlp_laplacian_jet
```

Trace the Poisson PINN training-gradient workload:

```bash
python3 scripts/run_trace_analysis.py --workload poisson_pinn
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
  --workload mlp_laplacian_auto \
  --warmup-runs 5 \
  --runs 100
```

Benchmark Poisson PINN derivative strategies:

```bash
python3 scripts/run_workload_benchmark.py \
  --workload poisson_pinn_hessian \
  --workload poisson_pinn_jvp_grad \
  --workload poisson_pinn_jet \
  --workload poisson_pinn_auto \
  --warmup-runs 3 \
  --runs 10
```

Run fixed Laplacian schedule benchmarks:

```bash
python3 scripts/run_laplacian_schedule_benchmark.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd \
  --schedule ror_remat --schedule for_remat --schedule jacrev_remat --schedule jacfwd_remat \
  --hidden-layers 64 \
  --hidden-dim 128 \
  --outer-steps 100 \
  --eval-every 10
```

Run Laplacian with warmup-based auto selection:

```bash
python3 scripts/run_laplacian_schedule_benchmark.py \
  --include-auto \
  --warmup-steps 3 \
  --warmup-loss-tolerance 0.10 \
  --max-params-for-forward-like 50000 \
  --memory-budget-mb 4096
```

Run fixed PINN schedule benchmarks:

```bash
python3 scripts/run_pinn_schedule_benchmark.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd \
  --schedule ror_remat --schedule for_remat --schedule jacrev_remat --schedule jacfwd_remat \
  --outer-steps 30 \
  --eval-every 10
```

Run PINN with warmup-based auto selection:

```bash
python3 scripts/run_pinn_schedule_benchmark.py \
  --include-auto \
  --warmup-steps 3 \
  --warmup-loss-tolerance 0.10 \
  --memory-budget-mb 4096
```

## What the Benchmarks Report

The workload benchmark reports JIT compile overhead and timed steady-state
execution for the explicit derivative strategies. The auto workload first runs
warmup-based strategy selection, then benchmarks the selected strategy as an
additional candidate.

The Laplacian and PINN schedule benchmarks report:

- Numerical metrics: final/best max absolute error versus a reference strategy.
- Runtime metrics: compile overhead, avg/p50/p90 derivative-evaluation time.
- Memory metrics: peak host memory and device memory when available.
- IR metrics: equation count, loop nesting, and higher-order-site counts.
- Auto-selection metrics: chosen schedule and runtime regret versus the oracle fixed baseline.

## How Auto Selection Works

The selector runs a short warmup for each candidate, records compile + runtime
behavior, applies simple guards such as parameter-count gating for forward-like
schedules, and picks the best candidate under a horizon-aware score.

This is exploratory. It is useful for validating that derivative strategy choice
materially affects compile/runtime behavior before adding a stronger static cost
model or learned policy.

## Repository Map

- `src/adscheduler/workloads.py`: MLP Laplacian, Poisson PINN, and general workload tracing helpers.
- `src/adscheduler/laplacian_benchmark.py`: fixed and auto schedule evaluation for MLP Laplacians.
- `src/adscheduler/pinn_benchmark.py`: fixed and auto schedule evaluation for Poisson PINNs.
- `src/adscheduler/ir_analysis.py`: recursive `jaxpr` feature extraction.
- `scripts/run_trace_analysis.py`: CLI for tracing derivative workloads and IR reports.
- `scripts/run_laplacian_schedule_benchmark.py`: CLI for Laplacian schedule benchmarks.
- `scripts/run_pinn_schedule_benchmark.py`: CLI for PINN schedule benchmarks.
- `scripts/run_workload_benchmark.py`: CLI for benchmarking derivative workload runtimes.

## Notes

- The MLP Laplacian and Poisson PINN workloads are synthetic. They are meant to exercise higher-order derivative tracing before adding larger real kernels.
- Forward-over-reverse currently uses a dense tangent basis and is mainly for smaller models/prototyping.
