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

`mlp_laplacian_auto` lowers the Laplacian schedule family
(`ror`, `for`, `jacrev`, `jacfwd`, and remat variants) to StableHLO, runs the
compiler analysis passes, then benchmarks the schedule with the best
compiler-surface score. The explicit `hessian`, `jvp_grad`, and `jet` workloads
also report StableHLO compiler scores, but they are not the auto candidates.

Benchmark Poisson PINN derivative strategies:

```bash
python3 scripts/run_workload_benchmark.py \
  --workload poisson_pinn_hessian \
  --workload poisson_pinn_jvp_grad \
  --workload poisson_pinn_jet \
  --workload poisson_pinn_auto \
  --warmup-runs 5 \
  --runs 100
```

`poisson_pinn_auto` uses the same compiler-pass-informed selection over the PINN
schedule family (`ror`, `for`, `jacrev`, `jacfwd`, and remat variants). The
explicit `hessian`, `jvp_grad`, and `jet` PINN workloads also report StableHLO
compiler scores separately.

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
  --input-dim 2 \
  --hidden-layers 6 \
  --hidden-dim 64 \
  --activation tanh \
  --output-dim 1 \
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

## StableHLO Passes

Lower schedules or workloads to StableHLO and run the first compiler-aware
analysis passes:

```bash
python3 scripts/run_stablehlo_passes.py \
  --workload laplacian_hessian \
  --workload laplacian_jvp_grad \
  --workload laplacian_jet
```

```bash
python3 scripts/run_stablehlo_passes.py \
  --laplacian-schedule ror \
  --laplacian-schedule for \
  --laplacian-schedule jacrev \
  --laplacian-schedule jacfwd
```

```bash
python3 scripts/run_stablehlo_passes.py \
  --pinn-schedule ror \
  --pinn-schedule for \
  --pinn-schedule jacrev \
  --pinn-schedule jacfwd
```

The current StableHLO passes are conservative analysis passes:

- `duplicate_operations`: repeated canonical operation signatures, as CSE candidates.
- `dead_results`: SSA values with no apparent textual uses.
- `elementwise_fusion_regions`: contiguous elementwise op runs that may be fusible.
- `expensive_operations`: counts ops likely to dominate runtime or block fusion.

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
- `src/adscheduler/stablehlo_passes.py`: StableHLO lowering and compiler-aware analysis passes.
- `src/adscheduler/ir_analysis.py`: recursive `jaxpr` feature extraction.
- `scripts/run_trace_analysis.py`: CLI for tracing derivative workloads and IR reports.
- `scripts/run_stablehlo_passes.py`: CLI for lowering to StableHLO and running analysis passes.
- `scripts/run_laplacian_schedule_benchmark.py`: CLI for Laplacian schedule benchmarks.
- `scripts/run_pinn_schedule_benchmark.py`: CLI for PINN schedule benchmarks.
- `scripts/run_workload_benchmark.py`: CLI for benchmarking derivative workload runtimes.

## Notes

- The MLP Laplacian and Poisson PINN workloads are synthetic. They are meant to exercise higher-order derivative tracing before adding larger real kernels.
- Forward-over-reverse currently uses a dense tangent basis and is mainly for smaller models/prototyping.
