# adscheduler

`adscheduler` is a small JAX meta-learning prototype for experimenting with AD schedule selection.
It traces MAML-style workloads to closed `jaxpr`s, extracts IR features, and benchmarks multiple higher-order differentiation schedules.

Current schedule set:
`ror`, `for`, `jacrev`, `jacfwd`, and rematerialized variants (`*_remat`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib numpy
```

## Quick Run

Trace and inspect IR:

```bash
python3 scripts/run_trace_analysis.py --inner-steps 5
python3 scripts/run_trace_analysis.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd
```

Add `--print-jaxpr` to dump full IR text.

Run fixed schedule benchmarks:

```bash
python3 scripts/run_schedule_benchmark.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd \
  --schedule ror_remat --schedule for_remat --schedule jacrev_remat --schedule jacfwd_remat \
  --outer-steps 100 \
  --eval-every 10 \
  --target-accuracy 0.85 \
  --meta-batch-size 4 \
  --meta-test-tasks 64
```

Run with warmup-based auto selection:

```bash
python3 scripts/run_schedule_benchmark.py \
  --include-auto \
  --warmup-steps 3 \
  --warmup-loss-tolerance 0.10 \
  --max-params-for-forward-like 50000 \
  --memory-budget-mb 4096
```

## What the Benchmark Reports

- Statistical metrics: final/best meta-test accuracy, iterations to target accuracy.
- Runtime metrics: compile overhead, avg/p50/p90 outer-step time.
- Memory metrics: peak host memory and device memory when available.
- IR metrics: equation count, loop nesting, and higher-order-site counts.
- Auto-selection metrics: chosen schedule and runtime regret versus the oracle fixed baseline.

## How Auto Selection Works

The selector runs a short warmup for each candidate schedule, records compile + runtime behavior, applies simple guards (for example parameter-count gating for forward-like schedules), and picks the best candidate under a horizon-aware score.

This is exploratory. It is useful for validating that schedule choice materially affects compile/runtime behavior before adding a stronger static cost model or learned policy.

## Repository Map

- `src/adscheduler/maml.py`: synthetic MAML objective, schedule registry, and tracing helpers.
- `src/adscheduler/ir_analysis.py`: recursive `jaxpr` feature extraction.
- `src/adscheduler/benchmark.py`: fixed and auto schedule evaluation.
- `scripts/run_trace_analysis.py`: CLI for tracing + IR reports.
- `scripts/run_schedule_benchmark.py`: CLI for benchmark and auto-selector runs.

## Notes

- The task generator is synthetic and intentionally simple. We are looking to expand the scope of this prototype.
- Forward-over-reverse currently uses a dense tangent basis and is mainly for smaller models/prototyping.
