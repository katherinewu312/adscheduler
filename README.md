# AD Scheduler Template (JAX IR)

This starter template focuses on your first two milestones:

1. trace a MAML-style meta-training program to `jaxpr`,
2. analyze the IR for loop structure, tensor shapes, and higher-order AD signals.

## Project Layout

- `src/adscheduler/maml.py`: synthetic MAML-style objective and meta-train step, plus `jaxpr` tracing helpers.
- `src/adscheduler/ir_analysis.py`: recursive `jaxpr` feature extraction utilities.
- `scripts/run_trace_analysis.py`: CLI that traces programs and prints analysis reports.
- `src/adscheduler/benchmark.py`: end-to-end schedule benchmarking (statistical + hardware + IR metrics).
- `scripts/run_schedule_benchmark.py`: CLI for full schedule evaluation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install jax jaxlib numpy
python scripts/run_trace_analysis.py --inner-steps 5
python scripts/run_trace_analysis.py --schedule ror --schedule for --schedule jacrev --schedule jacfwd
python scripts/run_schedule_benchmark.py --outer-steps 50 --eval-every 5 --target-accuracy 0.85
python scripts/run_schedule_benchmark.py --include-auto --warmup-steps 3
```

Add `--print-jaxpr` if you want to dump the full text IR.

## How This Maps to Your Two Steps

### Step 1: Trace a MAML-style training step to `jaxpr`

`trace_maml_programs(...)` in `maml.py` captures closed jaxprs for these schedules/programs:

- objective (`maml_meta_objective`)
- reverse-over-reverse meta-gradient (`ror`)
- forward-over-reverse meta-gradient (`for`)
- jacobian-based reverse-mode variant (`jacrev`)
- jacobian-based forward-mode variant (`jacfwd`)
- rematerialized/checkpointed versions (`ror_remat`, `for_remat`, `jacrev_remat`, `jacfwd_remat`)
- first-order MAML meta-gradient (inner update uses `stop_gradient`)

Schedule selection is registry-driven via `available_schedule_names()`,
`get_meta_grad_schedule(...)`, and `meta_train_step_with_schedule(...)`.

This gives you a direct handle for downstream compiler passes.

### Step 2: Analyze IR features

`analyze_closed_jaxpr(...)` in `ir_analysis.py` extracts:

- `primitive_histogram`
- `max_call_depth`
- `max_loop_nesting`, `loop_trip_count_hints`, and `loop_sites`
- `tensor_aval_histogram` (shape + dtype)
- `higher_order_sites` (heuristic markers for higher-order AD)

`compute_feature_delta(full, first_order)` compares full-vs-first-order gradient programs to estimate higher-order derivative overhead.
The CLI reports per-schedule deltas against RoR, so you can inspect schedule-level IR differences.

## Typical CLI Usage

```bash
python3 scripts/run_trace_analysis.py \
  --seed 0 \
  --in-dim 16 \
  --hidden-dim 64 \
  --n-support 32 \
  --n-query 32 \
  --inner-steps 10
```

## End-to-End Schedule Evaluation

This project now includes a benchmark runner that ties together:

- statistical performance:
  - `final_meta_test_accuracy`
  - `best_meta_test_accuracy`
  - `outer_iterations_to_target` (for a user-specified threshold)
- hardware performance:
  - `avg/p50/p90` wall-clock time per outer step
  - `compile_overhead`
  - `peak_host_memory_mb`
  - `peak_device_memory_mb` (when backend reports it)
- auto-selection quality (optional):
  - chosen schedule after warmup
  - runtime regret vs oracle fixed baseline
- IR complexity context:
  - total equation count
  - loop nesting
  - number of detected higher-order sites

Example:

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

Warmup-based auto selection:

```bash
python3 scripts/run_schedule_benchmark.py \
  --schedule ror --schedule for --schedule jacrev --schedule jacfwd \
  --schedule ror_remat --schedule for_remat --schedule jacrev_remat --schedule jacfwd_remat \
  --include-auto \
  --warmup-steps 3 \
  --warmup-loss-tolerance 0.10 \
  --max-params-for-forward-like 50000 \
  --memory-budget-mb 4096
```

Auto mode profiles candidate schedules on a short warmup, picks one schedule,
then reports regret versus the best fixed baseline for that run.

## Notes

- The higher-order detection is a template heuristic, not a formal proof.
- The synthetic task generator is intentionally simple; replace with your real task sampling pipeline when ready.
- The output is designed to be easy to plug into a later static cost model or profiler-driven schedule selector.
- Current FoR implementation propagates a dense tangent basis (`n_params x n_params`) and is intended for small models/prototyping.
- Rematerialized schedules (`*_remat`) expose JAX checkpointing tradeoffs for memory vs recomputation experiments.
