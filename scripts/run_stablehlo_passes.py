#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.laplacian_benchmark import (
    LaplacianBenchmarkConfig,
    available_laplacian_schedule_names,
)
from adscheduler.pinn_benchmark import (
    PINNBenchmarkConfig,
    available_pinn_schedule_names,
)
from adscheduler.stablehlo_passes import (
    available_stablehlo_pass_names,
    default_stablehlo_pass_names,
    format_stablehlo_pipeline_report,
    lower_laplacian_schedule_to_stablehlo,
    lower_pinn_schedule_to_stablehlo,
    lower_workload_to_stablehlo,
    run_stablehlo_pass_pipeline,
    score_stablehlo_optimization_surface,
)
from adscheduler.workloads import available_derivative_workload_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lower JAX derivative workloads to StableHLO and run IR analysis passes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workload",
        action="append",
        choices=available_derivative_workload_names(),
        dest="workloads",
        help="Named derivative workload to lower. Repeat for multiple.",
    )
    parser.add_argument(
        "--laplacian-schedule",
        action="append",
        choices=available_laplacian_schedule_names(),
        dest="laplacian_schedules",
        help="Laplacian schedule to lower. Repeat for multiple.",
    )
    parser.add_argument(
        "--pinn-schedule",
        action="append",
        choices=available_pinn_schedule_names(),
        dest="pinn_schedules",
        help="PINN schedule to lower. Repeat for multiple.",
    )
    parser.add_argument(
        "--pass",
        action="append",
        choices=available_stablehlo_pass_names(),
        dest="passes",
        help=(
            "StableHLO pass to run. Repeat for multiple. Defaults to: "
            f"{', '.join(default_stablehlo_pass_names())}."
        ),
    )
    parser.add_argument(
        "--print-stablehlo",
        action="store_true",
        help="Print the StableHLO module text before the pass report.",
    )

    parser.add_argument("--laplacian-num-points", type=int, default=64)
    parser.add_argument("--laplacian-input-dim", type=int, default=3)
    parser.add_argument("--laplacian-hidden-layers", type=int, default=64)
    parser.add_argument("--laplacian-hidden-dim", type=int, default=128)

    parser.add_argument("--pinn-grid-size", type=int, default=10)
    parser.add_argument("--pinn-hidden-layers", type=int, default=6)
    parser.add_argument("--pinn-hidden-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    programs = []

    for workload_name in args.workloads or ():
        programs.append(lower_workload_to_stablehlo(workload_name, seed=args.seed))

    laplacian_config = LaplacianBenchmarkConfig(
        seed=args.seed,
        outer_steps=1,
        num_points=args.laplacian_num_points,
        input_dim=args.laplacian_input_dim,
        hidden_layers=args.laplacian_hidden_layers,
        hidden_dim=args.laplacian_hidden_dim,
    )
    for schedule_name in args.laplacian_schedules or ():
        programs.append(
            lower_laplacian_schedule_to_stablehlo(
                schedule_name,
                config=laplacian_config,
            )
        )

    pinn_config = PINNBenchmarkConfig(
        seed=args.seed,
        outer_steps=1,
        grid_size=args.pinn_grid_size,
        hidden_layers=args.pinn_hidden_layers,
        hidden_dim=args.pinn_hidden_dim,
    )
    for schedule_name in args.pinn_schedules or ():
        programs.append(lower_pinn_schedule_to_stablehlo(schedule_name, config=pinn_config))

    if not programs:
        programs.append(lower_workload_to_stablehlo("mlp_laplacian_hessian", seed=args.seed))

    for program in programs:
        if args.print_stablehlo:
            print(f"=== {program.name} StableHLO ===")
            print(program.stablehlo_text)
            print()

        result = run_stablehlo_pass_pipeline(program, passes=args.passes)
        score = score_stablehlo_optimization_surface(result)
        print(f"=== {program.name} StableHLO Pass Report ===")
        print(format_stablehlo_pipeline_report(result))
        print("compiler_score:")
        print(f"  score: {score.score:.3f}")
        print(f"  estimated_optimized_operations: {score.estimated_optimized_operations:.3f}")
        print(f"  laplacian_recurrence_rewrites: {score.laplacian_recurrence_rewrites}")
        print(f"  constant_foldable_operations: {score.constant_foldable_operations}")
        print(f"  mixed_partial_cse_rewrites: {score.mixed_partial_cse_rewrites}")
        print(f"  symmetric_kernel_rewrites: {score.symmetric_kernel_rewrites}")
        print()


if __name__ == "__main__":
    main()
