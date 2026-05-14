#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.stablehlo_passes import (
    available_stablehlo_pass_names,
    default_stablehlo_pass_names,
    format_stablehlo_pipeline_report,
    lower_workload_to_stablehlo,
    run_stablehlo_pass_pipeline,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    programs = []

    for workload_name in args.workloads or ():
        programs.append(lower_workload_to_stablehlo(workload_name, seed=args.seed))

    if not programs:
        programs.append(lower_workload_to_stablehlo("mlp_laplacian_hessian", seed=args.seed))

    for program in programs:
        if args.print_stablehlo:
            print(f"=== {program.name} StableHLO ===")
            print(program.stablehlo_text)
            print()

        result = run_stablehlo_pass_pipeline(program, passes=args.passes)
        print(f"=== {program.name} StableHLO Pass Report ===")
        print(format_stablehlo_pipeline_report(result))
        print()


if __name__ == "__main__":
    main()
