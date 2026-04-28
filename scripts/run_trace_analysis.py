#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.ir_analysis import analyze_closed_jaxpr, format_feature_report
from adscheduler.workloads import (
    available_derivative_workload_names,
    normalize_derivative_workload_names,
    trace_derivative_workloads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace JAX derivative workloads to jaxpr and analyze IR features.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workload",
        action="append",
        choices=available_derivative_workload_names(),
        dest="workloads",
        help="Derivative workload to trace. Repeat for multiple.",
    )
    parser.add_argument(
        "--print-jaxpr",
        action="store_true",
        help="Print full jaxpr text for each selected workload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_workloads = normalize_derivative_workload_names(args.workloads)
    traced_workloads = trace_derivative_workloads(
        selected_workloads,
        seed=args.seed,
    )

    for workload_name in selected_workloads:
        workload = traced_workloads[workload_name]
        features = analyze_closed_jaxpr(
            workload.closed_jaxpr,
            known_inner_steps=workload.known_loop_steps,
            assume_outer_grad=workload.assume_outer_grad,
        )
        print(f"=== {workload.name.upper()} Derivative Workload Features ===")
        print(f"description: {workload.description}")
        print(format_feature_report(features))
        print()

        if args.print_jaxpr:
            print(f"=== {workload.name.upper()} JAXPR ===")
            print(workload.closed_jaxpr)
            print()


if __name__ == "__main__":
    main()
