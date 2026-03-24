#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.ir_analysis import (
    analyze_closed_jaxpr,
    compute_feature_delta,
    format_feature_report,
)
from adscheduler.maml import (
    SCHEDULE_ROR,
    ExampleConfig,
    available_schedule_names,
    make_example_state,
    normalize_schedule_names,
    trace_maml_programs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a MAML-style program to jaxpr and analyze IR features.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--n-support", type=int, default=16)
    parser.add_argument("--n-query", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-lr", type=float, default=0.1)
    parser.add_argument("--outer-lr", type=float, default=0.01)
    parser.add_argument(
        "--schedule",
        action="append",
        choices=available_schedule_names(),
        dest="schedules",
        help="Schedule to trace/analyze. Repeat to include multiple.",
    )
    parser.add_argument(
        "--print-jaxpr",
        action="store_true",
        help="Print full jaxpr text for objective/gradient/train-step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_schedules = normalize_schedule_names(args.schedules)

    config = ExampleConfig(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_support=args.n_support,
        n_query=args.n_query,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
    )

    params, batches, _ = make_example_state(seed=args.seed, config=config)

    traced = trace_maml_programs(
        params,
        batches,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        schedule_names=selected_schedules,
    )

    objective_features = analyze_closed_jaxpr(
        traced.objective,
        known_inner_steps=args.inner_steps,
        assume_outer_grad=False,
    )

    schedule_features = {
        schedule_name: analyze_closed_jaxpr(
            traced.schedule_meta_grads[schedule_name],
            known_inner_steps=args.inner_steps,
            assume_outer_grad=True,
        )
        for schedule_name in selected_schedules
    }
    first_order_features = analyze_closed_jaxpr(
        traced.first_order_meta_grad,
        known_inner_steps=args.inner_steps,
        assume_outer_grad=True,
    )

    print("=== Objective JAXPR Features ===")
    print(format_feature_report(objective_features))
    print()

    for schedule_name in selected_schedules:
        print(f"=== {schedule_name.upper()} Meta-Grad JAXPR Features ===")
        print(format_feature_report(schedule_features[schedule_name]))
        print()

    print("=== First-Order Meta-Grad JAXPR Features ===")
    print(format_feature_report(first_order_features))
    print()

    if SCHEDULE_ROR in schedule_features:
        full_vs_first_order = compute_feature_delta(
            schedule_features[SCHEDULE_ROR],
            first_order_features,
        )
        print("=== Higher-Order Work Estimate (RoR - First-Order) ===")
        print(f"total_equations_delta: {full_vs_first_order.total_equations_delta}")
        print("primitive_histogram_delta:")
        if full_vs_first_order.primitive_histogram_delta:
            for primitive, count in full_vs_first_order.primitive_histogram_delta.items():
                print(f"  - {primitive}: {count}")
        else:
            print("  - <none>")

    if SCHEDULE_ROR in schedule_features:
        for schedule_name in selected_schedules:
            if schedule_name == SCHEDULE_ROR:
                continue
            delta = compute_feature_delta(
                schedule_features[schedule_name],
                schedule_features[SCHEDULE_ROR],
            )
            print()
            print(f"=== {schedule_name.upper()} vs ROR (Schedule - RoR) ===")
            print(f"total_equations_delta: {delta.total_equations_delta}")
            print("primitive_histogram_delta:")
            if delta.primitive_histogram_delta:
                for primitive, count in delta.primitive_histogram_delta.items():
                    print(f"  - {primitive}: {count}")
            else:
                print("  - <none>")

    if SCHEDULE_ROR not in schedule_features:
        print("RoR baseline not selected; skipping schedule-vs-RoR deltas.")

    if args.print_jaxpr:
        print()
        print("=== Objective JAXPR ===")
        print(traced.objective)
        print()
        for schedule_name in selected_schedules:
            print(f"=== {schedule_name.upper()} Meta-Grad JAXPR ===")
            print(traced.schedule_meta_grads[schedule_name])
            print()
            print(f"=== {schedule_name.upper()} Train Step JAXPR ===")
            print(traced.schedule_train_steps[schedule_name])
            print()


if __name__ == "__main__":
    main()
