#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adscheduler.laplacian_benchmark import (  # noqa: E402
    LaplacianBenchmarkConfig,
    available_laplacian_schedule_names,
)
from adscheduler.pinn_benchmark import (  # noqa: E402
    PINNBenchmarkConfig,
    available_pinn_schedule_names,
)
from adscheduler.stablehlo_passes import (  # noqa: E402
    StableHLOProgram,
    _make_laplacian_lowering_args,
    _make_pinn_lowering_args,
    available_stablehlo_pass_names,
    default_stablehlo_pass_names,
    lower_laplacian_schedule_to_stablehlo,
    lower_pinn_schedule_to_stablehlo,
    lower_to_stablehlo,
    run_stablehlo_transform_pipeline,
)
from adscheduler.workloads import (  # noqa: E402
    available_derivative_workload_names,
    make_derivative_workload,
)


@dataclass(frozen=True)
class IRTarget:
    label: str
    program: StableHLOProgram
    source_args: tuple


def parse_args() -> argparse.Namespace:
    laplacian_defaults = LaplacianBenchmarkConfig(outer_steps=1)
    pinn_defaults = PINNBenchmarkConfig(outer_steps=1)

    parser = argparse.ArgumentParser(
        description="Print or write StableHLO IR for derivative benchmark workloads.",
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
            "Transform pass to run for after-pass IR. Repeat for multiple. Defaults to: "
            f"{', '.join(default_stablehlo_pass_names())}."
        ),
    )
    parser.add_argument(
        "--after-passes",
        action="store_true",
        help="Print/write the transformed StableHLO after the selected pass pipeline.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Print/write both original and after-pass StableHLO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write each StableHLO module to a .mlir file instead of printing IR to stdout.",
    )
    parser.add_argument(
        "--also-print",
        action="store_true",
        help="When --output-dir is used, also print IR to stdout.",
    )

    parser.add_argument("--laplacian-num-points", type=int, default=laplacian_defaults.num_points)
    parser.add_argument("--laplacian-input-dim", type=int, default=laplacian_defaults.input_dim)
    parser.add_argument(
        "--laplacian-hidden-layers",
        type=int,
        default=laplacian_defaults.hidden_layers,
    )
    parser.add_argument("--laplacian-hidden-dim", type=int, default=laplacian_defaults.hidden_dim)

    parser.add_argument("--pinn-grid-size", type=int, default=pinn_defaults.grid_size)
    parser.add_argument("--pinn-hidden-layers", type=int, default=pinn_defaults.hidden_layers)
    parser.add_argument("--pinn-hidden-dim", type=int, default=pinn_defaults.hidden_dim)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _selected_targets(args)
    emit_after = args.after_passes or args.both
    emit_original = not args.after_passes or args.both

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        if emit_original:
            _emit_program(
                target.program,
                label=f"{target.label} original StableHLO",
                suffix="original",
                output_dir=args.output_dir,
                also_print=args.also_print,
            )

        if emit_after:
            transformed = run_stablehlo_transform_pipeline(
                target.program,
                passes=args.passes,
                source_args=target.source_args,
            ).transformed_program
            _emit_program(
                transformed,
                label=f"{target.label} after-pass StableHLO",
                suffix="after_passes",
                output_dir=args.output_dir,
                also_print=args.also_print,
            )


def _selected_targets(args: argparse.Namespace) -> list[IRTarget]:
    targets: list[IRTarget] = []

    for workload_name in args.workloads or ():
        workload = make_derivative_workload(workload_name, seed=args.seed)
        program = lower_to_stablehlo(
            workload.name,
            workload.derivative_task,
            workload.args,
        )
        targets.append(IRTarget(workload.name, program, workload.args))

    laplacian_config = LaplacianBenchmarkConfig(
        seed=args.seed,
        outer_steps=1,
        num_points=args.laplacian_num_points,
        input_dim=args.laplacian_input_dim,
        hidden_layers=args.laplacian_hidden_layers,
        hidden_dim=args.laplacian_hidden_dim,
    )
    for schedule_name in args.laplacian_schedules or ():
        program = lower_laplacian_schedule_to_stablehlo(
            schedule_name,
            config=laplacian_config,
        )
        targets.append(
            IRTarget(
                program.name,
                program,
                _make_laplacian_lowering_args(laplacian_config),
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
        program = lower_pinn_schedule_to_stablehlo(schedule_name, config=pinn_config)
        targets.append(IRTarget(program.name, program, _make_pinn_lowering_args(pinn_config)))

    if not targets:
        workload = make_derivative_workload("mlp_laplacian_hessian", seed=args.seed)
        program = lower_to_stablehlo(
            workload.name,
            workload.derivative_task,
            workload.args,
        )
        targets.append(IRTarget(workload.name, program, workload.args))

    return targets


def _emit_program(
    program: StableHLOProgram,
    *,
    label: str,
    suffix: str,
    output_dir: Path | None,
    also_print: bool,
) -> None:
    if output_dir is None or also_print:
        print(f"// === {label} ===")
        print(program.stablehlo_text)
        print()

    if output_dir is not None:
        path = output_dir / f"{_safe_filename(program.name)}_{suffix}.mlir"
        path.write_text(program.stablehlo_text + "\n", encoding="utf-8")
        print(f"wrote {label}: {path}")


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "stablehlo"


if __name__ == "__main__":
    main()
