from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Any

LOOP_PRIMITIVES = {"scan", "while", "fori_loop"}
HIGHER_ORDER_PRIMITIVE_HINTS = {
    "custom_jvp_call",
    "custom_vjp_call",
    "jvp",
    "linear_call",
    "transpose_call",
    "stop_gradient",
}


@dataclass(frozen=True)
class JaxprSite:
    path: str
    primitive: str
    source: str
    trip_count_hint: int | None


@dataclass(frozen=True)
class JaxprFeatures:
    total_equations: int
    primitive_histogram: dict[str, int]
    max_call_depth: int
    max_loop_nesting: int
    loop_sites: list[JaxprSite]
    loop_trip_count_hints: list[int]
    tensor_aval_histogram: dict[str, int]
    higher_order_sites: list[JaxprSite]
    known_inner_steps: int | None


@dataclass(frozen=True)
class FeatureDelta:
    total_equations_delta: int
    primitive_histogram_delta: dict[str, int]


def analyze_closed_jaxpr(
    closed_jaxpr: Any,
    *,
    known_inner_steps: int | None = None,
    assume_outer_grad: bool = False,
) -> JaxprFeatures:
    primitive_hist = Counter()
    tensor_aval_hist = Counter()
    loop_sites: list[JaxprSite] = []
    loop_trip_count_hints: list[int] = []
    higher_order_sites: list[JaxprSite] = []

    max_call_depth = 0
    max_loop_nesting = 0
    total_equations = 0

    def walk(jaxpr: Any, call_depth: int, loop_depth: int, path_prefix: str) -> None:
        nonlocal max_call_depth, max_loop_nesting, total_equations

        max_call_depth = max(max_call_depth, call_depth)

        for i, eqn in enumerate(getattr(jaxpr, "eqns", [])):
            total_equations += 1
            primitive = eqn.primitive.name
            primitive_hist[primitive] += 1

            eqn_path = f"{path_prefix}/{primitive}[{i}]"
            source = _source_info_string(eqn)

            _record_var_avals(tensor_aval_hist, getattr(eqn, "invars", ()))
            _record_var_avals(tensor_aval_hist, getattr(eqn, "outvars", ()))

            current_loop_depth = loop_depth
            if primitive in LOOP_PRIMITIVES:
                current_loop_depth += 1
                max_loop_nesting = max(max_loop_nesting, current_loop_depth)
                trip_count_hint = _loop_trip_count_hint(eqn)
                if trip_count_hint is not None:
                    loop_trip_count_hints.append(trip_count_hint)
                loop_sites.append(
                    JaxprSite(
                        eqn_path,
                        primitive,
                        source,
                        trip_count_hint=trip_count_hint,
                    )
                )

            if _looks_higher_order(primitive, source, assume_outer_grad=assume_outer_grad):
                higher_order_sites.append(
                    JaxprSite(
                        eqn_path,
                        primitive,
                        source,
                        trip_count_hint=None,
                    )
                )

            for child_idx, child in enumerate(_extract_nested_jaxprs(eqn.params)):
                child_jaxpr = getattr(child, "jaxpr", child)
                walk(
                    child_jaxpr,
                    call_depth=call_depth + 1,
                    loop_depth=current_loop_depth,
                    path_prefix=f"{eqn_path}.sub{child_idx}",
                )

    _record_avals(tensor_aval_hist, getattr(closed_jaxpr, "in_avals", ()))
    _record_avals(tensor_aval_hist, getattr(closed_jaxpr, "out_avals", ()))

    walk(
        getattr(closed_jaxpr, "jaxpr", closed_jaxpr),
        call_depth=0,
        loop_depth=0,
        path_prefix="root",
    )

    return JaxprFeatures(
        total_equations=total_equations,
        primitive_histogram=dict(sorted(primitive_hist.items())),
        max_call_depth=max_call_depth,
        max_loop_nesting=max_loop_nesting,
        loop_sites=loop_sites,
        loop_trip_count_hints=sorted(loop_trip_count_hints),
        tensor_aval_histogram=dict(sorted(tensor_aval_hist.items())),
        higher_order_sites=higher_order_sites,
        known_inner_steps=known_inner_steps,
    )


def compute_feature_delta(
    lhs: JaxprFeatures,
    rhs: JaxprFeatures,
) -> FeatureDelta:
    primitive_delta: dict[str, int] = {}

    keys = set(lhs.primitive_histogram) | set(rhs.primitive_histogram)
    for primitive in sorted(keys):
        diff = lhs.primitive_histogram.get(primitive, 0) - rhs.primitive_histogram.get(primitive, 0)
        if diff:
            primitive_delta[primitive] = diff

    return FeatureDelta(
        total_equations_delta=lhs.total_equations - rhs.total_equations,
        primitive_histogram_delta=primitive_delta,
    )


def format_feature_report(features: JaxprFeatures) -> str:
    lines: list[str] = []

    lines.append(f"total_equations: {features.total_equations}")
    lines.append(f"max_call_depth: {features.max_call_depth}")
    lines.append(f"max_loop_nesting: {features.max_loop_nesting}")
    lines.append(f"loop_trip_count_hints: {features.loop_trip_count_hints}")
    if features.known_inner_steps is not None:
        lines.append(f"known_inner_steps: {features.known_inner_steps}")

    lines.append("primitive_histogram:")
    for primitive, count in features.primitive_histogram.items():
        lines.append(f"  - {primitive}: {count}")

    lines.append("tensor_aval_histogram:")
    for aval, count in features.tensor_aval_histogram.items():
        lines.append(f"  - {aval}: {count}")

    lines.append("loop_sites:")
    if features.loop_sites:
        for site in features.loop_sites:
            lines.append(
                "  - "
                f"path={site.path} primitive={site.primitive} "
                f"trip_count_hint={site.trip_count_hint} source={site.source}"
            )
    else:
        lines.append("  - <none>")

    lines.append("higher_order_sites:")
    if features.higher_order_sites:
        for site in features.higher_order_sites:
            lines.append(
                "  - "
                f"path={site.path} primitive={site.primitive} source={site.source}"
            )
    else:
        lines.append("  - <none>")

    return "\n".join(lines)


def _record_var_avals(hist: Counter[str], variables: Any) -> None:
    for var in variables:
        aval = getattr(var, "aval", None)
        if aval is not None:
            hist[_format_aval(aval)] += 1


def _record_avals(hist: Counter[str], avals: Any) -> None:
    for aval in avals:
        hist[_format_aval(aval)] += 1


def _format_aval(aval: Any) -> str:
    shape = getattr(aval, "shape", None)
    dtype = getattr(aval, "dtype", None)

    if shape is None:
        return str(aval)

    dim_str = ",".join(str(dim) for dim in shape)
    return f"shape=({dim_str}) dtype={dtype}"


def _extract_nested_jaxprs(value: Any) -> list[Any]:
    nested: list[Any] = []

    def recurse(obj: Any) -> None:
        if _is_jaxpr_like(obj):
            nested.append(obj)
            return

        if isinstance(obj, dict):
            for item in obj.values():
                recurse(item)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                recurse(item)

    recurse(value)
    return nested


def _is_jaxpr_like(obj: Any) -> bool:
    return hasattr(obj, "jaxpr") or hasattr(obj, "eqns")


def _source_info_string(eqn: Any) -> str:
    source_info = getattr(eqn, "source_info", None)
    if source_info is None:
        return "<unknown>"

    name_stack = getattr(source_info, "name_stack", None)
    if name_stack is not None:
        rendered = str(name_stack)
        if rendered:
            return rendered

    rendered = str(source_info)
    return rendered if rendered else "<unknown>"


def _looks_higher_order(
    primitive: str,
    source: str,
    *,
    assume_outer_grad: bool,
) -> bool:
    if primitive in HIGHER_ORDER_PRIMITIVE_HINTS:
        return True

    source_lower = source.lower()
    if "vjp" in source_lower or "jvp" in source_lower:
        return True

    if assume_outer_grad and "inner_grad" in source_lower:
        return True

    return False


def _loop_trip_count_hint(eqn: Any) -> int | None:
    if eqn.primitive.name == "scan":
        length = eqn.params.get("length")
        return int(length) if isinstance(length, Integral) else None

    if eqn.primitive.name == "fori_loop":
        start = eqn.params.get("lower")
        stop = eqn.params.get("upper")
        if isinstance(start, Integral) and isinstance(stop, Integral):
            return stop - start

    return None
