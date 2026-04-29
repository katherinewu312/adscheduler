from __future__ import annotations

import time
from dataclasses import dataclass
import re
from typing import Any, Callable, Sequence

import numpy as np

from adscheduler.stablehlo_passes import StableHLOProgram


class StableHLOExecutionUnavailable(RuntimeError):
    """Raised when no executable StableHLO runtime backend is available."""


@dataclass(frozen=True)
class StableHLOCompiledExecutable:
    program_name: str
    function_name: str
    backend: str
    compile_overhead_sec: float
    callable: Callable[..., Any]

    # New: prepare runtime args once, outside the timed loop.
    prepare_args: Callable[..., Sequence[Any]] | None = None

    # New: execute already-prepared runtime args inside the timed loop.
    call_prepared: Callable[[Sequence[Any]], Any] | None = None


@dataclass(frozen=True)
class _TensorArgSpec:
    shape: tuple[int, ...]
    dtype: np.dtype


def compile_stablehlo_with_iree(
    program: StableHLOProgram,
    *,
    function_name: str = "main",
    target_backend: str = "llvm-cpu",
    target_cpu: str = "generic",
    target_cpu_features: str = "",
    driver: str = "local-task",
) -> StableHLOCompiledExecutable:
    """Compile StableHLO text into an executable callable using IREE.

    This is intentionally optional: JAX can lower to StableHLO, but executing
    edited StableHLO text requires a separate compiler/runtime stack. IREE gives
    us a Python-accessible path from StableHLO MLIR text to a CPU executable.
    """

    try:
        import iree.compiler as iree_compiler
        import iree.runtime as iree_runtime
    except ImportError as exc:
        raise StableHLOExecutionUnavailable(
            "IREE is required to execute transformed StableHLO text. "
            "Install the optional runtime dependencies with "
            "`pip install iree-compiler iree-runtime`."
        ) from exc

    compile_start = time.perf_counter()
    extra_args = _iree_compile_extra_args(
        target_backend=target_backend,
        target_cpu=target_cpu,
        target_cpu_features=target_cpu_features,
    )
    try:
        vm_flatbuffer = iree_compiler.compile_str(
            program.stablehlo_text,
            input_type="stablehlo",
            target_backends=[target_backend],
            extra_args=extra_args,
        )
    except Exception as exc:
        raise StableHLOExecutionUnavailable(
            f"IREE could not compile transformed StableHLO for {program.name}: {exc}"
        ) from exc

    try:
        config = iree_runtime.Config(driver)
        vm_module = iree_runtime.VmModule.copy_buffer(
            config.vm_instance,
            vm_flatbuffer,
        )
        context = iree_runtime.SystemContext(config=config)
        if hasattr(context, "add_vm_module"):
            context.add_vm_module(vm_module)
        else:
            context.add_vm_modules(vm_module)
        module = _runtime_module_from_context(context, vm_module)
        runtime_fn = getattr(module, function_name)
    except Exception as exc:
        raise StableHLOExecutionUnavailable(
            f"IREE compiled {program.name}, but the runtime function "
            f"`{function_name}` could not be loaded: {exc}"
        ) from exc

    compile_overhead_sec = time.perf_counter() - compile_start
    input_specs = _parse_function_input_specs(program.stablehlo_text, function_name)

    def call_transformed_stablehlo(*args: Any) -> Any:
        flat_args = [_as_numpy_array(arg) for arg in _flatten_args(args)]
        runtime_args = _select_runtime_args(flat_args, input_specs)
        return _normalize_runtime_output(runtime_fn(*runtime_args))

    return StableHLOCompiledExecutable(
        program_name=program.name,
        function_name=function_name,
        backend=f"iree:{target_backend}/{driver}/cpu={target_cpu}",
        compile_overhead_sec=compile_overhead_sec,
        callable=call_transformed_stablehlo,
    )


def compile_stablehlo_with_xla(
    program: StableHLOProgram,
    *,
    platform: str | None = None,
) -> StableHLOCompiledExecutable:
    """Compile StableHLO text with XLA/PJRT when the local JAX build exposes it.

    JAX always sends lowered StableHLO through XLA for normal `jax.jit` code.
    Re-entering XLA from edited StableHLO text is less stable across jaxlib
    releases, so this function probes the private bridge APIs used by jaxlib
    builds that expose MLIR-to-XLA conversion. Callers should surface the
    unavailable error rather than silently claiming transformed IR was run.
    """

    compile_start = time.perf_counter()
    try:
        import jax
        from jaxlib import xla_client
    except ImportError as exc:
        raise StableHLOExecutionUnavailable(
            "JAX and jaxlib are required to execute transformed StableHLO via XLA/PJRT."
        ) from exc

    try:
        devices = jax.devices(platform) if platform else jax.devices()
        backend = devices[0].client
        executable_devices = xla_client.DeviceList(tuple(devices))
        executable = _compile_stablehlo_text(
            backend,
            program.stablehlo_text,
            executable_devices,
            xla_client.CompileOptions(),
        )
    except Exception as exc:
        raise StableHLOExecutionUnavailable(
            f"XLA/PJRT could not compile transformed StableHLO for {program.name}: {exc}"
        ) from exc
    
    input_specs = _parse_function_input_specs(program.stablehlo_text, "main")

    def prepare_transformed_args(*args: Any) -> list[Any]:
        # This is the expensive Python-side argument setup.
        # Do it once before benchmarking, not once per timed iteration.
        flat_args = [_as_numpy_array(arg) for arg in _flatten_args(args)]
        runtime_args = _select_runtime_args(flat_args, input_specs)
        return [jax.device_put(arg, devices[0]) for arg in runtime_args]

    def call_prepared_transformed_stablehlo(runtime_args: Sequence[Any]) -> Any:
        # This is the actual executable call.
        # Do not normalize to NumPy here; that can force host materialization.
        runtime_args_list = (
            runtime_args if isinstance(runtime_args, list) else list(runtime_args)
        )

        try:
            return executable.execute(runtime_args_list)
        except AttributeError:
            return executable.execute_sharded_on_local_devices([runtime_args_list])

    def call_transformed_stablehlo(*args: Any) -> Any:
        # Keep the old public behavior for compatibility.
        runtime_args = prepare_transformed_args(*args)
        output = call_prepared_transformed_stablehlo(runtime_args)
        return _normalize_runtime_output(output)

    return StableHLOCompiledExecutable(
        program_name=program.name,
        function_name="main",
        backend=f"xla:{platform or jax.default_backend()}",
        compile_overhead_sec=time.perf_counter() - compile_start,
        callable=call_transformed_stablehlo,
        prepare_args=prepare_transformed_args,
        call_prepared=call_prepared_transformed_stablehlo,
    )

def _compile_stablehlo_text(
    backend: Any,
    stablehlo_text: str,
    devices: Sequence[Any],
    compile_options: Any,
) -> Any:
    # Newer jaxlib/PJRT builds can compile StableHLO text directly through
    # compile_and_load. The older private MLIR bridge returned XlaComputation
    # objects that current PJRT clients no longer accept.
    attempts = (
        lambda: backend.compile_and_load(stablehlo_text, devices, compile_options),
        lambda: backend.compile(stablehlo_text, devices, compile_options),
        lambda: backend.compile(stablehlo_text, devices),
    )
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _stablehlo_text_to_xla_computation(stablehlo_text: str, xla_client: Any) -> Any:
    candidates = []
    xla_extension = getattr(xla_client, "_xla", None)
    if xla_extension is not None:
        candidates.append(getattr(xla_extension, "mlir", None))
        candidates.append(xla_extension)
    candidates.append(xla_client)

    function_names = (
        "mlir_module_to_xla_computation",
        "mlir_to_xla_computation",
        "MlirToXlaComputation",
    )
    last_error: Exception | None = None
    encoded = stablehlo_text.encode("utf-8")
    for candidate in candidates:
        if candidate is None:
            continue
        for function_name in function_names:
            converter = getattr(candidate, function_name, None)
            if converter is None:
                continue
            for payload in (stablehlo_text, encoded):
                try:
                    return converter(payload)
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:
                    last_error = exc
                    break

    message = (
        "This jaxlib build does not expose a StableHLO/MLIR text to XLA "
        "computation entry point."
    )
    if last_error is not None:
        message += f" Last converter error: {last_error}"
    raise StableHLOExecutionUnavailable(message)


def _iree_compile_extra_args(
    *,
    target_backend: str,
    target_cpu: str,
    target_cpu_features: str,
) -> list[str]:
    if target_backend != "llvm-cpu":
        return []
    return [
        f"--iree-llvmcpu-target-cpu={target_cpu}",
        f"--iree-llvmcpu-target-cpu-features={target_cpu_features}",
    ]


def _runtime_module_from_context(context: Any, vm_module: Any) -> Any:
    module_names = [getattr(vm_module, "name", ""), "module"]
    for module_name in module_names:
        if not module_name:
            continue
        try:
            module = getattr(context.modules, module_name)
        except AttributeError:
            module = None
        if module is not None and not callable(module):
            return module
        try:
            module = context.modules[module_name]
        except (KeyError, TypeError, AttributeError):
            module = None
        if module is not None and not callable(module):
            return module

    modules = [
        getattr(context.modules, name)
        for name in dir(context.modules)
        if not name.startswith("_")
    ]
    user_modules = [
        module
        for module in modules
        if not callable(module)
        and (
            hasattr(module, "main")
            or not module.__class__.__name__.lower().startswith("hal")
        )
    ]
    if not user_modules:
        raise StableHLOExecutionUnavailable(
            f"No executable IREE module was registered. VM module name: "
            f"{getattr(vm_module, 'name', '<unknown>')}"
        )
    for module in user_modules:
        if hasattr(module, "main"):
            return module
    return user_modules[0]


def _flatten_args(value: Any) -> list[Any]:
    if isinstance(value, dict):
        flat: list[Any] = []
        for key in sorted(value):
            flat.extend(_flatten_args(value[key]))
        return flat
    if isinstance(value, (tuple, list)):
        flat = []
        for item in value:
            flat.extend(_flatten_args(item))
        return flat
    return [value]


def _parse_function_input_specs(
    stablehlo_text: str,
    function_name: str,
) -> tuple[_TensorArgSpec, ...]:
    args_text = _function_args_text(stablehlo_text, function_name)
    if args_text is None:
        return ()

    specs: list[_TensorArgSpec] = []
    for arg_text in _split_top_level_commas(args_text):
        type_match = re.search(r"tensor<([^>]*)>", arg_text)
        if not type_match:
            continue
        spec = _parse_tensor_type(type_match.group(1))
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _function_args_text(stablehlo_text: str, function_name: str) -> str | None:
    name_pattern = re.compile(
        r"(?:\s+\w+)*\s+@" + re.escape(function_name) + r"\s*\("
    )
    for func_match in re.finditer(r"\bfunc\.func\b", stablehlo_text):
        name_match = name_pattern.match(stablehlo_text, func_match.end())
        if name_match is None:
            continue
        return _parenthesized_region_from(stablehlo_text, name_match.end() - 1)
    return None


def _parenthesized_region_from(text: str, open_index: int) -> str | None:
    if open_index < 0 or open_index >= len(text) or text[open_index] != "(":
        return None
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index]
    return None


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(0, angle_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "," and angle_depth == 0 and paren_depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_tensor_type(type_text: str) -> _TensorArgSpec | None:
    pieces = [piece for piece in type_text.split("x") if piece]
    if not pieces:
        return None
    dtype = _numpy_dtype_for_stablehlo(pieces[-1])
    if dtype is None:
        return None
    shape: list[int] = []
    for dim in pieces[:-1]:
        if dim == "?":
            return None
        try:
            shape.append(int(dim))
        except ValueError:
            return None
    return _TensorArgSpec(shape=tuple(shape), dtype=dtype)


def _numpy_dtype_for_stablehlo(type_name: str) -> np.dtype | None:
    mapping = {
        "bf16": np.dtype(np.float32),
        "f16": np.dtype(np.float16),
        "f32": np.dtype(np.float32),
        "f64": np.dtype(np.float64),
        "i1": np.dtype(np.bool_),
        "i8": np.dtype(np.int8),
        "i16": np.dtype(np.int16),
        "i32": np.dtype(np.int32),
        "i64": np.dtype(np.int64),
        "ui8": np.dtype(np.uint8),
        "ui16": np.dtype(np.uint16),
        "ui32": np.dtype(np.uint32),
        "ui64": np.dtype(np.uint64),
    }
    return mapping.get(type_name)


def _select_runtime_args(
    flat_args: list[np.ndarray],
    input_specs: tuple[_TensorArgSpec, ...],
) -> list[np.ndarray]:
    if not input_specs:
        if flat_args:
            raise StableHLOExecutionUnavailable(
                "Could not parse the transformed StableHLO @main input signature; "
                f"refusing to pass all {len(flat_args)} Python leaves blindly."
            )
        return []
    if len(flat_args) == len(input_specs):
        return flat_args

    selected: list[np.ndarray] = []
    search_index = 0
    for spec in input_specs:
        matched_index = _find_matching_arg(flat_args, spec, search_index)
        if matched_index is None:
            raise StableHLOExecutionUnavailable(
                "Could not align Python arguments to transformed StableHLO "
                f"signature. Expected tensor shape={spec.shape} dtype={spec.dtype}; "
                f"remaining Python leaves={_format_arg_specs(flat_args[search_index:])}."
            )
        selected.append(flat_args[matched_index])
        search_index = matched_index + 1

    return selected


def _find_matching_arg(
    flat_args: list[np.ndarray],
    spec: _TensorArgSpec,
    start_index: int,
) -> int | None:
    for index in range(start_index, len(flat_args)):
        arg = flat_args[index]
        if tuple(arg.shape) == spec.shape and np.dtype(arg.dtype) == spec.dtype:
            return index
    return None


def _format_arg_specs(args: list[np.ndarray]) -> str:
    return ", ".join(f"{tuple(arg.shape)}:{arg.dtype}" for arg in args[:10])


def _as_numpy_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _normalize_runtime_output(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_normalize_runtime_output(item) for item in value)
    if isinstance(value, list):
        return [_normalize_runtime_output(item) for item in value]
    if hasattr(value, "to_host"):
        return value.to_host()
    return np.asarray(value)
