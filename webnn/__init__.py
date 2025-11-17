"""
webnn_toy.py — a tiny WebNN-style API on top of PyTorch

This is a minimal, *toy* implementation that mimics a small slice of the Web Neural Network
(WebNN) API programming model using PyTorch tensors under the hood. It is intentionally
small and educational—not a full spec implementation and not production-ready.

Design goals
------------
- Graph-builder style API with `MLContext`, `MLGraphBuilder`, `MLOperand`, and `MLGraph`.
- A handful of common ops: `add`, `matmul`, `relu`, `sigmoid`, `tanh`, `softmax`,
  `conv2d`, `maxPool2d`, `averagePool2d`, `reshape`, `transpose`, `concat`, `clamp`.
- Inputs and constants, plus a `build(outputs)` that returns a compiled `MLGraph`.
- Execution via `graph.compute({name: numpy/torch}) -> dict[name -> torch.Tensor]`.
- Simple broadcasting for elementwise binary ops (delegated to PyTorch semantics).

Limitations
-----------
- Shapes and dtypes are lightly validated; type promotion is simplified.
- Basic device auto-selection picks GPU/NPU when available; no streaming I/O or tensor views.
- Operator surface and attributes cover only a subset needed for demos.
- No quantization, layouts, or memory planning.

Usage
-----
>>> from webnn_toy import MLContext, MLGraphBuilder
>>> ctx = MLContext()
>>> builder = MLGraphBuilder(ctx)
>>> x = builder.input('x', {'dataType': 'float32', 'dimensions': [None, 4]})
>>> w = builder.constant({'dataType': 'float32', 'dimensions': [4, 3]}, torch.randn(4, 3))
>>> y = builder.matmul(x, w)
>>> y = builder.relu(y)
>>> graph = builder.build({'y': y})
>>> out = graph.compute({'x': torch.randn(2, 4)})
>>> out['y'].shape
torch.Size([2, 3])

"""

import ctypes
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("LIBTORCH_USE_PYTORCH", "1")
os.environ.setdefault("LIBTORCH_BYPASS_VERSION_CHECK", "1")


def _load_torch_shared_libs() -> None:
    if sys.platform != "darwin":
        return
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        return
    for name in (
        "libtorch_cpu.dylib",
        "libc10.dylib",
        "libtorch.dylib",
        "libtorch_python.dylib",
    ):
        candidate = lib_dir / name
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate))
            except OSError:
                pass


_load_torch_shared_libs()

try:  # pragma: no cover - optional dependency
    import pywebnn_rust as _rust_backend

    if not hasattr(_rust_backend, "execute"):  # type: ignore[attr-defined]
        raise ImportError("pywebnn_rust missing execute")
    _HAS_RUST_BACKEND = True
except ImportError:  # pragma: no cover - optional dependency
    _rust_backend = None
    _HAS_RUST_BACKEND = False


TensorLike = Union[torch.Tensor, np.ndarray, float, int]


def _to_tensor(
    x: TensorLike,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (float, int)):
        t = torch.tensor(x)
    else:
        raise TypeError(f"Unsupported tensor-like type: {type(x)}")
    if dtype is not None and device is not None:
        t = t.to(dtype=dtype, device=device)
    elif dtype is not None:
        t = t.to(dtype)
    elif device is not None:
        t = t.to(device)
    return t


def _dtype_from_str(dt: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint32": torch.int64,
    }
    if dt not in mapping:
        raise ValueError(f"Unsupported dataType '{dt}' in toy impl")
    return mapping[dt]


def _auto_select_device() -> torch.device:
    """Return the best available accelerator device, otherwise CPU."""

    def _is_available(name: str) -> bool:
        try:
            if name == "cuda":
                return torch.cuda.is_available()
            if name == "xpu" and hasattr(torch, "xpu"):
                return torch.xpu.is_available()
            if name == "npu" and hasattr(torch, "npu"):
                return torch.npu.is_available()
            if name == "mps" and hasattr(torch.backends, "mps"):
                backend = torch.backends.mps
                return backend.is_available() and backend.is_built()
        except Exception:
            return False
        return False

    for candidate in ("cuda", "xpu", "npu", "mps"):
        if _is_available(candidate):
            return torch.device(candidate)
    return torch.device("cpu")


@dataclass
class MLOperandDescriptor:
    dataType: str = "float32"
    dimensions: Optional[List[Optional[int]]] = None


@dataclass
class _Node:
    op: str
    inputs: List["MLOperand"]
    attrs: Dict[str, Any]


class MLOperand:
    """Represents a node in the graph.

    Each operand may be an input, constant, or the result of an op node.
    """

    def __init__(
        self,
        builder: "MLGraphBuilder",
        operand_id: int,
        name: Optional[str],
        desc: MLOperandDescriptor,
        const_value: Optional[torch.Tensor] = None,
        node: Optional[_Node] = None,
    ):
        self._builder = builder
        self.id = operand_id
        self.name = name
        self.descriptor = desc
        self.const_value = const_value
        self.node = node

    def __repr__(self) -> str:  # pragma: no cover (debug aid)
        kind = (
            "const"
            if self.const_value is not None
            else ("input" if self.node is None else "op")
        )
        return f"MLOperand(kind={kind}, name={self.name}, desc={self.descriptor})"


class MLContext:
    """Toy execution context with trivial device selection."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        backend: str = "python",
    ):
        self._destroyed = False
        self._device = self._resolve_device(device)
        backend = backend.lower()
        if backend not in ("python", "rust"):
            raise ValueError(f"Unsupported backend '{backend}'")
        if backend == "rust" and not _HAS_RUST_BACKEND:
            raise RuntimeError("Rust backend is not available. Build it with maturin.")
        self._backend = backend

    @staticmethod
    def _resolve_device(
        device_like: Optional[Union[str, torch.device]],
    ) -> torch.device:
        if device_like is None:
            return _auto_select_device()
        if isinstance(device_like, torch.device):
            return device_like
        return torch.device(device_like)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

    def destroy(self):
        self._destroyed = True


class MLGraphBuilder:
    def __init__(self, context: MLContext):
        self._ctx = context
        self._operands: List[MLOperand] = []
        self._inputs: Dict[str, MLOperand] = {}
        self._constants: List[MLOperand] = []
        self._next_operand_id = 0

    def input(self, name: str, descriptor: Dict[str, Any]) -> MLOperand:
        desc = MLOperandDescriptor(
            dataType=descriptor.get("dataType", "float32"),
            dimensions=descriptor.get("dimensions"),
        )
        operand_id = self._allocate_operand_id()
        op = MLOperand(self, operand_id, name=name, desc=desc)
        self._operands.append(op)
        self._inputs[name] = op
        return op

    def constant(
        self,
        descriptor_or_type: Union[Dict[str, Any], str],
        buffer: Optional[TensorLike] = None,
    ) -> MLOperand:
        device = self._ctx.device
        if isinstance(descriptor_or_type, dict):
            desc = MLOperandDescriptor(
                dataType=descriptor_or_type.get("dataType", "float32"),
                dimensions=descriptor_or_type.get("dimensions"),
            )
            if buffer is None:
                raise ValueError("buffer required when passing a descriptor")
            t = _to_tensor(buffer, _dtype_from_str(desc.dataType), device=device)
        else:
            # constant(dataType, value)
            desc = MLOperandDescriptor(dataType=descriptor_or_type, dimensions=None)
            t = _to_tensor(buffer, _dtype_from_str(desc.dataType), device=device)
        operand_id = self._allocate_operand_id()
        const = MLOperand(self, operand_id, name=None, desc=desc, const_value=t)
        self._operands.append(const)
        self._constants.append(const)
        return const

    # -- helper to register op outputs ---------------------------------------
    def _emit(self, op: str, inputs: List[MLOperand], **attrs) -> MLOperand:
        node = _Node(op=op, inputs=inputs, attrs=attrs)
        # Infer a basic descriptor based on first input, but clone to avoid aliasing
        if inputs:
            first = inputs[0].descriptor
            dims = (
                list(first.dimensions)
                if first.dimensions is not None
                else None
            )
            base = MLOperandDescriptor(dataType=first.dataType, dimensions=dims)
        else:
            base = MLOperandDescriptor()
        operand_id = self._allocate_operand_id()
        out = MLOperand(self, operand_id, name=None, desc=base, node=node)
        self._operands.append(out)
        return out

    def _allocate_operand_id(self) -> int:
        oid = self._next_operand_id
        self._next_operand_id += 1
        return oid

    def add(self, a: MLOperand, b: MLOperand) -> MLOperand:
        return self._emit("add", [a, b])

    def clamp(
        self,
        x: MLOperand,
        minValue: Optional[float] = None,
        maxValue: Optional[float] = None,
    ) -> MLOperand:
        return self._emit("clamp", [x], min=minValue, max=maxValue)

    def relu(self, x: MLOperand) -> MLOperand:
        return self._emit("relu", [x])

    def sigmoid(self, x: MLOperand) -> MLOperand:
        return self._emit("sigmoid", [x])

    def tanh(self, x: MLOperand) -> MLOperand:
        return self._emit("tanh", [x])

    def softmax(self, x: MLOperand, axis: int = -1) -> MLOperand:
        return self._emit("softmax", [x], axis=axis)

    def matmul(self, a: MLOperand, b: MLOperand) -> MLOperand:
        return self._emit("matmul", [a, b])

    def batchNormalization(
        self,
        x: MLOperand,
        mean: MLOperand,
        variance: MLOperand,
        scale: Optional[MLOperand] = None,
        bias: Optional[MLOperand] = None,
        *,
        axis: int = 1,
        epsilon: float = 1e-5,
    ) -> MLOperand:
        inputs = [x, mean, variance]
        if scale is not None:
            inputs.append(scale)
        if bias is not None:
            inputs.append(bias)
        has_scale = scale is not None
        has_bias = bias is not None
        return self._emit(
            "batchNormalization",
            inputs,
            axis=axis,
            epsilon=float(epsilon),
            hasScale=has_scale,
            hasBias=has_bias,
        )

    def conv2d(
        self,
        x: MLOperand,
        w: MLOperand,
        b: Optional[MLOperand] = None,
        *,
        strides: Tuple[int, int] = (1, 1),
        dilations: Tuple[int, int] = (1, 1),
        padding: Union[str, Tuple[int, int, int, int]] = "same-upper",
        groups: int = 1,
    ) -> MLOperand:
        return self._emit(
            "conv2d",
            [x, w] + ([b] if b is not None else []),
            strides=strides,
            dilations=dilations,
            padding=padding,
            groups=groups,
        )

    def maxPool2d(
        self,
        x: MLOperand,
        *,
        windowDimensions: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        padding: Union[str, Tuple[int, int, int, int]] = "valid",
    ) -> MLOperand:
        return self._emit(
            "maxPool2d", [x], window=windowDimensions, strides=strides, padding=padding
        )

    def averagePool2d(
        self,
        x: MLOperand,
        *,
        windowDimensions: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        padding: Union[str, Tuple[int, int, int, int]] = "valid",
    ) -> MLOperand:
        return self._emit(
            "averagePool2d",
            [x],
            window=windowDimensions,
            strides=strides,
            padding=padding,
        )

    # -- tensor shape ops -----------------------------------------------------
    def reshape(self, x: MLOperand, newShape: List[int]) -> MLOperand:
        return self._emit("reshape", [x], shape=newShape)

    def transpose(
        self, x: MLOperand, permutation: Optional[List[int]] = None
    ) -> MLOperand:
        return self._emit("transpose", [x], permutation=permutation)

    def concat(self, inputs: List[MLOperand], axis: int = 0) -> MLOperand:
        return self._emit("concat", inputs, axis=axis)

    def gather(self, x: MLOperand, indices: MLOperand, *, axis: int = 0) -> MLOperand:
        out = self._emit("gather", [x, indices], axis=axis)
        out.descriptor.dimensions = _gather_output_dims(
            x.descriptor, indices.descriptor, axis
        )
        return out

    def slice(
        self,
        x: MLOperand,
        starts: List[int],
        sizes: List[int],
        strides: Optional[List[int]] = None,
    ) -> MLOperand:
        out = self._emit(
            "slice",
            [x],
            starts=list(starts),
            sizes=list(sizes),
            strides=list(strides) if strides is not None else None,
        )
        out.descriptor.dimensions = _slice_output_dims(x.descriptor, sizes, strides)
        return out

    # -- build ---------------------------------------------------------------
    def build(self, outputs: Dict[str, MLOperand]) -> "MLGraph":
        # Gather the subgraph reachable from the desired outputs
        visited: set[MLOperand] = set()
        order: List[MLOperand] = []

        def dfs(opnd: MLOperand):
            if opnd in visited:
                return
            visited.add(opnd)
            if opnd.node is not None:
                for i in opnd.node.inputs:
                    dfs(i)
                order.append(opnd)  # post-order ensures deps first

        for out in outputs.values():
            dfs(out)

        if self._ctx.backend == "rust":
            spec = self._build_rust_spec(order, outputs)
            return RustMLGraph(self._ctx, spec, self._inputs, outputs)
        return MLGraph(self._ctx, order, outputs, self._inputs)

    def _build_rust_spec(
        self, topo: List[MLOperand], outputs: Dict[str, MLOperand]
    ) -> Dict[str, Any]:
        nodes = []
        for opnd in topo:
            if opnd.node is None:
                continue
            nodes.append(
                {
                    "id": opnd.id,
                    "op": opnd.node.op,
                    "inputs": [inp.id for inp in opnd.node.inputs],
                    "attrs": _serialize_attrs(opnd.node.attrs),
                    "descriptor": _descriptor_dict(opnd.descriptor),
                }
            )
        constants = [
            {
                "id": const.id,
                "descriptor": _descriptor_dict(
                    const.descriptor, _tensor_shape(const.const_value)
                ),
                "data": _tensor_to_flat_list(const.const_value),
            }
            for const in self._constants
        ]
        inputs = [
            {
                "id": opnd.id,
                "name": name,
                "descriptor": _descriptor_dict(opnd.descriptor),
            }
            for name, opnd in self._inputs.items()
        ]
        outputs_map = {name: opnd.id for name, opnd in outputs.items()}
        return {
            "nodes": nodes,
            "constants": constants,
            "inputs": inputs,
            "outputs": outputs_map,
        }


class MLGraph:
    def __init__(
        self,
        ctx: MLContext,
        topo: List[MLOperand],
        outputs: Dict[str, MLOperand],
        inputs: Dict[str, MLOperand],
    ):
        self._ctx = ctx
        self._topo = topo
        self._outputs = outputs
        self._inputs = inputs
        self._destroyed = False

    def destroy(self):
        self._destroyed = True

    def compute(self, feeds: Dict[str, TensorLike]) -> Dict[str, torch.Tensor]:
        if self._destroyed:
            raise RuntimeError("graph destroyed")

        env: Dict[MLOperand, torch.Tensor] = {}
        device = self._ctx.device

        # Bind constants and inputs
        for opnd in self._topo:
            # We'll fill values for op nodes later
            pass
        # inputs
        for name, opnd in self._inputs.items():
            if name not in feeds:
                raise KeyError(f"Missing required input '{name}'")
            dt = _dtype_from_str(opnd.descriptor.dataType)
            env[opnd] = _to_tensor(feeds[name], dt, device=device)
        # constants anywhere in the graph may also appear in topo if referenced
        # we won't iterate all operands; resolve on-demand in _eval

        def resolve(opnd: MLOperand) -> torch.Tensor:
            if opnd in env:
                return env[opnd]
            if opnd.const_value is not None:
                env[opnd] = _to_tensor(
                    opnd.const_value,
                    _dtype_from_str(opnd.descriptor.dataType),
                    device=device,
                )
                return env[opnd]
            if opnd.node is None:
                raise RuntimeError("Operand without node/const encountered")
            # Evaluate
            vals = [resolve(i) for i in opnd.node.inputs]
            env[opnd] = _eval_node(opnd.node, vals)
            return env[opnd]

        results: Dict[str, torch.Tensor] = {}
        for name, opnd in self._outputs.items():
            results[name] = resolve(opnd)
        return results


class RustMLGraph:
    def __init__(
        self,
        ctx: MLContext,
        spec: Dict[str, Any],
        inputs: Dict[str, MLOperand],
        outputs: Dict[str, MLOperand],
    ):
        if not _HAS_RUST_BACKEND:
            raise RuntimeError("Rust backend is not available")
        self._ctx = ctx
        self._spec_json = json.dumps(spec)
        self._inputs = inputs
        self._outputs = outputs
        self._destroyed = False

    def destroy(self):
        self._destroyed = True

    def compute(self, feeds: Dict[str, TensorLike]) -> Dict[str, torch.Tensor]:
        if self._destroyed:
            raise RuntimeError("graph destroyed")
        payload = {}
        for name, opnd in self._inputs.items():
            if name not in feeds:
                raise KeyError(f"Missing required input '{name}'")
            payload[name] = _torch_to_payload(feeds[name], opnd.descriptor)
        feeds_json = json.dumps(payload)
        results_json = _rust_backend.execute(self._spec_json, feeds_json)
        raw_results = json.loads(results_json)
        device = self._ctx.device
        results: Dict[str, torch.Tensor] = {}
        for name, opnd in self._outputs.items():
            if name not in raw_results:
                raise KeyError(f"Rust backend missing output '{name}'")
            results[name] = _payload_to_torch(raw_results[name], device)
        return results


def _pad_arg(
    padding: Union[str, Tuple[int, int, int, int]],
    x: torch.Tensor,
    kernel: Tuple[int, int],
    strides: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if isinstance(padding, str):
        if padding in ("valid", "none"):
            return (0, 0, 0, 0)
        if padding in ("same-upper", "same-lower", "same"):
            # Compute SAME padding like in many frameworks; here we do symmetric pad
            h, w = x.shape[-2], x.shape[-1]
            sh, sw = strides
            kh, kw = kernel
            out_h = math.ceil(h / sh)
            out_w = math.ceil(w / sw)
            pad_h = max((out_h - 1) * sh + kh - h, 0)
            pad_w = max((out_w - 1) * sw + kw - w, 0)
            # divide into top/bottom, left/right; prefer "upper" (right/bottom) having extra pixel
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            return (left, right, top, bottom)
        raise ValueError(f"Unsupported padding string: {padding}")
    # explicit tuple l, r, t, b
    if len(padding) != 4:
        raise ValueError("padding must be 4-tuple (left, right, top, bottom)")
    return padding


def _eval_node(node: _Node, xs: List[torch.Tensor]) -> torch.Tensor:
    op = node.op
    if op == "add":
        return xs[0] + xs[1]
    if op == "clamp":
        x = xs[0]
        lo = node.attrs.get("min")
        hi = node.attrs.get("max")
        if lo is not None or hi is not None:
            if not x.is_floating_point():
                info = torch.iinfo(x.dtype)
                if lo is not None:
                    lo = max(int(lo), info.min)
                if hi is not None:
                    hi = min(int(hi), info.max)
            if lo is not None:
                x = torch.maximum(x, torch.tensor(lo, dtype=x.dtype, device=x.device))
            if hi is not None:
                x = torch.minimum(x, torch.tensor(hi, dtype=x.dtype, device=x.device))
        return x
    if op == "relu":
        return F.relu(xs[0])
    if op == "sigmoid":
        return torch.sigmoid(xs[0])
    if op == "tanh":
        return torch.tanh(xs[0])
    if op == "softmax":
        axis = node.attrs.get("axis", -1)
        return F.softmax(xs[0], dim=axis)
    if op == "matmul":
        return torch.matmul(xs[0], xs[1])
    if op == "batchNormalization":
        x = xs[0]
        if x.dim() == 0:
            raise RuntimeError("batchNormalization requires tensor rank >= 1")
        axis = int(node.attrs.get("axis", 1))
        axis = _normalize_axis(axis, x.dim())
        channel_dim = x.shape[axis]
        if channel_dim == 0:
            raise RuntimeError("batchNormalization channel axis is empty")
        epsilon = float(node.attrs.get("epsilon", 1e-5))
        mean = xs[1]
        variance = xs[2]
        next_idx = 3
        has_scale = bool(node.attrs.get("hasScale", len(xs) > next_idx))
        scale = xs[next_idx] if has_scale and len(xs) > next_idx else None
        if has_scale:
            next_idx += 1
        has_bias_default = len(xs) > next_idx
        has_bias = bool(node.attrs.get("hasBias", has_bias_default))
        bias = xs[next_idx] if has_bias and len(xs) > next_idx else None

        def _reshape_param(param: torch.Tensor, name: str) -> torch.Tensor:
            p = param.to(x.dtype)
            if p.dim() == x.dim():
                return p
            flat = p.reshape(-1)
            if flat.numel() == 0:
                raise RuntimeError(f"batchNormalization {name} tensor is empty")
            view = [1] * x.dim()
            if flat.numel() == 1:
                view[axis] = 1
                return flat.reshape(view)
            if channel_dim is None:
                raise RuntimeError("batchNormalization requires static channel size at runtime")
            expected = int(channel_dim)
            if flat.numel() != expected:
                raise RuntimeError(
                    f"batchNormalization {name} has incompatible size {flat.numel()} "
                    f"for channel dimension {expected}"
                )
            view[axis] = expected
            return flat.reshape(view)

        mean = _reshape_param(mean, "mean")
        variance = _reshape_param(variance, "variance")
        scale = _reshape_param(scale, "scale") if scale is not None else None
        bias = _reshape_param(bias, "bias") if bias is not None else None
        eps = torch.tensor(epsilon, dtype=x.dtype, device=x.device)
        y = (x - mean) * torch.rsqrt(variance + eps)
        if scale is not None:
            y = y * scale
        if bias is not None:
            y = y + bias
        return y
    if op == "conv2d":
        # Expect NCHW tensors and OIHW weights (like PyTorch); groups supported
        x, w = xs[0], xs[1]
        b = xs[2] if len(xs) == 3 else None
        strides = node.attrs.get("strides", (1, 1))
        dilations = node.attrs.get("dilations", (1, 1))
        padding = node.attrs.get("padding", "valid")
        groups = int(node.attrs.get("groups", 1))
        pad = _pad_arg(padding, x, (w.shape[-2], w.shape[-1]), strides)
        if any(pad):
            # torch conv2d expects (left,right,top,bottom) in pad via F.pad with mode='constant'
            x = F.pad(x, pad)
        y = F.conv2d(x, w, bias=b, stride=strides, dilation=dilations, groups=groups)
        return y
    if op == "maxPool2d" or op == "averagePool2d":
        x = xs[0]
        window = node.attrs["window"]
        strides = node.attrs.get("strides", (1, 1))
        padding = node.attrs.get("padding", "valid")
        pad = _pad_arg(padding, x, window, strides)
        if any(pad):
            x = F.pad(x, pad, value=0)
        if op == "maxPool2d":
            return F.max_pool2d(x, kernel_size=window, stride=strides)
        else:
            return F.avg_pool2d(x, kernel_size=window, stride=strides)
    if op == "reshape":
        shape = node.attrs["shape"]
        return torch.reshape(xs[0], shape)
    if op == "transpose":
        perm = node.attrs.get("permutation")
        if perm is None:
            perm = list(reversed(range(xs[0].ndim)))
        return xs[0].permute(*perm)
    if op == "concat":
        axis = int(node.attrs.get("axis", 0))
        return torch.cat(xs, dim=axis)
    if op == "gather":
        data, indices = xs
        axis = _normalize_axis(node.attrs.get("axis", 0), data.dim())
        idx = indices.to(torch.long)
        if data.shape[axis] == 0:
            raise RuntimeError("Cannot gather on empty axis")
        idx = idx.clamp(0, data.shape[axis] - 1)
        flat = idx.reshape(-1)
        gathered = torch.index_select(data, dim=axis, index=flat)
        target_shape = list(data.shape)
        target_shape.pop(axis)
        idx_shape = list(idx.shape)
        for offset, dim in enumerate(idx_shape):
            target_shape.insert(axis + offset, dim)
        if not target_shape:
            return gathered.reshape(())
        return gathered.reshape(target_shape)
    if op == "slice":
        tensor = xs[0]
        starts = node.attrs.get("starts", [])
        sizes = node.attrs.get("sizes", [])
        strides = node.attrs.get("strides") or []
        if not starts:
            return tensor
        if len(starts) != len(sizes):
            raise RuntimeError("slice starts and sizes must match in length")
        slices: List[slice] = []
        for dim, (start, size) in enumerate(zip(starts, sizes)):
            step = 1
            if dim < len(strides) and strides[dim] is not None:
                step = int(strides[dim])
            if step <= 0:
                raise RuntimeError("slice strides must be positive")
            stop = int(start) + int(size)
            slices.append(slice(int(start), stop, step))
        # pad remaining dimensions
        for _ in range(len(starts), tensor.ndim):
            slices.append(slice(None))
        return tensor[tuple(slices)]

    raise NotImplementedError(f"Op not implemented in toy runtime: {op}")


def _descriptor_dict(
    desc: MLOperandDescriptor, fallback_shape: Optional[List[int]] = None
) -> Dict[str, Any]:
    dims = desc.dimensions if desc.dimensions is not None else fallback_shape
    if dims is None:
        raise ValueError("Rust backend requires static tensor dimensions")
    shape = _normalize_shape(dims)
    return {"dataType": desc.dataType, "shape": shape}


def _normalize_shape(shape: List[Optional[int]]) -> List[int]:
    normalized: List[int] = []
    for dim in shape:
        if dim is None:
            raise ValueError("Rust backend does not support dynamic dimensions")
        normalized.append(int(dim))
    return normalized


def _tensor_shape(t: torch.Tensor) -> List[int]:
    return [int(d) for d in t.shape]


def _tensor_to_flat_list(t: torch.Tensor, dtype_str: str = None) -> List[float]:
    data = t.detach().cpu()
    if dtype_str is None:
        return data.reshape(-1).tolist()
    if dtype_str in ("float32", "float16", "bfloat16"):
        if data.dtype == torch.float16:
            data = data.to(torch.float32)
        return data.reshape(-1).tolist()
    data = data.to(torch.int64)
    return data.reshape(-1).tolist()


def _serialize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return {k: convert(v) for k, v in attrs.items()}


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        raise ValueError("Axis normalization on empty rank")
    ax = int(axis)
    if ax < 0:
        ax += rank
    if ax < 0 or ax >= rank:
        raise ValueError(f"Axis {axis} out of range for rank {rank}")
    return ax


def _gather_output_dims(
    data_desc: MLOperandDescriptor,
    index_desc: MLOperandDescriptor,
    axis: int,
) -> Optional[List[Optional[int]]]:
    dims = list(data_desc.dimensions or [])
    if not dims:
        return dims
    ax = _normalize_axis(axis, len(dims))
    idx_dims = list(index_desc.dimensions or [])
    dims.pop(ax)
    if idx_dims:
        for offset, dim in enumerate(idx_dims):
            dims.insert(ax + offset, dim)
    return dims


def _slice_output_dims(
    data_desc: MLOperandDescriptor,
    sizes: List[int],
    strides: Optional[List[int]] = None,
) -> Optional[List[Optional[int]]]:
    dims = list(data_desc.dimensions or [])
    if not sizes:
        return dims
    out: List[Optional[int]] = []
    for i, size in enumerate(sizes):
        dim = dims[i] if i < len(dims) else None
        step = 1
        if strides and i < len(strides) and strides[i] is not None:
            step = int(strides[i])
        eff = size if step == 1 else int(math.ceil(size / step))
        out.append(eff if dim is not None else None)
    if len(dims) > len(sizes):
        out.extend(dims[len(sizes) :])
    return out


def _torch_to_payload(
    tensor: TensorLike, descriptor: MLOperandDescriptor
) -> Dict[str, Any]:
    t = torch.as_tensor(tensor)
    dt = _dtype_from_str(descriptor.dataType)
    t = t.to(dt).detach().cpu()
    data = _tensor_to_flat_list(t, descriptor.dataType)
    shape = _tensor_shape(t)
    return {
        "data": data,
        "descriptor": {"dataType": descriptor.dataType, "shape": shape},
    }


def _payload_to_torch(payload: Dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype_str = payload["dataType"]
    dtype = _dtype_from_str(dtype_str)
    shape = [int(dim) for dim in payload["shape"]]
    np_dtype = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "uint32": np.int64,
    }[dtype_str]
    arr = np.array(payload["data"], dtype=np_dtype).reshape(shape)
    return torch.tensor(arr, device=device, dtype=dtype)


# --- Minimal test/demo -------------------------------------------------------
if __name__ == "__main__":
    # small smoke test for the module
    ctx = MLContext()
    b = MLGraphBuilder(ctx)
    x = b.input("x", {"dataType": "float32", "dimensions": [None, 4]})
    w = b.constant({"dataType": "float32", "dimensions": [4, 3]}, torch.randn(4, 3))
    y = b.matmul(x, w)
    y = b.relu(y)
    g = b.build({"y": y})
    out = g.compute({"x": torch.randn(2, 4)})
    print("Output shape:", tuple(out["y"].shape))
