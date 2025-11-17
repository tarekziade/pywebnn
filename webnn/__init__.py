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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, np.ndarray, float, int]


def _to_tensor(
    x: TensorLike, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
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
        "uint32": torch.int32,  # toy simplification
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
        name: Optional[str],
        desc: MLOperandDescriptor,
        const_value: Optional[torch.Tensor] = None,
        node: Optional[_Node] = None,
    ):
        self._builder = builder
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

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self._destroyed = False
        self._device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device_like: Optional[Union[str, torch.device]]) -> torch.device:
        if device_like is None:
            return _auto_select_device()
        if isinstance(device_like, torch.device):
            return device_like
        return torch.device(device_like)

    @property
    def device(self) -> torch.device:
        return self._device

    def destroy(self):
        self._destroyed = True


class MLGraphBuilder:
    def __init__(self, context: MLContext):
        self._ctx = context
        self._operands: List[MLOperand] = []
        self._inputs: Dict[str, MLOperand] = {}
        self._constants: List[MLOperand] = []

    def input(self, name: str, descriptor: Dict[str, Any]) -> MLOperand:
        desc = MLOperandDescriptor(
            dataType=descriptor.get("dataType", "float32"),
            dimensions=descriptor.get("dimensions"),
        )
        op = MLOperand(self, name=name, desc=desc)
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
        const = MLOperand(self, name=None, desc=desc, const_value=t)
        self._operands.append(const)
        self._constants.append(const)
        return const

    # -- helper to register op outputs ---------------------------------------
    def _emit(self, op: str, inputs: List[MLOperand], **attrs) -> MLOperand:
        node = _Node(op=op, inputs=inputs, attrs=attrs)
        # Infer a basic descriptor based on first input
        base = inputs[0].descriptor if inputs else MLOperandDescriptor()
        out = MLOperand(self, name=None, desc=base, node=node)
        self._operands.append(out)
        return out

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

        return MLGraph(self._ctx, order, outputs, self._inputs)


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

    raise NotImplementedError(f"Op not implemented in toy runtime: {op}")


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
