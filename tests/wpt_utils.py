import json
import math
import pathlib
from typing import Dict, List, Tuple

import torch

from webnn import MLContext, MLGraphBuilder, MLOperand

BACKENDS: List[str] = ["python"]
try:  # pragma: no cover - optional dependency
    import pywebnn_rust as _  # noqa: F401

    BACKENDS.append("rust")
except ImportError:  # pragma: no cover - optional dependency
    pass


DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
}

DTYPE_TOLERANCE: Dict[str, Tuple[float, float]] = {
    "float32": (1e-4, 1e-4),
    "float16": (1e-2, 1e-2),
}


class UnsupportedCase(Exception):
    """Raised when a WPT case depends on unsupported types or options."""


def load_wpt_cases(filename: str) -> List[dict]:
    data_path = pathlib.Path(__file__).with_name("data").joinpath(filename)
    with data_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def run_wpt_case(
    case: dict, backend: str = "python"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Tuple[float, float]]]:
    """Build the graph described by `case` and return computed + expected outputs."""

    ctx = MLContext(backend=backend)
    builder = MLGraphBuilder(ctx)

    operands, feeds = _prepare_inputs(builder, case["graph"]["inputs"])

    for op in case["graph"].get("operators", []):
        _apply_operator(builder, operands, op)

    outputs = case["graph"]["expectedOutputs"]
    output_map: Dict[str, torch.Tensor] = {}
    expected_tensors: Dict[str, torch.Tensor] = {}
    tolerances: Dict[str, Tuple[float, float]] = {}
    for name, spec in outputs.items():
        desc = spec["descriptor"]
        dtype = desc["dataType"]
        if dtype not in DTYPE_MAP:
            raise UnsupportedCase(f"Unsupported expected dtype '{dtype}'")
        expected_tensors[name] = _tensor_from_descriptor(desc, spec["data"])
        tolerances[name] = DTYPE_TOLERANCE[dtype]
        if name not in operands:
            raise UnsupportedCase(f"Output operand '{name}' not constructed")
        output_map[name] = operands[name]

    graph = builder.build(output_map)
    results = graph.compute(feeds)
    return results, expected_tensors, tolerances


# -- internal helpers ---------------------------------------------------------
def _tensor_from_descriptor(desc: Dict[str, object], values) -> torch.Tensor:
    dtype = desc["dataType"]
    if dtype not in DTYPE_MAP:
        raise UnsupportedCase(f"Unsupported dtype '{dtype}'")
    if isinstance(values, list):
        flat_values = values
    else:
        flat_values = [values]
    if any(v is None for v in flat_values):
        raise UnsupportedCase("Encountered non-finite literal in test data")
    tensor = torch.tensor(flat_values, dtype=DTYPE_MAP[dtype])
    shape = desc.get("shape")
    if shape is None:
        raise UnsupportedCase("Descriptor missing shape information")
    if any(dim is None for dim in shape):
        raise UnsupportedCase("Dynamic shapes are not supported in the toy runner")
    expected_elems = math.prod(shape) if shape else 1
    if expected_elems != tensor.numel():
        raise UnsupportedCase("Shape and data length mismatch in test case")
    return tensor.reshape(shape)


def _prepare_inputs(builder: MLGraphBuilder, inputs: Dict[str, dict]) -> Tuple[Dict[str, MLOperand], Dict[str, torch.Tensor]]:
    operands: Dict[str, MLOperand] = {}
    feeds: Dict[str, torch.Tensor] = {}
    for name, spec in inputs.items():
        desc = spec["descriptor"]
        if "shape" not in desc or desc["shape"] is None:
            raise UnsupportedCase(f"Input '{name}' missing shape information")
        tensor = _tensor_from_descriptor(desc, spec["data"])
        descriptor = {"dataType": desc["dataType"], "dimensions": desc.get("shape")}
        if spec.get("constant"):
            operands[name] = builder.constant(descriptor, tensor)
        else:
            operands[name] = builder.input(name, descriptor)
            feeds[name] = tensor
    return operands, feeds


def _apply_operator(builder: MLGraphBuilder, operands: Dict[str, MLOperand], op_spec: dict):
    args = {}
    for entry in op_spec.get("arguments", []):
        args.update(entry)

    name = op_spec["name"]
    if name == "add":
        result = builder.add(operands[args["a"]], operands[args["b"]])
    elif name == "clamp":
        options = args.get("options", {})
        result = builder.clamp(
            operands[args["input"]],
            minValue=options.get("minValue"),
            maxValue=options.get("maxValue"),
        )
    elif name == "relu":
        result = builder.relu(operands[args["input"]])
    elif name == "softmax":
        axis = args.get("axis", -1)
        result = builder.softmax(operands[args["input"]], axis=axis)
    elif name == "conv2d":
        result = _apply_conv2d(builder, operands, args)
    elif name == "maxPool2d":
        result = _apply_pool(
            builder.maxPool2d,
            operands[args["input"]],
            args.get("options", {}),
        )
    elif name == "matmul":
        result = builder.matmul(operands[args["a"]], operands[args["b"]])
    else:
        raise UnsupportedCase(f"Unsupported operator '{name}'")

    outputs = op_spec["outputs"]
    if isinstance(outputs, str):
        outputs = [outputs]
    if len(outputs) != 1:
        raise UnsupportedCase("Multi-output operators are not supported in the toy runner")
    operands[outputs[0]] = result


def _apply_conv2d(builder: MLGraphBuilder, operands: Dict[str, MLOperand], args: dict):
    options = args.get("options", {})
    input_layout = options.get("inputLayout", "nchw")
    filter_layout = options.get("filterLayout", "oihw")
    if input_layout != "nchw" or filter_layout != "oihw":
        raise UnsupportedCase(
            f"Unsupported conv2d layout input={input_layout} filter={filter_layout}"
        )

    padding = options.get("padding", "valid")
    if isinstance(padding, list):
        if len(padding) != 4:
            raise UnsupportedCase("conv2d padding list must have 4 elements")
        pad_top, pad_bottom, pad_left, pad_right = padding
        padding = (pad_left, pad_right, pad_top, pad_bottom)

    strides = tuple(options.get("strides", (1, 1)))
    dilations = tuple(options.get("dilations", (1, 1)))
    if len(strides) != 2 or len(dilations) != 2:
        raise UnsupportedCase("conv2d strides/dilations must have 2 values")

    bias_name = options.get("bias")
    bias = operands[bias_name] if bias_name else None

    return builder.conv2d(
        operands[args["input"]],
        operands[args["filter"]],
        bias,
        strides=tuple(int(s) for s in strides),
        dilations=tuple(int(d) for d in dilations),
        padding=padding,
        groups=int(options.get("groups", 1)),
    )


def _apply_pool(pool_fn, operand: MLOperand, options: dict) -> MLOperand:
    opts = options or {}
    window = opts.get("windowDimensions")
    if window is None:
        dims = operand.descriptor.dimensions
        if dims is None or len(dims or []) < 2:
            raise UnsupportedCase("Pool ops require windowDimensions")
        h, w = dims[-2], dims[-1]
        if h is None or w is None:
            raise UnsupportedCase("Cannot infer windowDimensions from dynamic shape")
        window = [h, w]
    if len(window) != 2:
        raise UnsupportedCase("windowDimensions must have 2 values")
    layout = opts.get("layout", "nchw")
    if layout != "nchw":
        raise UnsupportedCase(f"Unsupported pool layout '{layout}'")
    strides = tuple(opts.get("strides", (1, 1)))
    padding = opts.get("padding", "valid")
    if isinstance(padding, list):
        if len(padding) != 4:
            raise UnsupportedCase("Pool padding list must have 4 elements")
        pad_top, pad_bottom, pad_left, pad_right = padding
        padding = (pad_left, pad_right, pad_top, pad_bottom)
    dilations = tuple(opts.get("dilations", (1, 1)))
    if dilations != (1, 1):
        raise UnsupportedCase("Pool dilations other than 1 are not supported")
    rounding = opts.get("roundingType")
    if rounding not in (None, "floor"):
        raise UnsupportedCase(f"Unsupported roundingType '{rounding}'")
    if opts.get("outputSizes") not in (None, []):
        raise UnsupportedCase("Pool outputSizes not supported")
    return pool_fn(
        operand,
        windowDimensions=tuple(int(v) for v in window),
        strides=tuple(int(s) for s in strides),
        padding=padding,
    )
