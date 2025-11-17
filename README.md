# PyWebNN

PyWebNN is a toy WebNN-style graph builder and runtime that mirrors a tiny subset of
the [Web Neural Network specification](https://www.w3.org/TR/webnn/). It exists
solely for experimentation:

- The primary backend is implemented in pure Python on top of PyTorch tensors.
- A secondary backend is written in Rust using the `tch` crate and packaged
  into this project via [maturin](https://github.com/PyO3/maturin).
- Web Platform Tests (WPT) for ops such as `matmul`, `add`, `clamp`, `relu`,
  `softmax`, `maxPool2d`, and `conv2d` are ported to Python so both backends can be validated against
  the same reference data.

The intent is educational: to explore how a WebNN graph builder might look,
experiment with device selection (CPU/GPU/NPU), and compare a dynamic Python
runtime with a statically compiled Rust backend.

## Implemented operators

Both backends currently support the following operations:

- `input` and `constant` operands for feeding graph values.
- Elementwise math: `add`, `clamp`, activation functions (`relu`, `sigmoid`,
  `tanh`), and `softmax`.
- Linear algebra: `matmul`.
- Convolutional ops: `conv2d` (NCHW inputs, OIHW filters, optional bias),
  `maxPool2d`, and `averagePool2d`.
- Tensor transforms: `reshape`, `transpose`, and `concat`.

Every op listed above has at least one WPT-derived test vector in
`tests/data/` and is exercised via the conformance suites in
`tests/test_matmul.py` and `tests/test_wpt_ops.py`.

## Missing pieces

This is not a complete WebNN implementation. Notable omissions include:

- Many WebNN ops (e.g., `batchNormalization`, `gather`, `slice`, `element-wise`
  variations, activation/broadcast variants beyond the basics).
- Advanced conv2d features such as NHWC inputs, HWIO filters, fused activations,
  automatic padding modes beyond the simple SAME/VALID that the Rust backend
  understands.
- Tensor layout/quantization controls, operand type promotion, and shape
  inference logic from the spec.
- Execution features like streaming inputs, tensor views, memory planning, and
  async graph execution.

Extending the operator surface is straightforward: add the op to
`webnn/__init__.py`, teach the Rust backend how to evaluate it, and port the
corresponding WPT vectors into `tests/data/` so both backends stay in sync.
