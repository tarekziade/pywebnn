# PyWebNN

PyWebNN is a toy WebNN-style graph builder and runtime that mirrors a tiny subset of
the [Web Neural Network specification](https://www.w3.org/TR/webnn/). It exists
solely for experimentation:

- The primary backend is implemented in pure Python on top of PyTorch tensors.
- A secondary backend is written in Rust using the `tch` crate and packaged
  into this project via [maturin](https://github.com/PyO3/maturin).
- Web Platform Tests (WPT) for ops such as `matmul`, `add`, `clamp`, `relu`,
  `softmax`, `maxPool2d`, `gather`, `slice`, and `conv2d` are ported to Python so both backends can be validated against
  the same reference data.

The intent is educational: to explore how a WebNN graph builder might look,
experiment with device selection (CPU/GPU/NPU), and compare a dynamic Python
runtime with a statically compiled Rust backend.

## Operator coverage

The [WebNN specification](https://www.w3.org/TR/webnn/) defines the following
`MLGraphBuilder` methods. The table tracks PyWebNN’s current implementation
status; `done` means there is a backend implementation plus at least one
WPT-derived test vector, while `todo` denotes unimplemented operations.

| Operation | Status | Operation | Status | Operation | Status |
|-----------|--------|-----------|--------|-----------|--------|
| `abs` |  | `add` | ✓ | `argMax` |  |
| `argMin` |  | `averagePool2d` | ✓ | `batchNormalization` | ✓ |
| `cast` |  | `ceil` |  | `clamp` | ✓ |
| `concat` | ✓ | `constant` | ✓ | `conv2d` | ✓ |
| `convTranspose2d` |  | `cos` |  | `cumulativeSum` |  |
| `dequantizeLinear` |  | `div` |  | `elu` |  |
| `equal` |  | `erf` |  | `exp` |  |
| `expand` |  | `floor` |  | `gather` | ✓ |
| `gatherElements` |  | `gatherND` |  | `gelu` |  |
| `gemm` |  | `greater` |  | `greaterOrEqual` |  |
| `gru` |  | `gruCell` |  | `hardSigmoid` |  |
| `hardSwish` |  | `identity` |  | `input` | ✓ |
| `instanceNormalization` |  | `isInfinite` |  | `isNaN` |  |
| `l2Pool2d` |  | `layerNormalization` |  | `leakyRelu` |  |
| `lesser` |  | `lesserOrEqual` |  | `linear` |  |
| `log` |  | `logicalAnd` |  | `logicalNot` |  |
| `logicalOr` |  | `logicalXor` |  | `lstm` |  |
| `lstmCell` |  | `matmul` | ✓ | `max` |  |
| `maxPool2d` | ✓ | `min` |  | `mul` |  |
| `neg` |  | `notEqual` |  | `pad` |  |
| `pow` |  | `prelu` |  | `quantizeLinear` |  |
| `reciprocal` |  | `reduceL1` |  | `reduceL2` |  |
| `reduceLogSum` |  | `reduceLogSumExp` |  | `reduceMax` |  |
| `reduceMean` |  | `reduceMin` |  | `reduceProduct` |  |
| `reduceSum` |  | `reduceSumSquare` |  | `relu` | ✓ |
| `resample2d` |  | `reshape` | ✓ | `reverse` |  |
| `roundEven` |  | `scatterElements` |  | `scatterND` |  |
| `sigmoid` | ✓ | `sign` |  | `sin` |  |
| `slice` | ✓ | `softmax` | ✓ | `softplus` |  |
| `softsign` |  | `split` |  | `sqrt` |  |
| `sub` |  | `tan` |  | `tanh` | ✓ |
| `tile` |  | `transpose` | ✓ | `triangular` |  |
| `where` |  |   |   |   |   |

## Missing pieces

This is not a complete WebNN implementation. Notable omissions include:

- Many WebNN ops (e.g., `element-wise` variations, activation/broadcast
  variants beyond the basics).
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
