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
| `abs` | todo | `add` | done | `argMax` | todo |
| `argMin` | todo | `averagePool2d` | done | `batchNormalization` | done |
| `cast` | todo | `ceil` | todo | `clamp` | done |
| `concat` | done | `constant` | done | `conv2d` | done |
| `convTranspose2d` | todo | `cos` | todo | `cumulativeSum` | todo |
| `dequantizeLinear` | todo | `div` | todo | `elu` | todo |
| `equal` | todo | `erf` | todo | `exp` | todo |
| `expand` | todo | `floor` | todo | `gather` | done |
| `gatherElements` | todo | `gatherND` | todo | `gelu` | todo |
| `gemm` | todo | `greater` | todo | `greaterOrEqual` | todo |
| `gru` | todo | `gruCell` | todo | `hardSigmoid` | todo |
| `hardSwish` | todo | `identity` | todo | `input` | done |
| `instanceNormalization` | todo | `isInfinite` | todo | `isNaN` | todo |
| `l2Pool2d` | todo | `layerNormalization` | todo | `leakyRelu` | todo |
| `lesser` | todo | `lesserOrEqual` | todo | `linear` | todo |
| `log` | todo | `logicalAnd` | todo | `logicalNot` | todo |
| `logicalOr` | todo | `logicalXor` | todo | `lstm` | todo |
| `lstmCell` | todo | `matmul` | done | `max` | todo |
| `maxPool2d` | done | `min` | todo | `mul` | todo |
| `neg` | todo | `notEqual` | todo | `pad` | todo |
| `pow` | todo | `prelu` | todo | `quantizeLinear` | todo |
| `reciprocal` | todo | `reduceL1` | todo | `reduceL2` | todo |
| `reduceLogSum` | todo | `reduceLogSumExp` | todo | `reduceMax` | todo |
| `reduceMean` | todo | `reduceMin` | todo | `reduceProduct` | todo |
| `reduceSum` | todo | `reduceSumSquare` | todo | `relu` | done |
| `resample2d` | todo | `reshape` | done | `reverse` | todo |
| `roundEven` | todo | `scatterElements` | todo | `scatterND` | todo |
| `sigmoid` | done | `sign` | todo | `sin` | todo |
| `slice` | done | `softmax` | done | `softplus` | todo |
| `softsign` | todo | `split` | todo | `sqrt` | todo |
| `sub` | todo | `tan` | todo | `tanh` | done |
| `tile` | todo | `transpose` | done | `triangular` | todo |
| `where` | todo |   |   |   |   |

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
