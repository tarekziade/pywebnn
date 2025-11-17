# PyWebNN

PyWebNN is a toy WebNN-style graph builder and runtime that mirrors a tiny subset of
the [Web Neural Network specification](https://www.w3.org/TR/webnn/). It exists
solely for experimentation:

- The primary backend is implemented in pure Python on top of PyTorch tensors.
- A secondary backend is written in Rust using the `tch` crate and packaged
  into this project via [maturin](https://github.com/PyO3/maturin).
- Web Platform Tests (WPT) for ops such as `matmul`, `add`, `clamp`, `softmax`,
  and `conv2d` are ported to Python so both backends can be validated against
  the same reference data.

The intent is educational: to explore how a WebNN graph builder might look,
experiment with device selection (CPU/GPU/NPU), and compare a dynamic Python
runtime with a statically compiled Rust backend.
