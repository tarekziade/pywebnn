use anyhow::{anyhow, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::convert::TryFrom;
use tch::kind::Kind;
use tch::{Tensor, Device};

#[derive(Deserialize)]
struct Descriptor {
    #[serde(rename = "dataType")]
    data_type: String,
    shape: Vec<i64>,
}

#[derive(Deserialize)]
struct NodeSpec {
    id: usize,
    op: String,
    inputs: Vec<usize>,
    attrs: Value,
    descriptor: Descriptor,
}

#[derive(Deserialize)]
struct ConstantSpec {
    id: usize,
    descriptor: Descriptor,
    data: Vec<f64>,
}

#[derive(Deserialize)]
struct InputSpec {
    id: usize,
    name: String,
}

#[derive(Deserialize)]
struct GraphSpec {
    nodes: Vec<NodeSpec>,
    constants: Vec<ConstantSpec>,
    inputs: Vec<InputSpec>,
    outputs: HashMap<String, usize>,
}

#[derive(Deserialize)]
struct FeedTensor {
    data: Vec<f64>,
    descriptor: Descriptor,
}

#[derive(Serialize)]
struct OutputTensor {
    data: Vec<f64>,
    shape: Vec<i64>,
    #[serde(rename = "dataType")]
    data_type: String,
}

struct ValueTensor {
    tensor: Tensor,
    dtype: DataType,
}

impl ValueTensor {
    fn from_descriptor(desc: &Descriptor, data: &[f64]) -> Result<Self> {
        let dtype = DataType::from_str(&desc.data_type)?;
        let tensor = tensor_from_f64(data, dtype)?
            .reshape(&desc.shape);
        Ok(Self { tensor, dtype })
    }

    fn to_output(&self) -> Result<OutputTensor> {
        let shape = self.tensor.size();
        let flat = self.tensor.shallow_clone().reshape(&[-1]);
        let data = match self.dtype {
            DataType::F32 | DataType::F16 => {
                let as_float = flat.to_kind(Kind::Float);
                Vec::<f32>::try_from(as_float)?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect()
            }
            _ => {
                let as_int = flat.to_kind(Kind::Int64);
                Vec::<i64>::try_from(as_int)?
                    .into_iter()
                    .map(|v| v as f64)
                    .collect()
            }
        };
        Ok(OutputTensor {
            data,
            shape,
            data_type: self.dtype.as_str().to_string(),
        })
    }
}

fn tensor_from_f64(data: &[f64], dtype: DataType) -> Result<Tensor> {
    let tensor = match dtype {
        DataType::F32 => Tensor::f_from_slice(data).context("tensor convert f32 failed")?,
        DataType::F16 => Tensor::f_from_slice(data)
            .context("tensor convert f16 failed")?
            .to_kind(Kind::Half),
        DataType::I32 => {
            let vals: Vec<i32> = data.iter().map(|v| *v as i32).collect();
            Tensor::f_from_slice(&vals).context("tensor convert i32 failed")?
        }
        DataType::I64 => {
            let vals: Vec<i64> = data.iter().map(|v| *v as i64).collect();
            Tensor::f_from_slice(&vals).context("tensor convert i64 failed")?
        }
        DataType::U32 => {
            let vals: Vec<i64> = data.iter().map(|v| *v as i64).collect();
            Tensor::f_from_slice(&vals).context("tensor convert u32 failed")?
        }
    };
    Ok(tensor.to_kind(dtype.kind()))
}

fn scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    match dtype {
        DataType::F32 => Ok(Tensor::from(value as f32).to_device(device)),
        DataType::F16 => Ok(Tensor::from(value as f32).to_kind(Kind::Half).to_device(device)),
        DataType::I32 => {
            let t = Tensor::f_from_slice(&[value as i32]).context("scalar i32 failed")?;
            Ok(t.to_device(device))
        }
        DataType::I64 | DataType::U32 => {
            let t = Tensor::f_from_slice(&[value as i64]).context("scalar i64 failed")?;
            Ok(t.to_device(device))
        }
    }
}

#[derive(Clone, Copy)]
enum DataType {
    F32,
    F16,
    I32,
    I64,
    U32,
}

impl DataType {
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "float32" => Ok(Self::F32),
            "float16" => Ok(Self::F16),
            "int32" => Ok(Self::I32),
            "int64" => Ok(Self::I64),
            "uint32" => Ok(Self::U32),
            other => Err(anyhow!("Unsupported data type '{}'", other)),
        }
    }

    fn kind(self) -> Kind {
        match self {
            Self::F32 => Kind::Float,
            Self::F16 => Kind::Half,
            Self::I32 => Kind::Int,
            Self::I64 | Self::U32 => Kind::Int64,
        }
    }

    fn int_bounds(self) -> Option<(i64, i64)> {
        match self {
            Self::I32 => Some((i32::MIN as i64, i32::MAX as i64)),
            Self::I64 => Some((i64::MIN, i64::MAX)),
            Self::U32 => Some((0, u32::MAX as i64)),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U32 => "uint32",
        }
    }
}

fn run_graph(
    graph: &GraphSpec,
    feeds: &HashMap<String, FeedTensor>,
) -> Result<HashMap<String, OutputTensor>> {
    let mut env: HashMap<usize, ValueTensor> = HashMap::new();

    for input in &graph.inputs {
        let feed = feeds
            .get(&input.name)
            .with_context(|| format!("Missing feed for input '{}'", input.name))?;
        let value = ValueTensor::from_descriptor(&feed.descriptor, &feed.data)?;
        env.insert(input.id, value);
    }

    for cst in &graph.constants {
        let value = ValueTensor::from_descriptor(&cst.descriptor, &cst.data)?;
        env.insert(cst.id, value);
    }

    for node in &graph.nodes {
        let tensors: Vec<ValueTensor> = node
            .inputs
            .iter()
            .map(|idx| {
                env.get(idx)
                    .with_context(|| format!("Missing operand {}", idx))
                    .map(|val| ValueTensor {
                        tensor: val.tensor.shallow_clone(),
                        dtype: val.dtype,
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = eval_node(node, &tensors)?;
        env.insert(node.id, result);
    }

    let mut outputs = HashMap::new();
    for (name, id) in &graph.outputs {
        let value = env
            .get(id)
            .with_context(|| format!("Missing output operand {}", id))?;
        outputs.insert(name.clone(), value.to_output()?);
    }
    Ok(outputs)
}

fn eval_node(node: &NodeSpec, inputs: &[ValueTensor]) -> Result<ValueTensor> {
    let dtype = DataType::from_str(&node.descriptor.data_type)?;
    let tensor = match node.op.as_str() {
        "add" => {
            ensure_inputs("add", inputs, 2)?;
            &inputs[0].tensor + &inputs[1].tensor
        }
        "clamp" => {
            ensure_inputs("clamp", inputs, 1)?;
            apply_clamp(
                inputs[0].tensor.shallow_clone(),
                inputs[0].dtype,
                attr_f64(&node.attrs, "min")?,
                attr_f64(&node.attrs, "max")?,
            )?
        }
        "softmax" => {
            ensure_inputs("softmax", inputs, 1)?;
            let axis = attr_i64(&node.attrs, "axis")?.unwrap_or(-1);
            inputs[0].tensor.softmax(axis, Kind::Float)
        }
        "relu" => {
            ensure_inputs("relu", inputs, 1)?;
            inputs[0].tensor.relu()
        }
        "matmul" => {
            ensure_inputs("matmul", inputs, 2)?;
            inputs[0].tensor.matmul(&inputs[1].tensor)
        }
        "conv2d" => eval_conv2d(node, inputs)?,
        "maxPool2d" => eval_max_pool2d(node, inputs)?,
        "gather" => eval_gather(node, inputs)?,
        "slice" => eval_slice(node, inputs)?,
        other => return Err(anyhow!("Unsupported op '{}'", other)),
    };
    Ok(ValueTensor { tensor, dtype })
}

fn eval_conv2d(node: &NodeSpec, inputs: &[ValueTensor]) -> Result<Tensor> {
    if inputs.len() < 2 {
        return Err(anyhow!("conv2d requires at least 2 inputs"));
    }
    let mut input = inputs[0].tensor.shallow_clone();
    let filter = inputs[1].tensor.shallow_clone();
    let bias = if inputs.len() > 2 {
        Some(inputs[2].tensor.shallow_clone())
    } else {
        None
    };
    let strides = attr_list_i64(&node.attrs, "strides")?.unwrap_or_else(|| vec![1, 1]);
    let dilations = attr_list_i64(&node.attrs, "dilations")?.unwrap_or_else(|| vec![1, 1]);
    if strides.len() != 2 || dilations.len() != 2 {
        return Err(anyhow!(
            "conv2d strides and dilations must each have 2 elements"
        ));
    }
    let padding = resolve_padding(&node.attrs, &input, &filter, &strides)?;
    let groups = attr_i64(&node.attrs, "groups")?.unwrap_or(1);

    if padding.iter().any(|&p| p != 0) {
        input = input
            .f_constant_pad_nd(&padding)
            .context("constant_pad_nd failed")?;
    }
    let bias_ref = bias.as_ref();
    let result = input.conv2d(
        &filter,
        bias_ref,
        &strides,
        &[0, 0],
        &dilations,
        groups,
    );
    Ok(result)
}

fn eval_max_pool2d(node: &NodeSpec, inputs: &[ValueTensor]) -> Result<Tensor> {
    ensure_inputs("maxPool2d", inputs, 1)?;
    let mut input = inputs[0].tensor.shallow_clone();
    let window = attr_list_i64(&node.attrs, "window")?
        .ok_or_else(|| anyhow!("maxPool2d missing window dimensions"))?;
    if window.len() != 2 {
        return Err(anyhow!("maxPool2d window must have 2 elements"));
    }
    let strides = attr_list_i64(&node.attrs, "strides")?.unwrap_or_else(|| vec![1, 1]);
    if strides.len() != 2 {
        return Err(anyhow!("maxPool2d strides must have 2 elements"));
    }
    let padding = resolve_padding_with_kernel(
        node.attrs.get("padding"),
        input.size()[2],
        input.size()[3],
        window[0],
        window[1],
        &strides,
    )?;
    if padding.iter().any(|&p| p != 0) {
        input = input
            .f_constant_pad_nd(&padding)
            .context("maxPool2d padding failed")?;
    }
    Ok(input.max_pool2d(
        &window,
        &strides,
        &[0, 0],
        &[1, 1],
        false,
    ))
}

fn eval_gather(node: &NodeSpec, inputs: &[ValueTensor]) -> Result<Tensor> {
    ensure_inputs("gather", inputs, 2)?;
    let data = inputs[0].tensor.shallow_clone();
    let indices = inputs[1].tensor.shallow_clone();
    let axis = attr_i64(&node.attrs, "axis")?.unwrap_or(0);
    let axis = normalize_axis(axis, data.dim() as i64)?;
    let axis_len = data.size()[axis as usize];
    if axis_len == 0 {
        return Err(anyhow!("gather axis dimension is zero"));
    }
    let mut idx = indices
        .to_kind(Kind::Int64)
        .clamp(0, axis_len - 1);
    let idx_shape = idx.size();
    let data_shape = data.size();
    if idx_shape.is_empty() {
        let scalar_index = idx.int64_value(&[]);
        return data
            .f_select(axis, scalar_index)
            .context("gather select failed");
    }
    idx = idx.reshape(&[-1]);
    let gathered = data
        .f_index_select(axis, &idx)
        .context("gather index_select failed")?;
    let axis_usize = axis as usize;
    let mut output_shape =
        Vec::with_capacity(data_shape.len() - 1 + idx_shape.len());
    output_shape.extend_from_slice(&data_shape[..axis_usize]);
    output_shape.extend(idx_shape.iter().copied());
    if axis_usize + 1 < data_shape.len() {
        output_shape.extend_from_slice(&data_shape[axis_usize + 1..]);
    }
    if output_shape.is_empty() {
        return Ok(gathered.reshape(&[]));
    }
    Ok(gathered.reshape(&output_shape))
}

fn eval_slice(node: &NodeSpec, inputs: &[ValueTensor]) -> Result<Tensor> {
    ensure_inputs("slice", inputs, 1)?;
    let mut tensor = inputs[0].tensor.shallow_clone();
    let starts = attr_list_i64(&node.attrs, "starts")?.unwrap_or_else(|| Vec::new());
    let sizes = attr_list_i64(&node.attrs, "sizes")?.unwrap_or_else(|| Vec::new());
    let strides = attr_list_i64(&node.attrs, "strides")?.unwrap_or_else(|| Vec::new());
    if starts.is_empty() {
        return Ok(tensor);
    }
    if starts.len() != sizes.len() {
        return Err(anyhow!("slice starts/sizes length mismatch"));
    }
    for (dim, (start, size)) in starts.iter().zip(sizes.iter()).enumerate() {
        tensor = tensor
            .f_narrow(dim as i64, *start, *size)
            .context("slice narrow failed")?;
        let step = if dim < strides.len() { strides[dim] } else { 1 };
        if step > 1 {
            let idx = Tensor::arange(
                *size,
                (Kind::Int64, tensor.device()),
            )
            .f_slice(0, 0, *size, step)
            .context("slice stride index failed")?;
            tensor = tensor
                .f_index_select(dim as i64, &idx)
                .context("slice stride select failed")?;
        }
    }
    Ok(tensor)
}

fn apply_clamp(
    mut tensor: Tensor,
    dtype: DataType,
    mut min: Option<f64>,
    mut max: Option<f64>,
) -> Result<Tensor> {
    if let Some((lo, hi)) = dtype.int_bounds() {
        if let Some(m) = min {
            min = Some(m.max(lo as f64));
        }
        if let Some(m) = max {
            max = Some(m.min(hi as f64));
        }
    }
    if let Some(m) = min {
        let scalar = scalar_tensor(m, dtype, tensor.device())?;
        tensor = tensor
            .f_max_other(&scalar)
            .context("clamp min failed")?;
    }
    if let Some(m) = max {
        let scalar = scalar_tensor(m, dtype, tensor.device())?;
        tensor = tensor
            .f_min_other(&scalar)
            .context("clamp max failed")?;
    }
    Ok(tensor)
}

fn ensure_inputs(op: &str, inputs: &[ValueTensor], expected: usize) -> Result<()> {
    if inputs.len() != expected {
        Err(anyhow!(
            "Op '{}' expected {} inputs, got {}",
            op,
            expected,
            inputs.len()
        ))
    } else {
        Ok(())
    }
}

fn attr_f64(attrs: &Value, key: &str) -> Result<Option<f64>> {
    match attrs.get(key) {
        Some(Value::Number(num)) => Ok(num.as_f64()),
        Some(Value::Null) | None => Ok(None),
        Some(other) => Err(anyhow!("Attribute '{}' expected number, got {}", key, other)),
    }
}

fn attr_i64(attrs: &Value, key: &str) -> Result<Option<i64>> {
    match attrs.get(key) {
        Some(Value::Number(num)) => Ok(num.as_i64()),
        Some(Value::Null) | None => Ok(None),
        Some(Value::String(s)) if s == "None" => Ok(None),
        Some(other) => Err(anyhow!("Attribute '{}' expected int, got {}", key, other)),
    }
}

fn attr_list_i64(attrs: &Value, key: &str) -> Result<Option<Vec<i64>>> {
    match attrs.get(key) {
        Some(Value::Array(arr)) => {
            let mut values = Vec::with_capacity(arr.len());
            for v in arr {
                if let Some(n) = v.as_i64() {
                    values.push(n);
                } else {
                    return Err(anyhow!(
                        "Attribute '{}' expected integer list, got {:?}",
                        key,
                        v
                    ));
                }
            }
            Ok(Some(values))
        }
        Some(Value::Null) | None => Ok(None),
        Some(other) => Err(anyhow!(
            "Attribute '{}' expected list, got {}",
            key,
            other
        )),
    }
}

fn normalize_axis(axis: i64, rank: i64) -> Result<i64> {
    if rank == 0 {
        return Err(anyhow!("Cannot normalize axis for zero-rank tensor"));
    }
    let mut ax = axis;
    if ax < 0 {
        ax += rank;
    }
    if ax < 0 || ax >= rank {
        Err(anyhow!("Axis {} out of range for rank {}", axis, rank))
    } else {
        Ok(ax)
    }
}

fn resolve_padding(
    attrs: &Value,
    input: &Tensor,
    filter: &Tensor,
    strides: &[i64],
) -> Result<Vec<i64>> {
    resolve_padding_with_kernel(
        attrs.get("padding"),
        input.size()[2],
        input.size()[3],
        filter.size()[2],
        filter.size()[3],
        strides,
    )
}

fn resolve_padding_with_kernel(
    padding: Option<&Value>,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    strides: &[i64],
) -> Result<Vec<i64>> {
    match padding {
        Some(Value::Array(arr)) => {
            let vals = arr
                .iter()
                .map(|v| v.as_i64().ok_or_else(|| anyhow!("Padding must be integers")))
                .collect::<Result<Vec<_>>>()?;
            if vals.len() != 4 {
                return Err(anyhow!("Padding list must have 4 elements"));
            }
            Ok(vals)
        }
        Some(Value::String(kind)) => match kind.as_str() {
            "valid" | "none" => Ok(vec![0, 0, 0, 0]),
            "same" | "same-upper" | "same-lower" => {
                let sh = strides[0];
                let sw = strides[1];
                let out_h = (h + sh - 1) / sh;
                let out_w = (w + sw - 1) / sw;
                let pad_h = ((out_h - 1) * sh + kh - h).max(0);
                let pad_w = ((out_w - 1) * sw + kw - w).max(0);
                let top = pad_h / 2;
                let bottom = pad_h - top;
                let left = pad_w / 2;
                let right = pad_w - left;
                Ok(vec![left, right, top, bottom])
            }
            other => Err(anyhow!("Unsupported padding string: {}", other)),
        },
        Some(Value::Null) | None => Ok(vec![0, 0, 0, 0]),
        Some(other) => Err(anyhow!("Invalid padding specification: {}", other)),
    }
}

#[pyfunction]
fn execute(graph_json: &str, feeds_json: &str) -> PyResult<String> {
    let graph: GraphSpec = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("graph parse error: {e}")))?;
    let feeds: HashMap<String, FeedTensor> = serde_json::from_str(feeds_json)
        .map_err(|e| PyRuntimeError::new_err(format!("feeds parse error: {e}")))?;
    let outputs =
        run_graph(&graph, &feeds).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    serde_json::to_string(&outputs)
        .map_err(|e| PyRuntimeError::new_err(format!("serialize error: {e}")))
}

#[pymodule]
fn pywebnn_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute, m)?)?;
    Ok(())
}
