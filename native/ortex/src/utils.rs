//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{ArrayD, IxDyn};
use std::convert::TryInto;

use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel};

// BEAM sub-binaries are not guaranteed to be aligned. Decode into owned storage
// rather than manufacturing a mutable ndarray view over an immutable binary.
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<OrtexTensor>, String> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| "tensor shape overflow".to_owned())?;
    let bytes = dtype_bits
        .checked_div(8)
        .and_then(|size| count.checked_mul(size))
        .ok_or_else(|| "tensor byte size overflow".to_owned())?;
    if bytes != bin.len() {
        return Err("binary size does not match tensor shape and type".to_owned());
    }
    macro_rules! decode {
        ($ty:ty, $variant:ident, $convert:expr) => {{
            let data = bin
                .as_slice()
                .chunks_exact(std::mem::size_of::<$ty>())
                .map(|chunk| {
                    let v = <$ty>::from_ne_bytes(chunk.try_into().expect("fixed-size chunk"));
                    ($convert)(v)
                })
                .collect();
            let array = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?;
            Ok(ResourceArc::new(OrtexTensor::$variant(array)))
        }};
    }
    match (dtype_str.as_str(), dtype_bits) {
        ("bf", 16) => decode!(u16, bf16, half::bf16::from_bits),
        ("f", 16) => decode!(u16, f16, half::f16::from_bits),
        ("f", 32) => decode!(f32, f32, |v| v),
        ("f", 64) => decode!(f64, f64, |v| v),
        ("s", 8) => decode!(i8, s8, |v| v),
        ("s", 16) => decode!(i16, s16, |v| v),
        ("s", 32) => decode!(i32, s32, |v| v),
        ("s", 64) => decode!(i64, s64, |v| v),
        ("u", 8) => decode!(u8, u8, |v| v),
        ("u", 16) => decode!(u16, u16, |v| v),
        ("u", 32) => decode!(u32, u32, |v| v),
        ("u", 64) => decode!(u64, u64, |v| v),
        _ => Err("unsupported tensor type".to_owned()),
    }
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    _bits: usize,
    _limit: usize,
) -> NifResult<Binary<'a>> {
    Ok(reference.make_binary(env, |x| x.to_bytes()))
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn map_eps(env: rustler::env::Env, eps: Vec<Atom>) -> Vec<ExecutionProviderDispatch> {
    eps.iter()
        .map(|e| match &e.to_term(env).atom_to_string().unwrap()[..] {
            CPU => ort::CPUExecutionProvider::default().build(),
            CUDA => ort::CUDAExecutionProvider::default().build(),
            TENSORRT => ort::TensorRTExecutionProvider::default().build(),
            ACL => ort::ACLExecutionProvider::default().build(),
            ONEDNN => ort::OneDNNExecutionProvider::default().build(),
            COREML => ort::CoreMLExecutionProvider::default().build(),
            DIRECTML => ort::DirectMLExecutionProvider::default().build(),
            ROCM => ort::ROCmExecutionProvider::default().build(),
            _ => ort::CPUExecutionProvider::default().build(),
        })
        .collect()
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}

pub fn is_bool_input(inp: &ort::ValueType) -> bool {
    match inp {
        ort::ValueType::Tensor { ty, .. } => ty == &ort::TensorElementType::Bool,
        ort::ValueType::Map { value, .. } => value == &ort::TensorElementType::Bool,
        ort::ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ort::ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}
