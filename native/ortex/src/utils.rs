//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::tensor::OrtexTensor;
use ndarray::{ArrayD, IxDyn};

use rustler::types::Binary;
use rustler::ResourceArc;
use rustler::{Atom, Env, NifResult};

use ort::{ep::ExecutionProviderDispatch, session::builder::GraphOptimizationLevel};

// BEAM sub-binaries are not guaranteed to be aligned. Decode into owned storage
// rather than manufacturing a mutable ndarray view over an immutable binary.
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<OrtexTensor>, ort::Error> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| ort::Error::new("tensor shape overflow"))?;
    let bytes = dtype_bits
        .checked_div(8)
        .and_then(|size| count.checked_mul(size))
        .ok_or_else(|| ort::Error::new("tensor byte size overflow"))?;
    if bytes != bin.len() {
        return Err(ort::Error::new(
            "binary size does not match tensor shape and type",
        ));
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
            let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| ort::Error::new(e.to_string()))?;
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
        _ => Err(ort::Error::new("unsupported tensor type")),
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
pub fn map_eps(
    env: rustler::env::Env,
    eps: Vec<Atom>,
    fail: bool,
) -> Result<Vec<ExecutionProviderDispatch>, ort::Error> {
    eps.iter()
        .map(|e| {
            let name = e
                .to_term(env)
                .atom_to_string()
                .map_err(|_| ort::Error::new("invalid provider atom"))?;
            let provider = match name.as_str() {
                "cpu" => ort::ep::CPU::default().build(),
                #[cfg(feature = "cuda")]
                "cuda" => ort::ep::CUDA::default().build(),
                #[cfg(feature = "tensorrt")]
                "tensorrt" => ort::ep::TensorRT::default().build(),
                #[cfg(feature = "coreml")]
                "coreml" => ort::ep::CoreML::default().build(),
                #[cfg(feature = "directml")]
                "directml" => ort::ep::DirectML::default().build(),
                #[cfg(feature = "acl")]
                "acl" => ort::ep::ACL::default().build(),
                #[cfg(feature = "onednn")]
                "onednn" => ort::ep::OneDNN::default().build(),
                #[cfg(feature = "rocm")]
                "rocm" => ort::ep::ROCm::default().build(),
                _ => {
                    return Err(ort::Error::new(format!(
                        "unknown or uncompiled execution provider: {name}"
                    )))
                }
            };
            Ok(if fail {
                provider.error_on_failure()
            } else {
                provider
            })
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

pub fn is_bool_input(inp: &ort::value::ValueType) -> bool {
    match inp {
        ort::value::ValueType::Tensor { ty, .. } => ty == &ort::value::TensorElementType::Bool,
        ort::value::ValueType::Map { value, .. } => value == &ort::value::TensorElementType::Bool,
        ort::value::ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ort::value::ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}
