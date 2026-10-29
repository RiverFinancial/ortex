//! Copy output values into BEAM-owned terms before releasing the session lock.
use crate::tensor::OrtexTensor;
use ort::value::{
    DynValue, DynValueTypeMarker, IntoTensorElementType, TensorElementType, ValueType,
};
use ort::Error;
use rustler::{Encoder, Env, ResourceArc, Term};
use std::hash::Hash;

mod atoms {
    rustler::atoms! { tensor, sequence, map }
}

pub fn decode<'a>(env: Env<'a>, value: &DynValue) -> Result<Term<'a>, Error> {
    match value.dtype() {
        ValueType::Tensor { .. } => {
            let tensor: OrtexTensor = value.try_into()?;
            let shape = tensor.shape();
            let (dtype, bits) = tensor.dtype();
            Ok((
                atoms::tensor(),
                (ResourceArc::new(tensor), shape, dtype, bits),
            )
                .encode(env))
        }
        ValueType::Sequence(_) => {
            let values = value.try_extract_sequence::<DynValueTypeMarker>()?;
            let terms = values
                .iter()
                .map(|v| decode(env, v))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((atoms::sequence(), terms).encode(env))
        }
        ValueType::Map { key, value: ty } => {
            let map = match key {
                TensorElementType::Int64 => map_values::<i64>(env, value, *ty)?,
                TensorElementType::String => map_values::<String>(env, value, *ty)?,
                _ => return Err(Error::new(format!("unsupported map key type: {key}"))),
            };
            Ok((atoms::map(), map).encode(env))
        }
        other => Err(Error::new(format!("unsupported output type: {other}"))),
    }
}

fn map_values<'a, K>(
    env: Env<'a>,
    value: &DynValue,
    ty: TensorElementType,
) -> Result<Term<'a>, Error>
where
    K: IntoTensorElementType + Clone + Hash + Eq + Encoder,
{
    let mut result = Term::map_new(env);
    macro_rules! extract {
        ($t:ty) => {{
            for (key, value) in value.try_extract_key_values::<K, $t>()? {
                result = result
                    .map_put(key, value)
                    .map_err(|_| Error::new("failed to encode map output"))?;
            }
        }};
    }
    macro_rules! extract_float {
        ($t:ty) => {{
            for (key, value) in value.try_extract_key_values::<K, $t>()? {
                if !value.is_finite() {
                    return Err(Error::new(
                        "non-finite map values cannot be represented as BEAM floats",
                    ));
                }
                result = result
                    .map_put(key, value)
                    .map_err(|_| Error::new("failed to encode map output"))?;
            }
        }};
    }
    match ty {
        TensorElementType::Float32 => extract_float!(f32),
        TensorElementType::Float64 => extract_float!(f64),
        TensorElementType::Int8 => extract!(i8),
        TensorElementType::Int16 => extract!(i16),
        TensorElementType::Int32 => extract!(i32),
        TensorElementType::Int64 => extract!(i64),
        TensorElementType::Uint8 => extract!(u8),
        TensorElementType::Uint16 => extract!(u16),
        TensorElementType::Uint32 => extract!(u32),
        TensorElementType::Uint64 => extract!(u64),
        TensorElementType::Bool => extract!(bool),
        _ => return Err(Error::new(format!("unsupported map value type: {ty}"))),
    }
    Ok(result)
}
