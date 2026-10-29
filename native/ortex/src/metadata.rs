//! Structured metadata uses fixed atoms; names supplied by models stay binaries.
use crate::model::OrtexModel;
use ort::value::{Outlet, TensorElementType, ValueType};
use rustler::{Encoder, Env, NifResult, ResourceArc, Term};

mod atoms {
    rustler::atoms! {
        inputs, outputs, name, r#type = "type", producer, domain,
        description, graph_description, version, custom,
        tensor, map, sequence, optional, unsupported,
        s, u, f, bf, c, boolean = "bool", string
    }
}

fn dtype<'a>(env: Env<'a>, ty: TensorElementType) -> Term<'a> {
    use TensorElementType::*;
    let numeric = match ty {
        Int8 => Some((atoms::s(), 8)),
        Int16 => Some((atoms::s(), 16)),
        Int32 => Some((atoms::s(), 32)),
        Int64 => Some((atoms::s(), 64)),
        Uint8 => Some((atoms::u(), 8)),
        Uint16 => Some((atoms::u(), 16)),
        Uint32 => Some((atoms::u(), 32)),
        Uint64 => Some((atoms::u(), 64)),
        Float16 => Some((atoms::f(), 16)),
        Float32 => Some((atoms::f(), 32)),
        Float64 => Some((atoms::f(), 64)),
        Bfloat16 => Some((atoms::bf(), 16)),
        Complex64 => Some((atoms::c(), 64)),
        Complex128 => Some((atoms::c(), 128)),
        _ => None,
    };
    if let Some(numeric) = numeric {
        return numeric.encode(env);
    }
    match ty {
        Bool => atoms::boolean().encode(env),
        String => atoms::string().encode(env),
        _ => (atoms::unsupported(), ty.to_string()).encode(env),
    }
}

fn value_type<'a>(env: Env<'a>, value: &ValueType) -> Term<'a> {
    match value {
        ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => {
            let dims: Vec<Term> = shape
                .iter()
                .enumerate()
                .map(|(i, dim)| {
                    if *dim >= 0 {
                        dim.encode(env)
                    } else if let Some(symbol) = dimension_symbols.get(i).filter(|s| !s.is_empty())
                    {
                        symbol.encode(env)
                    } else {
                        rustler::types::atom::nil().encode(env)
                    }
                })
                .collect();
            (
                atoms::tensor(),
                dtype(env, *ty),
                rustler::types::tuple::make_tuple(env, &dims),
            )
                .encode(env)
        }
        ValueType::Map { key, value } => {
            (atoms::map(), dtype(env, *key), dtype(env, *value)).encode(env)
        }
        ValueType::Sequence(inner) => (atoms::sequence(), value_type(env, inner)).encode(env),
        ValueType::Optional(inner) => (atoms::optional(), value_type(env, inner)).encode(env),
    }
}

fn outlets<'a>(env: Env<'a>, outlets: &[Outlet]) -> NifResult<Term<'a>> {
    outlets
        .iter()
        .map(|outlet| {
            Term::map_new(env)
                .map_put(atoms::name(), outlet.name())?
                .map_put(atoms::r#type(), value_type(env, outlet.dtype()))
        })
        .collect::<NifResult<Vec<_>>>()
        .map(|v| v.encode(env))
}

pub fn get<'a>(env: Env<'a>, model: ResourceArc<OrtexModel>) -> NifResult<Term<'a>> {
    let error = |message: String| rustler::Error::Term(Box::new(message));
    let session = model
        .session
        .lock()
        .map_err(|_| error("Ortex session mutex poisoned; reload the model".into()))?;
    let metadata = session.metadata().map_err(|e| error(e.to_string()))?;
    let mut custom = Term::map_new(env);
    for key in metadata.custom_keys().map_err(|e| error(e.to_string()))? {
        if let Some(value) = metadata.custom(&key) {
            custom = custom.map_put(key, value)?;
        }
    }
    Term::map_new(env)
        .map_put(atoms::inputs(), outlets(env, session.inputs())?)?
        .map_put(atoms::outputs(), outlets(env, session.outputs())?)?
        .map_put(atoms::name(), metadata.name())?
        .map_put(atoms::producer(), metadata.producer())?
        .map_put(atoms::domain(), metadata.domain())?
        .map_put(atoms::description(), metadata.description())?
        .map_put(atoms::graph_description(), metadata.graph_description())?
        .map_put(atoms::version(), metadata.version())?
        .map_put(atoms::custom(), custom)
}
