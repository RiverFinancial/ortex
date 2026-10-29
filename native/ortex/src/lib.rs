//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod metadata;
mod model;
mod providers;
mod run_control;
mod runtime_logs;
mod tensor;
mod utils;
mod values;

use tensor::OrtexTensor;

use rustler::types::Binary;
use rustler::ResourceArc;
use rustler::{Atom, Env, NifResult, Term};

#[rustler::nif(schedule = "DirtyIo")]
fn init_with_options(
    env: Env,
    model_path: String,
    environment: ResourceArc<model::OrtexEnvironment>,
    options: model::SessionOptions,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    let model = model::init(env, model_path, environment, options)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(model))
}

#[rustler::nif(schedule = "DirtyIo")]
fn init_from_memory(
    env: Env,
    bytes: Binary,
    environment: ResourceArc<model::OrtexEnvironment>,
    options: model::SessionOptions,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    model::init_from_memory(env, bytes.as_slice(), environment, options)
        .map(ResourceArc::new)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn metadata<'a>(env: Env<'a>, model: ResourceArc<model::OrtexModel>) -> NifResult<Term<'a>> {
    metadata::get(env, model)
}

#[rustler::nif(schedule = "DirtyIo")]
fn end_profiling(model: ResourceArc<model::OrtexModel>) -> NifResult<String> {
    let error = |message: String| rustler::Error::Term(Box::new(message));
    let mut session = model
        .session
        .lock()
        .map_err(|_| error("Ortex session mutex poisoned; reload the model".into()))?;
    session.end_profiling().map_err(|e| error(e.to_string()))
}

#[rustler::nif(schedule = "DirtyIo")]
fn execution_providers(
    _environment: ResourceArc<model::OrtexEnvironment>,
) -> NifResult<Vec<(Atom, bool, Option<bool>)>> {
    providers::status().map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif]
fn show_session(
    model: ResourceArc<model::OrtexModel>,
) -> NifResult<(
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
)> {
    Ok(model::show(model))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    model::run(model, inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_named(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(String, ResourceArc<OrtexTensor>)>,
) -> NifResult<model::NamedOutputs> {
    model::run_named(model, inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_with_options(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
    options: model::InferenceOptions,
) -> NifResult<model::Outputs> {
    model::run_with_options(model, inputs, options)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_named_with_options(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(String, ResourceArc<OrtexTensor>)>,
    options: model::InferenceOptions,
) -> NifResult<model::NamedOutputs> {
    model::run_named_with_options(model, inputs, options)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_values<'a>(
    env: Env<'a>,
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<values::InputValue>,
    options: model::InferenceOptions,
) -> NifResult<Vec<Term<'a>>> {
    let inputs = model::positional_inputs(&model, inputs)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    model::infer(model, inputs, options, |v| values::decode(env, v))
        .map(|outputs| outputs.into_iter().map(|(_, v)| v).collect())
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_named_values<'a>(
    env: Env<'a>,
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(String, values::InputValue)>,
    options: model::InferenceOptions,
) -> NifResult<Vec<(String, Term<'a>)>> {
    model::infer(model, inputs, options, |v| values::decode(env, v))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn from_binary(bin: Binary, shape: Term, dtype: Term) -> NifResult<ResourceArc<OrtexTensor>> {
    let shape: Vec<usize> = rustler::types::tuple::get_tuple(shape)?
        .iter()
        .map(|x| -> NifResult<usize> { Ok(x.decode::<usize>())? })
        .collect::<NifResult<Vec<usize>>>()?;
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;

    utils::from_binary(bin, shape, dtype_str, dtype_bits)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    utils::to_binary(env, reference, bits, limit)
}

#[rustler::nif]
pub fn slice(
    tensor: ResourceArc<OrtexTensor>,
    start_indicies: Vec<isize>,
    lengths: Vec<isize>,
    strides: Vec<isize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.slice(
        start_indicies,
        lengths,
        strides,
    )))
}

#[rustler::nif]
pub fn reshape(
    tensor: ResourceArc<OrtexTensor>,
    shape: Vec<usize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.reshape(shape)?))
}

#[rustler::nif]
pub fn concatenate(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: Term,
    axis: i32,
) -> NifResult<ResourceArc<OrtexTensor>> {
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;
    let concatted = tensor::concatenate(tensors, (&dtype_str, dtype_bits), axis as usize);
    Ok(ResourceArc::new(concatted))
}

#[rustler::nif(schedule = "DirtyIo")]
fn environment(
    env: Env,
    options: Option<model::PoolOptions>,
) -> NifResult<ResourceArc<model::OrtexEnvironment>> {
    model::environment(env, model::EnvironmentOptions::legacy(options, None))
        .map(ResourceArc::new)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn environment_with_logs(
    env: Env,
    options: Option<model::PoolOptions>,
    log_level: Option<i32>,
) -> NifResult<ResourceArc<model::OrtexEnvironment>> {
    model::environment(env, model::EnvironmentOptions::legacy(options, log_level))
        .map(ResourceArc::new)
        .map_err(|e| rustler::Error::Term(Box::new(e)))
}

#[rustler::nif(schedule = "DirtyIo")]
fn take_runtime_logs(
    environment: ResourceArc<model::OrtexEnvironment>,
) -> (runtime_logs::Entries, usize) {
    environment
        .logs
        .as_ref()
        .map(|logs| logs.drain())
        .unwrap_or_default()
}

rustler::init!("Elixir.Ortex.Native");

// Option preparation must not queue behind inference's dirty-I/O mutex waits.
#[rustler::nif(schedule = "DirtyCpu")]
fn new_run_control(
    model: ResourceArc<model::OrtexModel>,
    options: model::InferenceOptions,
) -> NifResult<ResourceArc<run_control::RunControl>> {
    run_control::RunControl::new(model, options)
        .map(ResourceArc::new)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

// Runtime termination only sets a flag; it must not queue behind inference NIFs
// waiting for a session mutex on the dirty scheduler pool.
#[rustler::nif]
fn cancel_run(control: ResourceArc<run_control::RunControl>) -> NifResult<Atom> {
    control
        .cancel()
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(rustler::types::atom::ok())
}

#[rustler::nif]
fn run_started(control: ResourceArc<run_control::RunControl>) -> bool {
    control.started()
}

#[rustler::nif]
fn reserve_run(
    model: ResourceArc<model::OrtexModel>,
    control: ResourceArc<run_control::RunControl>,
) -> NifResult<Atom> {
    control
        .reserve(&model)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(rustler::types::atom::ok())
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_controlled<'a>(
    env: Env<'a>,
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<values::InputValue>,
    control: ResourceArc<run_control::RunControl>,
) -> NifResult<Vec<Term<'a>>> {
    model::infer_controlled_positional(model, inputs, &control, |v| values::decode(env, v))
        .map(|outputs| outputs.into_iter().map(|(_, v)| v).collect())
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[cfg_attr(feature = "benchmark-dirty-cpu", rustler::nif(schedule = "DirtyCpu"))]
#[cfg_attr(
    not(feature = "benchmark-dirty-cpu"),
    rustler::nif(schedule = "DirtyIo")
)]
fn run_named_controlled<'a>(
    env: Env<'a>,
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(String, values::InputValue)>,
    control: ResourceArc<run_control::RunControl>,
) -> NifResult<Vec<(String, Term<'a>)>> {
    model::infer_controlled(model, inputs, &control, |v| values::decode(env, v))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn environment_with_options(
    env: Env,
    options: model::EnvironmentOptions,
) -> NifResult<ResourceArc<model::OrtexEnvironment>> {
    model::environment(env, options)
        .map(ResourceArc::new)
        .map_err(|e| rustler::Error::Term(Box::new(e)))
}

#[rustler::nif]
fn inference_scheduler() -> Atom {
    if cfg!(feature = "benchmark-dirty-cpu") {
        constants::ortex_atoms::dirty_cpu()
    } else {
        constants::ortex_atoms::dirty_io()
    }
}
