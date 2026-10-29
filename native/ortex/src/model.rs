//! Session resources own their environment and serialize runs through a mutex.
use crate::tensor::OrtexTensor;
use crate::utils::{map_eps, map_opt_level};
use crate::values::InputValue;
use ort::environment::{Environment, GlobalThreadPoolOptions};
use ort::session::{builder::SessionBuilder, Session, SessionInputValue};
use ort::session::{OutputSelector, RunOptions};
use ort::value::{DynValue, ValueType};
use ort::Error;
use rustler::{Atom, Env, NifMap, Resource, ResourceArc};
use std::sync::Mutex;

pub type Description = Vec<(String, String, Option<Vec<i64>>)>;
pub type Outputs = Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>;
pub type NamedOutputs = Vec<(String, (ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize))>;

#[derive(NifMap, Default)]
pub struct InferenceOptions {
    outputs: Option<Vec<String>>,
    tag: Option<String>,
}

pub struct OrtexEnvironment {
    environment: Environment,
    global_pool: bool,
}
// Resource encoding only copies the resource handle; it cannot mutate ort's
// environment or its internal callback state across an unwind.
impl std::panic::RefUnwindSafe for OrtexEnvironment {}
#[rustler::resource_impl(name = "OrtexEnvironmentV1")]
impl Resource for OrtexEnvironment {}

pub struct OrtexModel {
    pub(crate) session: Mutex<Session>,
    // Metadata is immutable and does not need to wait for an inference lock.
    inputs: Description,
    outputs: Description,
}
#[rustler::resource_impl(name = "OrtexModelV2")]
impl Resource for OrtexModel {}

#[derive(NifMap)]
pub struct PoolOptions {
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    spinning: Option<bool>,
}

pub fn environment(options: Option<PoolOptions>) -> Result<OrtexEnvironment, String> {
    let global_pool = options.is_some();
    #[cfg(feature = "load-dynamic")]
    let builder = {
        let path = std::env::var_os("ORT_DYLIB_PATH")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                )
                .into()
            });
        ort::init_from(std::path::PathBuf::from(path))
            .map_err(|e| format!("Failed to load ONNX Runtime: {e}"))?
    };
    #[cfg(not(feature = "load-dynamic"))]
    let builder = ort::init();
    // ort::Error::new itself calls the runtime API, so loader failures above
    // must remain plain strings until the library has initialized successfully.
    (|| -> Result<OrtexEnvironment, Error> {
        let mut builder = builder.with_name("Ortex");
        if let Some(options) = options {
            let mut pool = GlobalThreadPoolOptions::default();
            if let Some(n) = options.intra_threads {
                pool = pool.with_intra_threads(n)?;
            }
            if let Some(n) = options.inter_threads {
                pool = pool.with_inter_threads(n)?;
            }
            if let Some(spin) = options.spinning {
                pool = pool.with_spin_control(spin)?;
            }
            builder = builder.with_global_thread_pool(pool);
        }
        Ok(OrtexEnvironment {
            environment: builder.build()?,
            global_pool,
        })
    })()
    .map_err(|e| e.to_string())
}

#[derive(NifMap)]
pub struct SessionOptions {
    providers: Vec<Atom>,
    optimization_level: i32,
    error_on_failure: bool,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    parallel_execution: Option<bool>,
    intra_op_spinning: Option<bool>,
    inter_op_spinning: Option<bool>,
    independent_thread_pool: bool,
    profiling: Option<String>,
    optimized_model_path: Option<String>,
    deterministic_compute: Option<bool>,
    memory_pattern: Option<bool>,
    log_id: Option<String>,
}

fn session_builder(
    env: Env,
    environment: ResourceArc<OrtexEnvironment>,
    opts: SessionOptions,
) -> Result<SessionBuilder, Error> {
    if environment.global_pool
        && !opts.independent_thread_pool
        && (opts.intra_threads.is_some()
            || opts.inter_threads.is_some()
            || opts.intra_op_spinning.is_some()
            || opts.inter_op_spinning.is_some())
    {
        return Err(Error::new("session thread counts/spinning require independent_thread_pool: true when using a global pool"));
    }
    let providers = map_eps(env, opts.providers, opts.error_on_failure)?;
    let mut builder = Session::builder(&environment.environment)?
        .with_optimization_level(map_opt_level(opts.optimization_level))?
        .with_execution_providers(providers)?;
    if opts.independent_thread_pool {
        builder = builder.with_independent_thread_pool()?;
    }
    if let Some(n) = opts.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    if let Some(n) = opts.inter_threads {
        builder = builder.with_inter_threads(n)?;
    }
    if let Some(v) = opts.parallel_execution {
        builder = builder.with_parallel_execution(v)?;
    }
    if let Some(v) = opts.intra_op_spinning {
        builder = builder.with_intra_op_spinning(v)?;
    }
    if let Some(v) = opts.inter_op_spinning {
        builder = builder.with_inter_op_spinning(v)?;
    }
    if let Some(v) = opts.profiling {
        builder = builder.with_profiling(v)?;
    }
    if let Some(v) = opts.optimized_model_path {
        builder = builder.with_optimized_model_path(v)?;
    }
    if let Some(v) = opts.deterministic_compute {
        builder = builder.with_deterministic_compute(v)?;
    }
    if let Some(v) = opts.memory_pattern {
        builder = builder.with_memory_pattern(v)?;
    }
    if let Some(v) = opts.log_id {
        builder = builder.with_log_id(v)?;
    }
    Ok(builder)
}

pub fn init(
    env: Env,
    path: String,
    environment: ResourceArc<OrtexEnvironment>,
    opts: SessionOptions,
) -> Result<OrtexModel, Error> {
    let mut builder = session_builder(env, environment, opts)?;
    Ok(from_session(builder.commit_from_file(path)?))
}

pub fn init_from_memory(
    env: Env,
    bytes: &[u8],
    environment: ResourceArc<OrtexEnvironment>,
    opts: SessionOptions,
) -> Result<OrtexModel, Error> {
    let mut builder = session_builder(env, environment, opts)?;
    // The copying constructor owns the graph after this call; it does not retain
    // a borrow into the caller's BEAM binary.
    Ok(from_session(builder.commit_from_memory(bytes)?))
}

fn from_session(session: Session) -> OrtexModel {
    let describe = |outlets: &[ort::value::Outlet]| {
        outlets
            .iter()
            .map(|o| {
                let dims = match o.dtype() {
                    ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
                    _ => None,
                };
                (o.name().to_owned(), format!("{:#?}", o.dtype()), dims)
            })
            .collect()
    };
    OrtexModel {
        inputs: describe(session.inputs()),
        outputs: describe(session.outputs()),
        session: Mutex::new(session),
    }
}

pub fn show(model: ResourceArc<OrtexModel>) -> (Description, Description) {
    (model.inputs.clone(), model.outputs.clone())
}

pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Outputs, Error> {
    run_with_options(model, inputs, InferenceOptions::default())
}

pub fn run_with_options(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
    options: InferenceOptions,
) -> Result<Outputs, Error> {
    let inputs = positional_inputs(&model, inputs)?;
    run_named_with_options(model, inputs, options)
        .map(|outputs| outputs.into_iter().map(|(_, tensor)| tensor).collect())
}

pub fn positional_inputs<T>(model: &OrtexModel, inputs: Vec<T>) -> Result<Vec<(String, T)>, Error> {
    if inputs.len() != model.inputs.len() {
        return Err(Error::new(format!(
            "Expected {} inputs, got {}",
            model.inputs.len(),
            inputs.len()
        )));
    }
    Ok(model
        .inputs
        .iter()
        .zip(inputs)
        .map(|(descriptor, input)| (descriptor.0.clone(), input))
        .collect())
}

pub fn run_named(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<(String, ResourceArc<OrtexTensor>)>,
) -> Result<NamedOutputs, Error> {
    run_named_with_options(model, inputs, InferenceOptions::default())
}

pub fn run_named_with_options(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<(String, ResourceArc<OrtexTensor>)>,
    options: InferenceOptions,
) -> Result<NamedOutputs, Error> {
    let inputs = inputs
        .into_iter()
        .map(|(name, input)| (name, InputValue::Tensor(input)))
        .collect();
    infer(model, inputs, options, |value| {
        let tensor: OrtexTensor = value.try_into()?;
        let shape = tensor.shape();
        let (dtype, bits) = tensor.dtype();
        Ok((ResourceArc::new(tensor), shape, dtype, bits))
    })
}

pub fn infer<T>(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<(String, InputValue)>,
    options: InferenceOptions,
    mut decode: impl FnMut(&DynValue) -> Result<T, Error>,
) -> Result<Vec<(String, T)>, Error> {
    let mut session = model
        .session
        .lock()
        .map_err(|_| Error::new("Ortex session mutex poisoned; reload the model"))?;
    if inputs.len() != session.inputs().len() {
        return Err(Error::new(format!(
            "Expected {} inputs, got {}",
            session.inputs().len(),
            inputs.len()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    let mut values: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(inputs.len());
    for (name, input) in &inputs {
        if !seen.insert(name) {
            return Err(Error::new(format!("Duplicate input {name:?}")));
        }
        let descriptor = session
            .inputs()
            .iter()
            .find(|d| d.name() == name)
            .ok_or_else(|| Error::new(format!("Unknown input {name:?}")))?;
        let value = input.prepare(descriptor.dtype())?;
        values.push((name.clone(), value));
    }
    let names: Vec<String> = options.outputs.unwrap_or_else(|| {
        session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect()
    });
    if names.is_empty() {
        return Err(Error::new("outputs must be non-empty"));
    }
    let mut seen = std::collections::HashSet::new();
    for name in &names {
        if !seen.insert(name) {
            return Err(Error::new(format!("Duplicate output {name:?}")));
        }
        if !session.outputs().iter().any(|d| d.name() == name) {
            return Err(Error::new(format!("Unknown output {name:?}")));
        }
    }
    let mut run_options = RunOptions::new()?;
    if let Some(tag) = options.tag {
        run_options = run_options.with_tag(tag)?;
    }
    let run_options = run_options.with_outputs(
        names
            .iter()
            .fold(OutputSelector::no_default(), |selector, name| {
                selector.with(name)
            }),
    );
    let outputs = session.run_with_options(values, &run_options)?;
    names
        .iter()
        .map(|name| {
            let value = outputs
                .get(name)
                .ok_or_else(|| Error::new(format!("Missing output {name}")))?;
            Ok((name.clone(), decode(value)?))
        })
        .collect()
}
