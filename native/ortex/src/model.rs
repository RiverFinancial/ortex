//! Session resources own their environment and serialize runs through a mutex.
use crate::providers::{map_eps, ProviderConfig};
use crate::run_control::{PreparedRun, RunControl};
use crate::runtime_logs::LogBuffer;
use crate::tensor::OrtexTensor;
use crate::utils::map_opt_level;
use crate::values::InputValue;
use ort::environment::{Environment, GlobalThreadPoolOptions};
use ort::session::{builder::SessionBuilder, Session, SessionInputValue};
use ort::value::{DynValue, ValueType};
use ort::Error;
use rustler::{Atom, Env, NifMap, Resource, ResourceArc};
use std::sync::Mutex;

pub type Description = Vec<(String, String, Option<Vec<i64>>)>;
pub type Outputs = Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>;
pub type NamedOutputs = Vec<(String, (ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize))>;

#[derive(NifMap, Default)]
pub struct InferenceOptions {
    pub(crate) outputs: Option<Vec<String>>,
    pub(crate) tag: Option<String>,
}

pub struct OrtexEnvironment {
    environment: Environment,
    global_pool: bool,
    pub(crate) logs: Option<std::sync::Arc<LogBuffer>>,
}
// Resource encoding only copies the resource handle; it cannot mutate ort's
// environment or its internal callback state across an unwind.
impl std::panic::RefUnwindSafe for OrtexEnvironment {}
#[rustler::resource_impl(name = "OrtexEnvironmentV2")]
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

#[derive(NifMap)]
pub struct EnvironmentOptions {
    global_thread_pool: Option<PoolOptions>,
    log_level: Option<i32>,
    name: String,
    runtime_path: Option<String>,
    providers: Vec<ProviderConfig>,
}

impl EnvironmentOptions {
    pub fn legacy(pool: Option<PoolOptions>, log_level: Option<i32>) -> Self {
        Self {
            global_thread_pool: pool,
            log_level,
            name: "Ortex".into(),
            runtime_path: None,
            providers: vec![ProviderConfig::Name(crate::constants::ortex_atoms::cpu())],
        }
    }
}

pub fn environment(env: Env, config: EnvironmentOptions) -> Result<OrtexEnvironment, String> {
    if config.name.is_empty()
        || config.name.contains('\0')
        || config
            .runtime_path
            .as_ref()
            .is_some_and(|p| p.is_empty() || p.contains('\0'))
    {
        return Err("environment name and runtime path must be non-empty without NUL bytes".into());
    }
    let options = config.global_thread_pool;
    let level = config
        .log_level
        .map(crate::runtime_logs::level)
        .transpose()?;
    let global_pool = options.is_some();
    #[cfg(feature = "load-dynamic")]
    let builder = {
        let path = config
            .runtime_path
            .map(std::ffi::OsString::from)
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").filter(|value| !value.is_empty()))
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
    let builder = {
        if config.runtime_path.is_some() {
            return Err("runtime_path requires load-dynamic".into());
        }
        ort::init()
    };
    // ort::Error::new itself calls the runtime API, so loader failures above
    // must remain plain strings until the library has initialized successfully.
    (|| -> Result<OrtexEnvironment, Error> {
        let mut builder = builder
            .with_name(config.name)
            .with_execution_providers(map_eps(env, config.providers, true)?);
        let logs = level.map(LogBuffer::new);
        if let Some(logs) = &logs {
            builder = builder.with_logger(logs.logger());
        }
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
        let environment = builder.build()?;
        if let Some(level) = level {
            environment.set_log_level(level);
        }
        Ok(OrtexEnvironment {
            environment,
            global_pool,
            logs,
        })
    })()
    .map_err(|e| e.to_string())
}

#[derive(NifMap)]
pub struct SessionOptions {
    providers: Option<Vec<ProviderConfig>>,
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
    dimension_overrides: std::collections::HashMap<String, i64>,
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
    let mut builder = Session::builder(&environment.environment)?
        .with_optimization_level(map_opt_level(opts.optimization_level))?;
    if let Some(providers) = opts.providers {
        builder = builder
            .with_no_environment_execution_providers()?
            .with_execution_providers(map_eps(env, providers, opts.error_on_failure)?)?;
    }
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
    for (name, size) in opts.dimension_overrides {
        if name.is_empty() || name.contains('\0') || size < 0 {
            return Err(Error::new(
                "dimension overrides require non-empty names and non-negative i64 sizes",
            ));
        }
        builder = builder.with_dimension_override(name, size)?;
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
    decode: impl FnMut(&DynValue) -> Result<T, Error>,
) -> Result<Vec<(String, T)>, Error> {
    let prepared = PreparedRun::new(&model, options)?;
    infer_prepared(model, inputs, &prepared, decode)
}

pub fn infer_controlled<T>(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<(String, InputValue)>,
    control: &RunControl,
    decode: impl FnMut(&DynValue) -> Result<T, Error>,
) -> Result<Vec<(String, T)>, Error> {
    control.claim(&model)?;
    infer_prepared(model, inputs, control.prepared(), decode)
}

pub fn infer_controlled_positional<T>(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<InputValue>,
    control: &RunControl,
    decode: impl FnMut(&DynValue) -> Result<T, Error>,
) -> Result<Vec<(String, T)>, Error> {
    control.claim(&model)?;
    let inputs = positional_inputs(&model, inputs)?;
    infer_prepared(model, inputs, control.prepared(), decode)
}

pub(crate) fn output_names(model: &OrtexModel) -> Vec<String> {
    model.outputs.iter().map(|d| d.0.clone()).collect()
}

fn infer_prepared<T>(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<(String, InputValue)>,
    prepared: &PreparedRun,
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
    let outputs = session.run_with_options(values, prepared.options())?;
    prepared
        .names()
        .iter()
        .map(|name| {
            let value = outputs
                .get(name)
                .ok_or_else(|| Error::new(format!("Missing output {name}")))?;
            Ok((name.clone(), decode(value)?))
        })
        .collect()
}
