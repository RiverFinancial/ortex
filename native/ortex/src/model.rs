//! Session resources own their environment and serialize runs through a mutex.
use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_eps, map_opt_level};
use ort::environment::{Environment, GlobalThreadPoolOptions};
use ort::session::{Session, SessionInputValue};
use ort::value::ValueType;
use ort::Error;
use rustler::{Atom, Env, NifMap, Resource, ResourceArc};
use std::sync::Mutex;

pub type Description = Vec<(String, String, Option<Vec<i64>>)>;
pub type Outputs = Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>;

pub struct OrtexEnvironment {
    environment: Environment,
    global_pool: bool,
}
// Resource encoding only copies the resource handle; it cannot mutate ort's
// environment or its internal callback state across an unwind.
impl std::panic::RefUnwindSafe for OrtexEnvironment {}
#[rustler::resource_impl]
impl Resource for OrtexEnvironment {}

pub struct OrtexModel {
    session: Mutex<Session>,
    // Metadata is immutable and does not need to wait for an inference lock.
    inputs: Description,
    outputs: Description,
}
#[rustler::resource_impl]
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

pub fn init(
    env: Env,
    path: String,
    environment: ResourceArc<OrtexEnvironment>,
    opts: SessionOptions,
) -> Result<OrtexModel, Error> {
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
    let session = builder.commit_from_file(path)?;
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
    Ok(OrtexModel {
        inputs: describe(session.inputs()),
        outputs: describe(session.outputs()),
        session: Mutex::new(session),
    })
}

pub fn show(model: ResourceArc<OrtexModel>) -> (Description, Description) {
    (model.inputs.clone(), model.outputs.clone())
}

pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Outputs, Error> {
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
    let mut values: Vec<SessionInputValue<'_>> = Vec::with_capacity(inputs.len());
    for (input, descriptor) in inputs.iter().zip(session.inputs()) {
        let input: &OrtexTensor = input;
        if is_bool_input(descriptor.dtype()) {
            values.push((&input.clone().to_bool()?).try_into()?);
        } else {
            values.push(input.try_into()?);
        }
    }
    let names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_owned())
        .collect();
    let outputs = session.run(&values[..])?;
    names
        .iter()
        .map(|name| {
            let value = outputs
                .get(name)
                .ok_or_else(|| Error::new(format!("Missing output {name}")))?;
            let tensor: OrtexTensor = value.try_into()?;
            let shape = tensor.shape();
            let (dtype, bits) = tensor.dtype();
            Ok((ResourceArc::new(tensor), shape, dtype, bits))
        })
        .collect()
}
