//! Session resources own their environment and serialize runs through a mutex.
use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_eps, map_opt_level};
use ort::environment::Environment;
use ort::session::{Session, SessionInputValue};
use ort::value::ValueType;
use ort::Error;
use rustler::{Atom, Env, Resource, ResourceArc};
use std::sync::Mutex;

pub type Description = Vec<(String, String, Option<Vec<i64>>)>;
pub type Outputs = Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>;

pub struct OrtexEnvironment {
    environment: Environment,
}
// Resource encoding only copies the resource handle; it cannot mutate ort's
// environment or its internal callback state across an unwind.
impl std::panic::RefUnwindSafe for OrtexEnvironment {}
#[rustler::resource_impl(name = "OrtexEnvironmentV1")]
impl Resource for OrtexEnvironment {}

pub struct OrtexModel {
    session: Mutex<Session>,
    // Metadata is immutable and does not need to wait for an inference lock.
    inputs: Description,
    outputs: Description,
}
#[rustler::resource_impl(name = "OrtexModelV2")]
impl Resource for OrtexModel {}

pub fn environment() -> Result<OrtexEnvironment, String> {
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
        let builder = builder.with_name("Ortex");

        Ok(OrtexEnvironment {
            environment: builder.build()?,
        })
    })()
    .map_err(|e| e.to_string())
}

pub fn init(
    env: Env,
    path: String,
    environment: ResourceArc<OrtexEnvironment>,
    providers: Vec<Atom>,
    optimization: i32,
) -> Result<OrtexModel, Error> {
    let providers = map_eps(env, providers, false)?;
    let mut builder = Session::builder(&environment.environment)?
        .with_optimization_level(map_opt_level(optimization))?
        .with_execution_providers(providers)?;
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
