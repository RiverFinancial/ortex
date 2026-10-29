//! Provider configuration keeps the legacy atom form and adds validated option maps.
use crate::constants::ortex_atoms as atoms;
use ort::ep::{self, ExecutionProvider, ExecutionProviderDispatch};
use ort::Error;
use rustler::{Atom, Env, NifUntaggedEnum};
use std::collections::HashMap;

#[derive(NifUntaggedEnum)]
pub enum ProviderConfig {
    Name(Atom),
    Config((Atom, HashMap<String, String>)),
}

fn boolean(key: &str, value: &str) -> Result<bool, Error> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(Error::new(format!(
            "provider option {key:?} requires true/false or 1/0"
        ))),
    }
}

fn known(options: &HashMap<String, String>, allowed: &[&str]) -> Result<(), Error> {
    for (key, value) in options {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::new(format!("unknown provider option {key:?}")));
        }
        if key.contains('\0') || value.contains('\0') {
            return Err(Error::new("provider options must not contain NUL bytes"));
        }
    }
    Ok(())
}

#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "coreml",
    feature = "onednn",
    feature = "rocm"
))]
fn configured<T>(
    options: HashMap<String, String>,
    allowed: &[&str],
) -> Result<ExecutionProviderDispatch, Error>
where
    T: ep::ArbitrarilyConfigurableExecutionProvider + Default + Into<ExecutionProviderDispatch>,
{
    known(&options, allowed)?;
    let mut provider = T::default();
    for (key, value) in options {
        provider = provider.with_arbitrary_config(key, value);
    }
    Ok(provider.into())
}

pub fn map_eps(
    env: Env,
    providers: Vec<ProviderConfig>,
    fail: bool,
) -> Result<Vec<ExecutionProviderDispatch>, Error> {
    providers
        .into_iter()
        .map(|config| {
            let (name, options) = match config {
                ProviderConfig::Name(name) => (name, HashMap::new()),
                ProviderConfig::Config(pair) => pair,
            };
            let name = name
                .to_term(env)
                .atom_to_string()
                .map_err(|_| Error::new("invalid provider atom"))?;
            let provider = match name.as_str() {
                "cpu" => {
                    known(&options, &["arena_allocator"])?;
                    let mut provider = ep::CPU::default();
                    if let Some(value) = options.get("arena_allocator") {
                        provider =
                            provider.with_arena_allocator(boolean("arena_allocator", value)?);
                    }
                    provider.build()
                }
                #[cfg(feature = "cuda")]
                "cuda" => configured::<ep::CUDA>(
                    options,
                    &[
                        "device_id",
                        "gpu_mem_limit",
                        "arena_extend_strategy",
                        "cudnn_conv_algo_search",
                        "cudnn_conv_use_max_workspace",
                        "cudnn_conv1d_pad_to_nc1d",
                        "enable_cuda_graph",
                        "use_tf32",
                        "do_copy_in_default_stream",
                    ],
                )?,
                #[cfg(feature = "tensorrt")]
                "tensorrt" => configured::<ep::TensorRT>(
                    options,
                    &[
                        "device_id",
                        "trt_max_workspace_size",
                        "trt_fp16_enable",
                        "trt_int8_enable",
                        "trt_engine_cache_enable",
                        "trt_engine_cache_path",
                        "trt_timing_cache_enable",
                        "trt_timing_cache_path",
                        "trt_min_subgraph_size",
                        "trt_max_partition_iterations",
                    ],
                )?,
                #[cfg(feature = "coreml")]
                "coreml" => configured::<ep::CoreML>(
                    options,
                    &[
                        "MLComputeUnits",
                        "ModelFormat",
                        "EnableOnSubgraphs",
                        "RequireStaticInputShapes",
                        "SpecializationStrategy",
                        "ModelCacheDirectory",
                        "ProfileComputePlan",
                        "AllowLowPrecisionAccumulationOnGPU",
                    ],
                )?,
                #[cfg(feature = "onednn")]
                "onednn" => configured::<ep::OneDNN>(options, &["use_arena"])?,
                #[cfg(feature = "rocm")]
                "rocm" => configured::<ep::ROCm>(
                    options,
                    &[
                        "device_id",
                        "gpu_mem_limit",
                        "arena_extend_strategy",
                        "miopen_conv_exhaustive_search",
                        "miopen_conv_use_max_workspace",
                        "do_copy_in_default_stream",
                        "enable_hip_graph",
                        "tunable_op_enable",
                        "tunable_op_tuning_enable",
                        "tunable_op_max_tuning_duration_ms",
                    ],
                )?,
                #[cfg(feature = "acl")]
                "acl" => {
                    known(&options, &["fast_math"])?;
                    let mut provider = ep::ACL::default();
                    if let Some(value) = options.get("fast_math") {
                        provider = provider.with_fast_math(boolean("fast_math", value)?);
                    }
                    provider.build()
                }
                #[cfg(feature = "directml")]
                "directml" => {
                    known(&options, &["device_id"])?;
                    let mut provider = ep::DirectML::default();
                    if let Some(value) = options.get("device_id") {
                        let id =
                            value
                                .parse::<i32>()
                                .ok()
                                .filter(|n| *n >= 0)
                                .ok_or_else(|| {
                                    Error::new("DirectML device_id must be a non-negative i32")
                                })?;
                        provider = provider.with_device_id(id);
                    }
                    provider.build()
                }
                _ => {
                    return Err(Error::new(format!(
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

pub fn status() -> Result<Vec<(Atom, bool, Option<bool>)>, Error> {
    macro_rules! provider {
        ($name:ident, $feature:literal, $ty:ident) => {{
            #[cfg(feature = $feature)]
            let available = Some(ep::$ty::is_available()?);
            #[cfg(not(feature = $feature))]
            let available = None;
            (atoms::$name(), cfg!(feature = $feature), available)
        }};
    }
    Ok(vec![
        (atoms::cpu(), true, Some(ep::CPU::is_available()?)),
        provider!(cuda, "cuda", CUDA),
        provider!(tensorrt, "tensorrt", TensorRT),
        provider!(coreml, "coreml", CoreML),
        provider!(directml, "directml", DirectML),
        provider!(acl, "acl", ACL),
        provider!(onednn, "onednn", OneDNN),
        provider!(rocm, "rocm", ROCm),
    ])
}
