//! Immutable, one-attempt run options with thread-safe runtime termination.
use crate::model::{self, InferenceOptions, OrtexModel};
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
use ort::Error;
use rustler::{Resource, ResourceArc};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct PreparedRun {
    names: Vec<String>,
    options: RunOptions<HasSelectedOutputs>,
}

// SAFETY: ort restricts selected RunOptions' Sync because selectors may contain
// mutable preallocated outputs. This private constructor only selects names;
// it never preallocates values. No mutable options/selector access is exposed.
// After publication the only operation besides immutable inference reads is
// terminate(), explicitly designed by ONNX Runtime for cross-thread calls.
unsafe impl Sync for PreparedRun {}

impl PreparedRun {
    pub fn new(model: &OrtexModel, options: InferenceOptions) -> Result<Self, Error> {
        let tag = options.tag;
        let declared = model::output_names(model);
        let names = options.outputs.unwrap_or_else(|| declared.clone());
        if names.is_empty() {
            return Err(Error::new("outputs must be non-empty"));
        }
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            if !seen.insert(name) {
                return Err(Error::new(format!("Duplicate output {name:?}")));
            }
            if !declared.contains(name) {
                return Err(Error::new(format!("Unknown output {name:?}")));
            }
        }
        let mut options = RunOptions::new()?;
        if let Some(tag) = tag {
            options = options.with_tag(tag)?;
        }
        // Tag configuration is completed before options are shared.
        let selector = names
            .iter()
            .fold(OutputSelector::no_default(), |s, name| s.with(name));
        Ok(Self {
            names,
            options: options.with_outputs(selector),
        })
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn options(&self) -> &RunOptions<HasSelectedOutputs> {
        &self.options
    }
}

pub struct RunControl {
    model: ResourceArc<OrtexModel>,
    prepared: PreparedRun,
    claimed: AtomicBool,
}
// Encoding only copies the opaque resource handle; it cannot mutate run options.
impl std::panic::RefUnwindSafe for RunControl {}
#[rustler::resource_impl(name = "OrtexRunControlV1")]
impl Resource for RunControl {}

impl RunControl {
    pub fn new(model: ResourceArc<OrtexModel>, options: InferenceOptions) -> Result<Self, Error> {
        let prepared = PreparedRun::new(&model, options)?;
        Ok(Self {
            model,
            prepared,
            claimed: AtomicBool::new(false),
        })
    }
    pub fn claim(&self, model: &OrtexModel) -> Result<(), Error> {
        if !std::ptr::eq(&*self.model, model) {
            return Err(Error::new("run control belongs to another model"));
        }
        self.claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::new("run control has already been used"))?;
        Ok(())
    }
    pub fn prepared(&self) -> &PreparedRun {
        &self.prepared
    }
    pub fn cancel(&self) -> Result<(), Error> {
        self.prepared.options.terminate()
    }
    pub fn started(&self) -> bool {
        self.claimed.load(Ordering::Acquire)
    }
}
