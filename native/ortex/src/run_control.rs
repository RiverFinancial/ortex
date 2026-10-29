//! Immutable, one-attempt run options with thread-safe runtime termination.
use crate::model::{self, InferenceOptions, OrtexModel};
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
use ort::Error;
use rustler::{Resource, ResourceArc};
use std::sync::atomic::{AtomicU8, Ordering};

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
    state: AtomicU8,
}
// Encoding only copies the opaque resource handle; it cannot mutate run options.
impl std::panic::RefUnwindSafe for RunControl {}
#[rustler::resource_impl(name = "OrtexRunControlV2")]
impl Resource for RunControl {}

impl RunControl {
    pub fn new(model: ResourceArc<OrtexModel>, options: InferenceOptions) -> Result<Self, Error> {
        let prepared = PreparedRun::new(&model, options)?;
        Ok(Self {
            model,
            prepared,
            state: AtomicU8::new(0),
        })
    }
    fn check_model(&self, model: &OrtexModel) -> Result<(), Error> {
        if !std::ptr::eq(&*self.model, model) {
            return Err(Error::new("run control belongs to another model"));
        }
        Ok(())
    }
    // Reserve ownership before spawning a worker or installing its deadline.
    // A rejected duplicate must never acquire authority to cancel another call.
    pub fn reserve(&self, model: &OrtexModel) -> Result<(), Error> {
        self.check_model(model)?;
        self.state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::new("run control has already been used"))?;
        Ok(())
    }
    pub fn claim(&self, model: &OrtexModel) -> Result<(), Error> {
        self.check_model(model)?;
        // Direct internal NIF calls can enter from ready (0); public managed
        // calls enter from reserved (1). Neither can execute twice (2).
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                if state < 2 {
                    Some(2)
                } else {
                    None
                }
            })
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
        self.state.load(Ordering::Acquire) == 2
    }
}
