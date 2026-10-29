# Run cancellation

`Ortex.RunControl.new(model, outputs: [...], tag: "...")` prepares immutable
options for one model-bound inference attempt. Pass it with `control: control` to
`Ortex.run/3`. A different process can request termination with
`Ortex.RunControl.cancel/1`, and `started?/1` reports whether the attempt has been
claimed, including waiting for the session mutex. A claimed control cannot be
reused even if inference fails. Invalid options are rejected before an attempt.

Termination calls ONNX Runtime's run-options flag without acquiring the session
mutex or queuing on the dirty scheduler pool. It is cooperative: a kernel can
finish before observing the flag. Cancelling a control never cancels another
attempt on that session. The control owns a model reference until it is released.
A cancelled control can be cancelled repeatedly, including before execution.

The native wrapper has a narrow `unsafe impl Sync` for prepared selected-output
options. ort's generic selected-output type cannot be Sync because it permits
preallocated output buffers. Our constructor uses only output names, never
preallocates values, keeps fields private, and exposes no mutable options. Its
only concurrent mutation is the runtime's documented cross-thread termination
operation. Adding shared/preallocated values to that constructor would violate
this invariant and requires a different design.

Synthetic Loop tests wait until the native attempt is claimed, then terminate it
from concurrent processes. They also cancel a queued attempt while another holds
the model mutex, check post-cancellation scores, prevent reuse/cross-model use,
and exercise controlled string and ZipMap outputs. Both runtime 1.17.3 and 1.24.4
are covered. Long-running provider kernels still need provider-specific validation.
