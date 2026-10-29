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

## Managed deadlines and ownership

`Ortex.run(model, inputs, timeout: milliseconds)` creates a control automatically
and raises `Ortex.TimeoutError` if caller wait reaches its deadline. An explicit
control can also be paired with a timeout. Zero expires before starting a worker;
`:infinity` is the default. Validation/control preparation precedes the deadline;
it then covers input transfer, mutex waiting, inference, and output conversion.

Finite deadlines and explicit controls use a separate native worker plus an owner
watcher. The watcher monitors the caller and worker. Caller death or a deadline
abort requests runtime termination and kills the BEAM worker; any executing NIF
retains its resources until it returns. Caller wait is bounded, while native
termination remains cooperative. An uninterruptible kernel can keep the session
occupied until it finishes. Ordinary unbounded calls without a control keep the
synchronous path and its allocation profile.

The caller owns an Erlang reply alias. On return, exception, or timeout the alias
is removed, pending reply/monitor messages are flushed, and late replies are
ignored. Worker exceptions are re-raised with their original stack. Completed
attempts do not trigger watcher cancellation, so rejecting a reused or wrong-model
control cannot cancel its legitimate in-flight attempt.

Tests exercise actual loop deadlines, caller death, deadlines behind an occupied
mutex, post-cancel inference, worker exception stacks, resource ownership, worker
termination, response/deadline races, and lack of late messages. Telemetry emits
the deadline and cancellation mode in the caller's span, with TimeoutError as an
exception event. Control preparation uses dirty CPU scheduling independently of
the dirty I/O pool used by inference mutex waits.
