# Inference scheduling decision

Keep dirty I/O scheduling for inference. A session run can wait for the per-model
mutex or the runtime's worker threads. Scheduling those waits on dirty CPU can
occupy that entire pool and delay unrelated CPU NIFs, including input conversion
and control preparation. Control creation uses dirty CPU; cancellation itself is
a short normal NIF that only sets the runtime flag.

A local Apple silicon experiment used four normal, four dirty CPU, and four dirty
I/O schedulers. It verified the loaded NIF's actual mode, checked golden tree-model
scores, and compared a 2 MiB dirty-CPU input-copy probe while 16 native calls
contended for one finite counting-Loop session. Ten rounds used 20,000 iterations
per call, with one probe after each inference pool became occupied.

| Measurement | Dirty I/O inference | Dirty CPU inference |
| --- | --- | --- |
| Tree serial calls/second | 141,233 | 153,527 |
| Tree concurrent calls/second (8 callers) | 86,062 | 86,749 |
| Tree p50 / p99, microseconds | 4 / 29 | 5 / 16 |
| Contended copy p50, microseconds | 52 | 71,779 |
| Contended copy p99, microseconds | 97 | 77,888 |

Tree throughput is similar in this short local sample. The contended probe shows
that moving mutex waits to dirty CPU can delay other native CPU work. This is a
workload-specific result, not a claim about every graph/provider. Heavy independent
CPU sessions, neural-network workloads, or accelerators need separate measurement
before changing policy. Avoiding native mutex waits through a different queueing
architecture would also change this tradeoff.

## Reproduce

Use a source checkout with a supported runtime library in `ORT_DYLIB_PATH`:

```sh
export MIX_ENV=test ORTEX_BUILD=true
export ERL_FLAGS='+S 4:4 +SDcpu 4:4 +SDio 4'
ORTEX_BENCHMARK_DIRTY_CPU=0 mix compile --force --warnings-as-errors
ORTEX_BENCHMARK_DIRTY_CPU=0 mix run bench/schedulers.exs
ORTEX_BENCHMARK_DIRTY_CPU=1 mix compile --force --warnings-as-errors
ORTEX_BENCHMARK_DIRTY_CPU=1 mix run bench/schedulers.exs
# Restore the default native artifact before ordinary use:
ORTEX_BENCHMARK_DIRTY_CPU=0 mix compile --force --warnings-as-errors
```

Each build/run uses a fresh VM. The script reads the native scheduling mode and
rejects a stale artifact. The test-only config flag enables the diagnostic Cargo
feature `benchmark-dirty-cpu`; normal and precompiled builds never enable it.
This mode is for the experiment, not a supported deployment setting. It shares
the CPU pool with control preparation, which is another reason to avoid that
policy for contended sessions. CI compile-checks the alternate feature together
with all exposed provider mappings; normal runtime tests use the default policy.
