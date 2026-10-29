# Execution provider configuration

Provider priority follows list order. Atoms retain default configuration; a tuple
adds an option map whose keys and values are UTF-8 binaries without NUL bytes:

```elixir
model = Ortex.load("model.onnx",
  providers: [{:cpu, %{"arena_allocator" => "true"}}],
  intra_threads: 1
)
```

The same options work with `load_from_memory/2`. Unknown providers, uncompiled
providers, malformed maps, and unknown option keys raise. Keyword loading fails
on provider registration errors by default. `error_on_failure: false` explicitly
permits runtime fallback after a registration failure; it does not suppress invalid
Ortex configuration. Positional loading retains its legacy registration behavior.

Set `config :ortex, :environment, providers: [...]` before startup for VM-wide
defaults. Keyword calls inherit them when `:providers` is omitted. Explicit
session provider lists replace the defaults, rather than appending them; `[]`
suppresses environment providers and allows ONNX Runtime's CPU fallback. Inherited
environment providers register strictly. To configure session fallback explicitly,
supply the session provider list with `error_on_failure: false`. Positional calls
retain their explicit provider arguments and CPU defaults.

The precompiled NIFs support CPU registration. Accelerator providers need a source
build with the matching Cargo feature, a matching runtime library, dependencies,
and hardware. Setting a feature does not install its runtime/device dependencies.

| Provider / Cargo feature | Supported binary keys |
| --- | --- |
| `cpu` (always compiled) | `arena_allocator` |
| `cuda` | `device_id`, `gpu_mem_limit`, `arena_extend_strategy`, `cudnn_conv_algo_search`, `cudnn_conv_use_max_workspace`, `cudnn_conv1d_pad_to_nc1d`, `enable_cuda_graph`, `use_tf32`, `do_copy_in_default_stream` |
| `tensorrt` | `device_id`, `trt_max_workspace_size`, `trt_fp16_enable`, `trt_int8_enable`, `trt_engine_cache_enable`, `trt_engine_cache_path`, `trt_timing_cache_enable`, `trt_timing_cache_path`, `trt_min_subgraph_size`, `trt_max_partition_iterations` |
| `coreml` | `MLComputeUnits`, `ModelFormat`, `EnableOnSubgraphs`, `RequireStaticInputShapes`, `SpecializationStrategy`, `ModelCacheDirectory`, `ProfileComputePlan`, `AllowLowPrecisionAccumulationOnGPU` |
| `directml` | `device_id` |
| `acl` | `fast_math` |
| `onednn` | `use_arena` |
| `rocm` | `device_id`, `gpu_mem_limit`, `arena_extend_strategy`, `miopen_conv_exhaustive_search`, `miopen_conv_use_max_workspace`, `do_copy_in_default_stream`, `enable_hip_graph`, `tunable_op_enable`, `tunable_op_tuning_enable`, `tunable_op_max_tuning_duration_ms` |

CPU `arena_allocator` and ACL `fast_math` accept `"true"`/`"false"` or `"1"`/`"0"`.
DirectML `device_id` must be a non-negative i32 string. Other values are forwarded
to the provider's ONNX Runtime registration API for validation. Use the exact
provider spelling, such as `"0"` for a device ID, `"1"` for CUDA booleans, or
`"CPUAndNeuralEngine"` for CoreML `MLComputeUnits`. The curated key set omits
options requiring caller-owned native pointers; arbitrary config is not exposed.

```elixir
Ortex.execution_providers()
# %{cpu: %{compiled: true, available: true},
#   cuda: %{compiled: false, available: nil}, ...}
```

`:compiled` reports NIF registration support. For compiled providers, `:available`
reports the loaded runtime's advertised support. For uncompiled providers it is
`nil`, since ort does not expose that provider's query without its Cargo feature.
An available provider may still fail registration because device dependencies are
missing, and graph operators may still execute on CPU. Availability is not a
placement report. Strict session registration remains the usability check.

CPU arena configurations are tested against runtime 1.17.3 and 1.24.4. CI compiles
all exposed provider mappings together, but accelerator runtime validation needs
the relevant libraries and hardware; compile success does not establish it.
