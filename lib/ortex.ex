defmodule Ortex do
  @moduledoc """
  Documentation for `Ortex`.

  `Ortex` is an Elixir wrapper around [ONNX Runtime](https://onnxruntime.ai/) using
  [Rustler](https://hexdocs.pm/rustler) and [ORT](https://github.com/pykeio/ort).
  """

  @doc """
  Load an `Ortex.Model` from disk. Optionally pass the execution providers as a list
  of descending priority and graph optimization level 1-3. Any graph optimization level
  beyond the range of 1-3 will disable graph optimization.

  The keyword form accepts `providers`, `optimization_level` (0-3),
  `intra_threads`, `inter_threads`, `parallel_execution`, `intra_op_spinning`,
  `inter_op_spinning`, `independent_thread_pool`, `environment`, `profiling`,
  `optimized_model_path`, `deterministic_compute`, `memory_pattern`, and `log_id`.
  Thread counts are non-negative integers; zero leaves selection to the runtime.
  See `Ortex.Environment` for shared global pools.

  The source build targets API 17 by default and uses `ORT_DYLIB_PATH` to
  find the runtime library. Accelerator providers require explicit Cargo features
  and a matching runtime. Unknown or uncompiled providers raise. Keyword calls
  fail on provider registration errors unless `error_on_failure: false` is set.
  Positional arities preserve legacy optimization and registration behavior.

  ## Examples

      iex> Ortex.load("./models/tinymodel.onnx")
      iex> Ortex.load("./models/tinymodel.onnx", intra_threads: 1, intra_op_spinning: false)
      iex> Ortex.load("./models/tinymodel.onnx", [:cpu], 0)

  """
  defdelegate load(path), to: Ortex.Model
  defdelegate load(path, options_or_providers), to: Ortex.Model
  defdelegate load(path, providers, optimization_level), to: Ortex.Model

  @doc """
  Loads an ONNX model directly from a binary using the same keyword options as
  `load/2`. The copying constructor owns the model after loading; the caller may
  discard the binary. Models referencing external tensor-data files should be
  loaded from disk with their external files available.

      model = Ortex.load_from_memory(File.read!("model.onnx"), intra_threads: 1)
  """
  defdelegate load_from_memory(bytes, opts \\ []), to: Ortex.Model

  @doc """
  Returns model metadata and structured input/output descriptors.

  Each descriptor contains a binary `:name` and a recursive `:type`:
  `{:tensor, dtype, shape}`, `{:map, key_dtype, value_dtype}`,
  `{:sequence, type}`, or `{:optional, type}`. Numeric dtypes use Nx tuples,
  with `:bool` and `:string` for those element types. Unrecognized element types
  return `{:unsupported, binary_name}`. Tensor shapes are tuples containing
  integers, binary symbolic names, or `nil` for unnamed dynamic dimensions.

  Model fields are `:name`, `:producer`, `:domain`, `:description`,
  `:graph_description`, `:version`, and `:custom`. Custom keys and values remain
  binaries. Optional model fields may be `nil`; runtime-provided empty strings
  are preserved. Descriptor order follows the graph declaration.

  Describing a type does not imply that `run/2` can convert its values. Map,
  sequence, optional, string, and boolean values require separate value support.

  This API does not parse a Rust Debug rendering. `Native.show_session/1` keeps
  its existing format for callers that still use it.
  """
  defdelegate metadata(model), to: Ortex.Model

  @doc """
  Run a forward pass through a model.

  This takes a model and tuple of `Nx.Tensors`,
  optionally transfers them to the `Ortex.Backend` if they aren't there already,
  and runs a forward pass through the model. This will return a tuple of `Ortex.Backend`
  tensors, it's up to the user to transfer these back to another backend if additional
  ops are required.

  If there is only one input you can optionally pass a bare tensor rather than a tuple.

  A map of binary input names to tensors returns a map of binary output names
  to tensors. Input names must match the graph exactly; map iteration order
  does not affect binding. Positional calls continue to return tuples.

      outputs = Ortex.run(model, %{"features" => features})

  ## Examples

      iex> model = Ortex.load("./models/tinymodel.onnx")
      iex> {%Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}},
      ...>  %Nx.Tensor{shape: {1, 10}}} = Ortex.run(
      ...>    model, {
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32),
      ...>      Nx.broadcast(0, {1, 100}) |> Nx.as_type(:f32)
      ...>    })

  """
  defdelegate run(model, tensors), to: Ortex.Model
end
