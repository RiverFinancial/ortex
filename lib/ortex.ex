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
  `optimized_model_path`, `deterministic_compute`, `memory_pattern`,
  `dimension_overrides`, and `log_id`.
  Thread counts are non-negative integers; zero leaves selection to the runtime.
  See `Ortex.Environment` for shared global pools.

  `dimension_overrides` maps binary symbolic dimension names to non-negative i64
  sizes, for example `%{"batch" => 1}`. This specializes free dimensions when
  creating the session. ONNX Runtime ignores names absent from the graph.

  The source build targets API 17 by default and uses `ORT_DYLIB_PATH` to
  find the runtime library. Accelerator providers require explicit Cargo features
  and a matching runtime. Unknown or uncompiled providers raise. Keyword calls
  fail on provider registration errors unless `error_on_failure: false` is set.
  Positional arities preserve legacy optimization and registration behavior.
  Keyword loading inherits environment providers when this option is omitted;
  an explicit provider list replaces the environment defaults.

  Provider entries may be atoms or `{provider, %{binary_key => binary_value}}`
  tuples, in descending priority. For example,
  `providers: [{:cpu, %{"arena_allocator" => "true"}}]`. Unknown options raise;
  accelerator option values are validated during runtime registration. See
  `docs/providers.md` for supported keys and build requirements.

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

  Describing a type does not imply that `run/2` can convert its values.
  Optional values require separate value support.

  This API does not parse a Rust Debug rendering. `Native.show_session/1` keeps
  its existing format for callers that still use it.
  """
  defdelegate metadata(model), to: Ortex.Model

  @doc """
  Ends session profiling, flushes its Chrome trace JSON, and returns its file path.

  Enable profiling when loading with `profiling: prefix`. Ending profiling waits
  for any inference holding the session lock; the model remains usable afterward.
  Returns an empty binary if profiling was not enabled. This session API works at
  the default API 17 floor. Per-run profiling toggles require runtime API 25 and
  are not exposed by the current API 17/24 runtime matrix.

      model = Ortex.load("model.onnx", profiling: "/tmp/ortex-profile")
      Ortex.run(model, inputs)
      path = Ortex.end_profiling(model)
  """
  defdelegate end_profiling(model), to: Ortex.Model

  @doc """
  Reports supported execution providers and their build/runtime availability.

  Returns a map from provider atoms to `%{compiled: boolean, available: boolean | nil}`.
  `:compiled` means this NIF has registration support; `:available` means the
  loaded runtime advertises the provider. `:available` is `nil` when this NIF
  lacks registration support and cannot query that provider through ort.
  Availability does not guarantee device
  usability, dependency loading, or support for a particular graph. Registration
  with `error_on_failure: true` remains the check when creating a session.
  """
  def execution_providers do
    case Ortex.Native.execution_providers(Ortex.Environment.get().reference) do
      {:error, message} ->
        raise message

      providers ->
        Map.new(providers, fn {name, compiled, available} ->
          {name, %{compiled: compiled, available: available}}
        end)
    end
  end

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

  Sequence outputs become lists and map outputs become maps with integer or
  binary keys and numeric or boolean values, including ZipMap classifier
  probabilities. Numeric tensors, including scalars, remain Nx tensors;
  boolean tensors use Nx's `{:u, 8}` representation. Non-finite map floats and
  unsupported map element types raise a recoverable error.

  String tensors use `Ortex.StringTensor` for both input and output. Its data is
  a flat list of UTF-8 binaries with a tuple shape. Numeric scalar tensors retain
  shape `{}` and can be read with `Nx.to_number/1`.

  Numeric inputs borrow aligned Ortex storage until the synchronous run returns.
  Transferring from another backend still copies into that storage. Reusing a
  tensor already on `Ortex.Backend` avoids the transfer on subsequent runs.

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

  @doc """
  Runs inference with a keyword list of per-run options.

  `:outputs` selects a non-empty list of binary output names. Only selected
  outputs are requested from the runtime; positional calls return them as a
  tuple in the requested order, while named calls return a map. Omitting this
  option returns every output in graph order.

  `:tag` is a binary identifying the run in runtime logs. Options apply to this
  call only and do not mutate the session's defaults.

  `:control` accepts an `Ortex.RunControl` prepared for this model, allowing a
  different process to cancel the inference attempt. Its immutable output/tag
  options replace per-call `:outputs`/`:tag`, which cannot be passed alongside it.

      Ortex.run(model, %{"features" => features}, outputs: ["scores"], tag: "risk-score")
  """
  defdelegate run(model, tensors, opts), to: Ortex.Model
end
