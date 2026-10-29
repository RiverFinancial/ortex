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
  Use `Ortex.Environment.get/0` to pass the environment handle explicitly.

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
  Run a forward pass through a model.

  This takes a model and tuple of `Nx.Tensors`,
  optionally transfers them to the `Ortex.Backend` if they aren't there already,
  and runs a forward pass through the model. This will return a tuple of `Ortex.Backend`
  tensors, it's up to the user to transfer these back to another backend if additional
  ops are required.

  If there is only one input you can optionally pass a bare tensor rather than a tuple.

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
