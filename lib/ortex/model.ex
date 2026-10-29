defmodule Ortex.Model do
  @moduledoc """
  A model for running Ortex inference with.

  Implements a human-readable representation of a model including the name, dimension, and
  type of each input and output

  ```
  #Ortex.Model<
  inputs: [{"x", "Int32", [nil, 100]}, {"y", "Float32", [nil, 100]}]
  outputs: [
    {"9", "Float32", [nil, 10]},
    {"onnx::Add_7", "Float32", [nil, 10]},
    {"onnx::Add_8", "Float32", [nil, 10]}
  ]>
  ```

  `nil` values represent dynamic dimensions
  """

  @enforce_keys [:reference]
  defstruct [:reference]

  @doc false
  def load(path), do: load(path, [:cpu], 3)

  def load(path, options_or_providers) when is_list(options_or_providers) do
    if Keyword.keyword?(options_or_providers) do
      {environment, options} = Ortex.Options.validate!(options_or_providers)
      load_with_options(path, environment, options)
    else
      load(path, options_or_providers, 3)
    end
  end

  def load(path, providers, optimization) do
    {environment, options} = Ortex.Options.legacy!(providers, optimization)
    load_with_options(path, environment, options)
  end

  @doc false
  def load_from_memory(bytes, opts \\ []) when is_binary(bytes) do
    {environment, options} = Ortex.Options.validate!(opts)
    environment = environment || Ortex.Environment.get()
    Ortex.Native.init_from_memory(bytes, environment.reference, options) |> to_model()
  end

  defp load_with_options(path, environment, options) when is_binary(path) do
    environment = environment || Ortex.Environment.get()
    Ortex.Native.init_with_options(path, environment.reference, options) |> to_model()
  end

  defp to_model({:error, message}), do: raise(message)
  defp to_model(reference), do: %Ortex.Model{reference: reference}

  @doc false
  def metadata(%__MODULE__{reference: reference}) do
    case Ortex.Native.metadata(reference) do
      {:error, message} -> raise message
      metadata -> metadata
    end
  end

  @doc false
  def run(model, inputs), do: run(model, inputs, [])

  @doc false
  def run(model, inputs, opts), do: run_validated(model, inputs, Ortex.RunOptions.validate!(opts))

  defp run_validated(%__MODULE__{reference: reference}, inputs, opts)
       when is_map(inputs) and not is_struct(inputs) do
    inputs =
      Enum.map(inputs, fn
        {name, %Nx.Tensor{} = tensor} when is_binary(name) -> {name, input_reference(tensor)}
        _ -> raise ArgumentError, "named inputs must map binary names to Nx tensors"
      end)

    case Ortex.Native.run_named_values(reference, inputs, opts) do
      {:error, message} -> raise message
      outputs -> Map.new(outputs, fn {name, tensor} -> {name, output_tensor(tensor)} end)
    end
  end

  defp run_validated(%Ortex.Model{} = model, tensor, opts) when not is_tuple(tensor) do
    run_validated(model, {tensor}, opts)
  end

  defp run_validated(%Ortex.Model{reference: model}, tensors, opts) do
    # Move tensors into Ortex backend and pass the reference to the Ortex NIF
    output =
      case Ortex.Native.run_values(
             model,
             tensors
             |> Tuple.to_list()
             |> Enum.map(&input_reference/1),
             opts
           ) do
        {:error, msg} -> raise msg
        output -> output
      end

    # Pack the output into new Ortex.Backend tensor(s)
    output
    |> Enum.map(&output_tensor/1)
    |> List.to_tuple()
  end

  defp input_reference(%Nx.Tensor{} = tensor) do
    %Nx.Tensor{data: %Ortex.Backend{ref: reference}} = Nx.backend_transfer(tensor, Ortex.Backend)
    reference
  end

  defp output_tensor({ref, shape, dtype_atom, dtype_bits}) do
    %Nx.Tensor{
      data: %Ortex.Backend{ref: ref},
      shape: shape |> List.to_tuple(),
      type: {dtype_atom, dtype_bits},
      names: List.duplicate(nil, length(shape))
    }
  end

  defp output_tensor({:tensor, descriptor}), do: output_tensor(descriptor)
  defp output_tensor({:sequence, values}), do: Enum.map(values, &output_tensor/1)
  defp output_tensor({:map, values}), do: values
end

defimpl Inspect, for: Ortex.Model do
  import Inspect.Algebra

  def inspect(%Ortex.Model{reference: model}, inspect_opts) do
    case Ortex.Native.show_session(model) do
      {:error, msg} ->
        raise msg

      {inputs, outputs} ->
        force_unfit(
          concat([
            color("#Ortex.Model<", :map, inspect_opts),
            line(),
            nest(concat(["  inputs: ", Inspect.Algebra.to_doc(inputs, inspect_opts)]), 2),
            line(),
            nest(concat(["  outputs: ", Inspect.Algebra.to_doc(outputs, inspect_opts)]), 2),
            color(">", :map, inspect_opts)
          ])
        )
    end
  end
end
