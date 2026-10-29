defmodule Ortex.Telemetry do
  @moduledoc """
  Telemetry events emitted by public model loading and inference calls.

  The event prefixes are `[:ortex, :session, :load]` and `[:ortex, :run]`, each
  followed by `:start`, `:stop`, or `:exception`. Start measurements contain
  `:system_time`; stop/exception measurements contain `:duration` in native time
  units. Convert durations with `System.convert_time_unit/3`.

  Load metadata includes `:source` (`:file` or `:memory`), `:path` or `:byte_size`,
  `:providers`, and `:optimization_level`. A `nil` provider option means that
  the session inherits environment providers. Successful loads add the opaque
  `:model` reference. Run metadata includes the model reference, `:tag`,
  `:selected_outputs`, and `:inputs` summaries; successful runs add `:outputs`.
  Summaries include tensor shapes/dtypes or container kind/size, without tensor
  values or model bytes. Named I/O summaries use binary keys; positional ones
  use lists in argument/result order.

  Exceptions include `:kind`, `:reason`, and `:stacktrace` and are re-raised with
  their original stack. Invalid option lists are rejected before starting spans.
  Telemetry handlers run synchronously in the caller, including their overhead
  in observed latency. The runtime log bridge and runtime profiling are separate
  from these public-call events.
  """

  @doc false
  def load(source, options, fun) do
    metadata = Map.merge(source, Map.take(options, [:providers, :optimization_level]))

    :telemetry.span([:ortex, :session, :load], metadata, fn ->
      model = fun.()
      {model, Map.put(metadata, :model, model.reference)}
    end)
  end

  @doc false
  def run(model, inputs, options, fun) do
    metadata = %{
      model: model.reference,
      tag: options.tag,
      selected_outputs: options.outputs,
      inputs: summarize(inputs)
    }

    :telemetry.span([:ortex, :run], metadata, fn ->
      outputs = fun.()
      {outputs, Map.put(metadata, :outputs, summarize(outputs))}
    end)
  end

  defp summarize(inputs) when is_map(inputs) and not is_struct(inputs),
    do: Map.new(inputs, fn {name, value} -> {name, describe(value)} end)

  defp summarize(inputs) when is_tuple(inputs),
    do: inputs |> Tuple.to_list() |> Enum.map(&describe/1)

  defp summarize(input), do: [describe(input)]

  defp describe(%Nx.Tensor{shape: shape, type: type}), do: %{shape: shape, type: type}
  defp describe(%Ortex.StringTensor{shape: shape}), do: %{shape: shape, type: :string}
  defp describe(value) when is_list(value), do: %{kind: :sequence, size: length(value)}
  defp describe(value) when is_map(value), do: %{kind: :map, size: map_size(value)}
  defp describe(_), do: %{kind: :unsupported}
end
