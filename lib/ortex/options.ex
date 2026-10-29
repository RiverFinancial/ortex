defmodule Ortex.Options do
  @moduledoc false
  @defaults %{
    providers: nil,
    optimization_level: 3,
    error_on_failure: true,
    intra_threads: nil,
    inter_threads: nil,
    parallel_execution: nil,
    intra_op_spinning: nil,
    inter_op_spinning: nil,
    independent_thread_pool: false,
    profiling: nil,
    optimized_model_path: nil,
    deterministic_compute: nil,
    memory_pattern: nil,
    dimension_overrides: %{},
    log_id: nil
  }

  def validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "expected a keyword list")
    keys = Keyword.keys(opts)
    unknown = keys -- [:environment | Map.keys(@defaults)]
    if unknown != [], do: raise(ArgumentError, "unknown Ortex options: #{inspect(unknown)}")

    if length(keys) != length(Enum.uniq(keys)),
      do: raise(ArgumentError, "duplicate Ortex options")

    {environment, opts} = Keyword.pop(opts, :environment)

    for {key, value} <- opts do
      valid =
        case key do
          :providers ->
            is_list(value) and Enum.all?(value, &valid_provider?/1)

          :optimization_level ->
            is_integer(value) and value in 0..3

          key when key in [:intra_threads, :inter_threads] ->
            is_integer(value) and value >= 0 and value <= 2_147_483_647

          key when key in [:profiling, :optimized_model_path, :log_id] ->
            is_binary(value) and not String.contains?(value, <<0>>)

          :dimension_overrides ->
            is_map(value) and not is_struct(value) and
              Enum.all?(value, fn {name, size} ->
                is_binary(name) and name != "" and String.valid?(name) and
                  not String.contains?(name, <<0>>) and
                  is_integer(size) and size >= 0 and size <= 9_223_372_036_854_775_807
              end)

          _ ->
            is_boolean(value)
        end

      unless valid,
        do: raise(ArgumentError, "invalid Ortex option #{inspect(key)}: #{inspect(value)}")
    end

    if environment != nil and not is_struct(environment, Ortex.Environment),
      do: raise(ArgumentError, "environment must be an Ortex.Environment handle")

    {environment, Map.merge(@defaults, Map.new(opts))}
  end

  defp valid_provider?(provider) when is_atom(provider), do: true

  defp valid_provider?({provider, options})
       when is_atom(provider) and is_map(options) and not is_struct(options) do
    Enum.all?(options, fn {key, value} ->
      is_binary(key) and key != "" and String.valid?(key) and not String.contains?(key, <<0>>) and
        is_binary(value) and String.valid?(value) and not String.contains?(value, <<0>>)
    end)
  end

  defp valid_provider?(_), do: false

  def legacy!(providers, optimization) when is_list(providers) and is_integer(optimization) do
    validate!(
      providers: providers,
      optimization_level: if(optimization in 1..3, do: optimization, else: 0),
      error_on_failure: false
    )
  end
end
