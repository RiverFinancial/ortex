defmodule Ortex.RunOptions do
  @moduledoc false

  def validate!(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "expected a keyword list of run options")

    keys = Keyword.keys(opts)
    unknown = keys -- [:outputs, :tag]
    if unknown != [], do: raise(ArgumentError, "unknown Ortex run options: #{inspect(unknown)}")

    if length(keys) != length(Enum.uniq(keys)),
      do: raise(ArgumentError, "duplicate Ortex run options")

    for {key, value} <- opts do
      valid =
        case key do
          :outputs ->
            is_list(value) and value != [] and Enum.all?(value, &valid_string?/1) and
              length(value) == length(Enum.uniq(value))

          :tag ->
            valid_string?(value)
        end

      unless valid,
        do: raise(ArgumentError, "invalid Ortex run option #{inspect(key)}: #{inspect(value)}")
    end

    Map.merge(%{outputs: nil, tag: nil}, Map.new(opts))
  end

  defp valid_string?(value),
    do: is_binary(value) and String.valid?(value) and not String.contains?(value, <<0>>)
end
