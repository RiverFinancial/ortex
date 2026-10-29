defmodule Ortex.StringTensor do
  @moduledoc """
  A string tensor with a tuple shape and flat, row-major list of UTF-8 binaries.

  Nx tensors represent numeric values. Use this type for ONNX string tensors,
  including scalar strings (`shape: {}`) and empty dimensions. Strings cannot
  contain NUL bytes, as required by the runtime's string input constructor.

      input = Ortex.StringTensor.new(["hello", "π"], {2})
      {output} = Ortex.run(model, input)
      output.data
  """
  @enforce_keys [:data, :shape]
  defstruct [:data, :shape]

  @type t :: %__MODULE__{data: [String.t()], shape: tuple()}

  @doc "Creates a validated string tensor from flat values and a tuple shape."
  @spec new([String.t()], tuple()) :: t()
  def new(data, shape) do
    unless is_tuple(shape), do: raise(ArgumentError, "string tensor shape must be a tuple")
    dims = Tuple.to_list(shape)

    unless Enum.all?(dims, &(is_integer(&1) and &1 >= 0 and &1 <= 9_223_372_036_854_775_807)),
      do: raise(ArgumentError, "string tensor dimensions must be non-negative i64 integers")

    unless is_list(data) and
             Enum.all?(
               data,
               &(is_binary(&1) and String.valid?(&1) and not String.contains?(&1, <<0>>))
             ),
           do: raise(ArgumentError, "string tensor data must be UTF-8 binaries without NUL bytes")

    unless Enum.product(dims) == length(data),
      do: raise(ArgumentError, "string tensor shape does not match value count")

    %__MODULE__{data: data, shape: shape}
  end
end
