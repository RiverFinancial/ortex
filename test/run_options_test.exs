defmodule Ortex.RunOptionsTest do
  use ExUnit.Case, async: true

  setup do
    model = Ortex.load(Path.expand("fixtures/named_io.onnx", __DIR__), intra_threads: 1)
    {:ok, model: model, inputs: {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])}}
  end

  defp values(tensor), do: tensor |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  test "selected positional outputs follow requested order", c do
    {difference, sum} = Ortex.run(c.model, c.inputs, outputs: ["Δ", "sum"], tag: "test π")
    assert values(difference) == [5.0, 4.0]
    assert values(sum) == [9.0, 6.0]
    {only} = Ortex.run(c.model, c.inputs, outputs: ["Δ"])
    assert values(only) == [5.0, 4.0]
    assert tuple_size(Ortex.run(c.model, c.inputs)) == 2
  end

  test "named output selection returns only requested binary keys", c do
    {left, right} = c.inputs
    result = Ortex.run(c.model, %{"z_left" => left, "a_right" => right}, outputs: ["Δ"])
    assert Map.keys(result) == ["Δ"]
    assert values(result["Δ"]) == [5.0, 4.0]
  end

  test "unknown output fails without changing subsequent runs", c do
    assert_raise RuntimeError, ~r/Unknown output/, fn ->
      Ortex.run(c.model, c.inputs, outputs: ["unknown"])
    end

    {sum, _} = Ortex.run(c.model, c.inputs)
    assert values(sum) == [9.0, 6.0]
  end

  test "invalid and duplicate options fail before inference", c do
    for options <- [
          [:tag],
          %{tag: "x"},
          [unknown: true],
          [tag: "a", tag: "b"],
          [outputs: []],
          [outputs: ["sum", "sum"]],
          [outputs: [:sum]],
          [outputs: "sum"],
          [outputs: ["sum" <> <<0>>]],
          [tag: 123],
          [tag: <<0>>],
          [tag: <<255>>]
        ] do
      assert_raise ArgumentError, fn -> Ortex.run(c.model, c.inputs, options) end
    end
  end

  test "native output validation rejects duplicates and empty selections", c do
    refs =
      c.inputs
      |> Tuple.to_list()
      |> Enum.map(fn tensor ->
        %Nx.Tensor{data: %Ortex.Backend{ref: ref}} = Nx.backend_transfer(tensor, Ortex.Backend)
        ref
      end)

    for outputs <- [[], ["sum", "sum"], ["unknown"]] do
      assert {:error, message} =
               Ortex.Native.run_with_options(c.model.reference, refs, %{
                 outputs: outputs,
                 tag: nil
               })

      assert is_binary(message)
    end
  end

  test "concurrent run options do not leak between calls", c do
    1..16
    |> Task.async_stream(
      fn n ->
        name = if rem(n, 2) == 0, do: "sum", else: "Δ"
        {output} = Ortex.run(c.model, c.inputs, outputs: [name], tag: "run-#{n}")
        assert values(output) == if(name == "sum", do: [9.0, 6.0], else: [5.0, 4.0])
      end,
      max_concurrency: 4
    )
    |> Enum.each(fn result -> assert result == {:ok, true} end)
  end
end
