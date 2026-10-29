defmodule Ortex.NamedIOTest do
  use ExUnit.Case, async: true

  setup do
    path = Path.expand("fixtures/named_io.onnx", __DIR__)
    model = Ortex.load(path, intra_threads: 1)
    {:ok, model: model, path: path, left: Nx.tensor([7.0, 5.0]), right: Nx.tensor([2.0, 1.0])}
  end

  defp values(tensor), do: tensor |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  test "names bind independently of map order and preserve Unicode outputs", c do
    result = Ortex.run(c.model, %{"a_right" => c.right, "z_left" => c.left})
    assert Enum.sort(Map.keys(result)) == ["sum", "Δ"]
    assert values(result["sum"]) == [9.0, 6.0]
    assert values(result["Δ"]) == [5.0, 4.0]
  end

  test "positional results and memory-loaded models agree", c do
    {sum, difference} = Ortex.run(c.model, {c.left, c.right})
    model = Ortex.load_from_memory(File.read!(c.path), intra_threads: 1)
    named = Ortex.run(model, %{"z_left" => c.left, "a_right" => c.right})
    assert values(sum) == values(named["sum"])
    assert values(difference) == values(named["Δ"])
  end

  test "missing, extra, and unknown input names raise without poisoning the session", c do
    assert_raise RuntimeError, ~r/Expected 2 inputs, got 1/, fn ->
      Ortex.run(c.model, %{"z_left" => c.left})
    end

    assert_raise RuntimeError, ~r/Expected 2 inputs, got 3/, fn ->
      Ortex.run(c.model, %{"z_left" => c.left, "a_right" => c.right, "extra" => c.left})
    end

    assert_raise RuntimeError, ~r/Unknown input/, fn ->
      Ortex.run(c.model, %{"z_left" => c.left, "unknown" => c.right})
    end

    assert values(Ortex.run(c.model, %{"z_left" => c.left, "a_right" => c.right})["Δ"]) == [
             5.0,
             4.0
           ]
  end

  test "keys must be binaries and values must be tensors", c do
    for inputs <- [%{z_left: c.left}, %{"z_left" => [1, 2]}] do
      assert_raise ArgumentError, ~r/binary names to Nx tensors/, fn ->
        Ortex.run(c.model, inputs)
      end
    end
  end

  test "native boundary rejects duplicate names", c do
    %Nx.Tensor{data: %Ortex.Backend{ref: ref}} = Nx.backend_transfer(c.left, Ortex.Backend)

    assert {:error, message} =
             Ortex.Native.run_named(c.model.reference, [{"z_left", ref}, {"z_left", ref}])

    assert message =~ "Duplicate input"
  end

  test "concurrent named runs share a model safely", c do
    1..16
    |> Task.async_stream(
      fn _ ->
        result = Ortex.run(c.model, %{"z_left" => c.left, "a_right" => c.right})
        values(result["Δ"])
      end,
      max_concurrency: 4
    )
    |> Enum.each(fn result -> assert result == {:ok, [5.0, 4.0]} end)
  end
end
