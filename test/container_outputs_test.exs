defmodule Ortex.ContainerOutputsTest do
  use ExUnit.Case, async: true

  defp path(name), do: Path.expand("fixtures/#{name}.onnx", __DIR__)
  defp load(name), do: Ortex.load(path(name), intra_threads: 1)
  defp values(tensor), do: tensor |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  test "ZipMap integer class keys decode to a list of row maps" do
    model = load("metadata_zipmap")
    {mapped} = Ortex.run(model, Nx.tensor([[0.25, 0.75], [1.0, 0.0]]))
    assert mapped == [%{0 => 0.25, 1 => 0.75}, %{0 => 1.0, 1 => 0.0}]
  end

  test "ZipMap string class keys stay binaries including Unicode" do
    model = load("zipmap_strings")

    result =
      Ortex.run(model, %{"probabilities" => Nx.tensor([[0.25, 0.75]])}, outputs: ["mapped"])

    assert result == %{"mapped" => [%{"safe" => 0.25, "risk π" => 0.75}]}
  end

  test "tensor sequence order, shapes, and values are preserved" do
    model = load("sequence_tensors")
    {[x, y]} = Ortex.run(model, {Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])})
    assert x.shape == {2}
    assert values(x) == [1.0, 2.0]
    assert values(y) == [3.0, 4.0]
  end

  test "empty tensor sequences and zero-input graphs decode" do
    assert {[]} = Ortex.run(load("sequence_empty"), {})
    assert %{"sequence" => []} = Ortex.run(load("sequence_empty"), %{})
  end

  test "copied nested values outlive model and source process" do
    result =
      Task.async(fn ->
        model = Ortex.load_from_memory(File.read!(path("sequence_tensors")), intra_threads: 1)
        Ortex.run(model, %{"x" => Nx.tensor([7.0, 8.0]), "y" => Nx.tensor([9.0, 10.0])})
      end)
      |> Task.await()

    :erlang.garbage_collect()
    assert %{"sequence" => [x, y]} = result
    assert values(x) == [7.0, 8.0]
    assert values(y) == [9.0, 10.0]
  end

  test "non-finite map values produce a recoverable error" do
    model = load("metadata_zipmap")
    input = Nx.broadcast(Nx.Constants.nan(), {1, 2})
    assert_raise RuntimeError, ~r/non-finite map values/, fn -> Ortex.run(model, input) end
    assert {[%{0 => 0.25, 1 => 0.75}]} = Ortex.run(model, Nx.tensor([[0.25, 0.75]]))
  end
end
