defmodule Ortex.MetadataTest do
  use ExUnit.Case, async: true

  defp fixture(name), do: Path.expand("fixtures/#{name}.onnx", __DIR__)
  defp load(name), do: Ortex.load(fixture(name), intra_threads: 1)

  test "model fields and custom properties are structured binaries" do
    assert %{
             name: "metadata_rich",
             producer: "River Ortex tests",
             domain: "org.riverfinancial.ortex.tests",
             description: "Synthetic model description",
             graph_description: "Synthetic graph description",
             version: 42,
             custom: %{"feature.order" => ~s(["a","b","c"]), "unicode.π" => "value π"}
           } = Ortex.metadata(load("metadata_rich"))
  end

  test "tensor descriptors preserve names, symbolic dimensions, and unnamed dimensions" do
    metadata = Ortex.metadata(load("metadata_rich"))
    assert [%{name: "features", type: {:tensor, {:f, 32}, {"batch", nil, 3}}}] = metadata.inputs

    assert [%{name: "prediction", type: {:tensor, {:f, 32}, {"batch", nil, 3}}}] =
             metadata.outputs
  end

  test "scalar bools and strings are described even before their value decoder is used" do
    assert [%{type: {:tensor, :bool, {}}}] = Ortex.metadata(load("metadata_bool")).inputs
    assert [%{type: {:tensor, :string, {nil}}}] = Ortex.metadata(load("metadata_string")).outputs
    assert %{} = Ortex.metadata(load("metadata_string")).custom
  end

  test "sequence/map descriptors are recursive terms" do
    metadata = Ortex.metadata(load("metadata_zipmap"))
    assert [%{name: "mapped", type: {:sequence, {:map, {:s, 64}, {:f, 32}}}}] = metadata.outputs
  end

  test "file and memory metadata agree; legacy descriptors remain usable" do
    model = load("metadata_rich")
    memory = Ortex.load_from_memory(File.read!(fixture("metadata_rich")), intra_threads: 1)
    assert Ortex.metadata(memory) == Ortex.metadata(model)
    assert {[{"features", legacy, [-1, -1, 3]}], [_]} = Ortex.Native.show_session(model.reference)
    assert is_binary(legacy)
  end

  test "metadata terms outlive their model and can be read during inference" do
    copied = Task.async(fn -> Ortex.metadata(load("metadata_rich")) end) |> Task.await()
    :erlang.garbage_collect()
    assert copied.version == 42
    assert copied.custom["unicode.π"] == "value π"

    model = load("metadata_rich")

    1..16
    |> Task.async_stream(
      fn _ ->
        assert Ortex.metadata(model).version == 42
        {output} = Ortex.run(model, Nx.tensor([[[1.0, 2.0, 3.0]]]))

        assert [1.0, 2.0, 3.0] ==
                 output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
      end,
      max_concurrency: 4
    )
    |> Enum.each(fn result -> assert {:ok, true} = result end)
  end
end
