defmodule Ortex.GraphTest do
  use ExUnit.Case, async: true
  @fixture Path.expand("fixtures/graph_summary.onnx", __DIR__)

  test "model and opset versions remain structured and independent of the runtime" do
    summary = Ortex.Graph.from_file(@fixture)
    assert summary.ir_version == 13
    assert summary.producer == "River Ortex tests"
    assert summary.producer_version == "fixture-1"
    assert summary.domain == "org.riverfinancial.ortex.tests"
    assert summary.model_version == 7
    assert summary.graph_name == "graph_summary"
    assert summary.opsets == [%{domain: "", version: 15}, %{domain: "local.tests", version: 1}]
  end

  test "operator counts include nested graphs and local function bodies once" do
    summary = Ortex.Graph.from_file(@fixture)
    assert summary.graph_node_count == 3
    assert summary.node_count == 6
    assert summary.subgraph_count == 2

    assert summary.operators == %{
             {"", "MatMul"} => 1,
             {"", "If"} => 1,
             {"", "Identity"} => 1,
             {"", "Neg"} => 1,
             {"", "Add"} => 1,
             {"local.tests", "Double"} => 1
           }

    assert [%{domain: "local.tests", name: "Double", opsets: [%{domain: "", version: 15}]}] =
             summary.functions

    refute summary.zipmap?
  end

  test "dense, scalar, packed, sparse, and string initializer estimates are distinct" do
    summary = Ortex.Graph.from_file(@fixture)
    assert length(summary.initializers) == 7
    assert summary.initializer_bytes == %{known: 94, unknown_count: 1}
    by_name = Map.new(summary.initializers, &{&1.name, &1})
    assert by_name["weights"].logical_bytes == 24
    assert by_name["unused_scalar"].shape == {}
    assert by_name["unused_scalar"].logical_bytes == 8
    assert by_name["unused_packed"].logical_bytes == 2
    assert by_name["unused_words"].logical_bytes == nil
    assert by_name["sparse"].shape == {4, 4}
    assert by_name["sparse"].logical_bytes == 40
    assert by_name["sparse"].values.logical_bytes == 8
    assert by_name["sparse"].indices.logical_bytes == 32
    assert length(by_name["branch_scalar"].path) > 1
  end

  test "external tensor descriptors are returned without opening their file" do
    summary = Ortex.Graph.from_binary(File.read!(@fixture))
    external = Enum.find(summary.initializers, &(&1.name == "external"))
    assert external.external

    assert external.external_data == %{
             "location" => "../../missing.weights",
             "offset" => "0",
             "length" => "16"
           }

    assert external.logical_bytes == 16
    assert summary == Ortex.Graph.from_file(@fixture)
  end

  test "ZipMap is recognized by its domain and operator" do
    summary = Ortex.Graph.from_file(Path.expand("fixtures/metadata_zipmap.onnx", __DIR__))
    assert summary.zipmap?
    assert summary.operators == %{{"ai.onnx.ml", "ZipMap"} => 1}
    assert summary.initializer_bytes == %{known: 0, unknown_count: 0}
  end

  test "invalid protobuf and missing graphs raise useful errors" do
    bytes = File.read!(@fixture)

    for invalid <- [<<>>, <<0>>, <<0x3A, 0xFF>>, binary_part(bytes, 0, byte_size(bytes) - 1)] do
      assert_raise ArgumentError, ~r/ONNX protobuf/, fn -> Ortex.Graph.from_binary(invalid) end
    end
  end

  test "future fields are tolerated and summaries do not retain a large model binary" do
    payload = :binary.copy(<<0>>, 8_388_608)
    bytes = File.read!(@fixture) <> <<0xA2, 0x06>> <> varint(byte_size(payload)) <> payload
    summary = Ortex.Graph.from_binary(bytes)
    assert summary == Ortex.Graph.from_file(@fixture)

    for string <- binaries(summary), byte_size(string) > 0 do
      assert :binary.referenced_byte_size(string) == byte_size(string)
    end
  end

  defp varint(n) when n < 128, do: <<n>>
  defp varint(n), do: <<Bitwise.bor(Bitwise.band(n, 127), 128)>> <> varint(Bitwise.bsr(n, 7))
  defp binaries(value) when is_binary(value), do: [value]

  defp binaries(value) when is_map(value),
    do: value |> Map.to_list() |> Enum.flat_map(&binaries/1)

  defp binaries(value) when is_tuple(value),
    do: value |> Tuple.to_list() |> Enum.flat_map(&binaries/1)

  defp binaries(value) when is_list(value), do: Enum.flat_map(value, &binaries/1)
  defp binaries(_), do: []
end
