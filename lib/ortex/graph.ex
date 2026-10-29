defmodule Ortex.Graph do
  @moduledoc """
  ONNX graph summaries from protobuf bytes, without creating a runtime session.

  Reports model/opset versions, static operator counts, ZipMap presence, and
  dense/sparse initializer descriptors with logical byte estimates. Traversal
  includes nested graph attributes and local function bodies. Counts describe
  serialized nodes, not execution frequency or provider placement.

  Tensor payloads are not decoded into numbers or returned. Initializer sizes
  are logical estimates from shape and element width, including packed sub-byte
  types; they exclude protobuf overhead and tensor-valued node attributes.
  String/unknown types have no byte estimate. Sparse initializers estimate stored
  values plus indices, not their equivalent dense shape. External data metadata
  is reported without opening paths, reading weights, or validating their contents.

  This is a summary parser, not the ONNX semantic checker. Models still need
  runtime validation. Unknown protobuf fields remain compatible with later ONNX
  schemas; unfamiliar element types retain their numeric code.
  """
  alias Ortex.Graph.Onnx.ModelProto

  @bits %{
    1 => 32,
    2 => 8,
    3 => 8,
    4 => 16,
    5 => 16,
    6 => 32,
    7 => 64,
    9 => 8,
    10 => 16,
    11 => 64,
    12 => 32,
    13 => 64,
    14 => 64,
    15 => 128,
    16 => 16,
    17 => 8,
    18 => 8,
    19 => 8,
    20 => 8,
    21 => 4,
    22 => 4,
    23 => 4,
    24 => 8,
    25 => 2,
    26 => 2
  }

  @doc "Reads an ONNX protobuf file and returns its graph summary."
  def from_file(path), do: path |> File.read!() |> from_binary()

  @doc "Returns a graph summary from ONNX protobuf bytes. Raises on invalid encoding or an absent graph."
  def from_binary(bytes) when is_binary(bytes) do
    model =
      case ModelProto.decode(bytes) do
        {:ok, model} ->
          model

        {:error, error} ->
          raise ArgumentError, "invalid ONNX protobuf: #{Exception.message(error)}"
      end

    if model.graph == nil, do: raise(ArgumentError, "ONNX protobuf has no graph")
    state = walk_graph(model.graph, [], %{operators: %{}, initializers: [], graphs: 0})

    state =
      Enum.reduce(model.functions, state, fn function, state ->
        path = [
          "function:#{copy(function.domain)}:#{copy(function.name)}:#{copy(function.overload)}"
        ]

        state = walk_nodes(function.node, path, state)
        walk_attributes(function.attribute_proto, path, state)
      end)

    initializers = Enum.reverse(state.initializers)
    known = for %{logical_bytes: bytes} <- initializers, is_integer(bytes), do: bytes

    %{
      ir_version: model.ir_version,
      producer: copy(model.producer_name),
      producer_version: copy(model.producer_version),
      domain: copy(model.domain),
      model_version: model.model_version,
      graph_name: copy(model.graph.name),
      opsets: opsets(model.opset_import),
      graph_node_count: length(model.graph.node),
      node_count: Enum.sum(Map.values(state.operators)),
      subgraph_count: state.graphs - 1,
      operators: state.operators,
      zipmap?: Map.has_key?(state.operators, {"ai.onnx.ml", "ZipMap"}),
      initializers: initializers,
      initializer_bytes: %{
        known: Enum.sum(known),
        unknown_count: length(initializers) - length(known)
      },
      functions:
        Enum.map(model.functions, fn function ->
          %{
            name: copy(function.name),
            domain: copy(function.domain),
            overload: copy(function.overload),
            opsets: opsets(function.opset_import)
          }
        end)
    }
  end

  defp opsets(imports), do: Enum.map(imports, &%{domain: copy(&1.domain), version: &1.version})

  defp walk_graph(graph, parents, state) do
    path = parents ++ ["graph:" <> copy(graph.name)]

    dense =
      Enum.map(graph.initializer, fn tensor ->
        Map.merge(tensor_info(tensor), %{kind: :dense, path: path})
      end)

    sparse =
      Enum.map(graph.sparse_initializer, fn tensor ->
        values = tensor_info(tensor.values)
        indices = tensor_info(tensor.indices)

        size =
          if is_integer(values.logical_bytes) and is_integer(indices.logical_bytes),
            do: values.logical_bytes + indices.logical_bytes

        %{
          kind: :sparse,
          name: values.name,
          shape: List.to_tuple(tensor.dims),
          path: path,
          logical_bytes: size,
          values: values,
          indices: indices
        }
      end)

    state = %{
      state
      | graphs: state.graphs + 1,
        initializers: Enum.reverse(dense ++ sparse, state.initializers)
    }

    walk_nodes(graph.node, path, state)
  end

  defp walk_nodes(nodes, parents, state) do
    nodes
    |> Enum.with_index()
    |> Enum.reduce(state, fn {node, index}, state ->
      key = {copy(node.domain), copy(node.op_type)}
      state = %{state | operators: Map.update(state.operators, key, 1, &(&1 + 1))}
      walk_attributes(node.attribute, parents ++ ["node:#{index}:#{copy(node.name)}"], state)
    end)
  end

  defp walk_attributes(attributes, parents, state) do
    Enum.reduce(attributes, state, fn attribute, state ->
      graphs = if attribute.g, do: [attribute.g | attribute.graphs], else: attribute.graphs

      graphs
      |> Enum.with_index()
      |> Enum.reduce(state, fn {graph, index}, state ->
        walk_graph(graph, parents ++ ["attribute:#{copy(attribute.name)}:#{index}"], state)
      end)
    end)
  end

  defp tensor_info(nil),
    do: %{
      name: "",
      shape: {},
      data_type: nil,
      logical_bytes: nil,
      external: false,
      external_data: %{}
    }

  defp tensor_info(tensor) do
    # Copy every returned binary: small protobuf sub-binaries must not retain a
    # potentially large source model after this temporary decoded struct dies.
    external = Map.new(tensor.external_data, &{copy(&1.key), copy(&1.value)})

    %{
      name: copy(tensor.name),
      shape: List.to_tuple(tensor.dims),
      data_type: tensor.data_type,
      logical_bytes: logical_bytes(tensor.dims, tensor.data_type),
      external: tensor.data_location == 1 or map_size(external) > 0,
      external_data: external
    }
  end

  defp logical_bytes(dims, type) do
    bits = @bits[type]
    if bits && Enum.all?(dims, &(&1 >= 0)), do: div(Enum.product(dims) * bits + 7, 8)
  end

  defp copy(nil), do: ""
  defp copy(binary), do: :binary.copy(binary)
end
