defmodule Ortex.MemoryLoadingTest do
  use ExUnit.Case, async: true

  @path Path.expand("fixtures/xgboost.onnx", __DIR__)
  @cases elem(Code.eval_file(Path.expand("fixtures/xgboost_cases.ex", __DIR__)), 0)

  test "memory and file loading preserve scores and descriptors with the same options" do
    for opts <- [
          [],
          [optimization_level: 0, intra_threads: 1],
          [intra_threads: 2, intra_op_spinning: false]
        ] do
      file = Ortex.load(@path, opts)
      memory = Ortex.load_from_memory(File.read!(@path), opts)

      assert Ortex.Native.show_session(memory.reference) ==
               Ortex.Native.show_session(file.reference)

      assert scores(memory) == scores(file)
      assert_reference(memory)
    end
  end

  test "the model remains usable after the process owning its source binary exits" do
    model =
      Task.async(fn ->
        Ortex.load_from_memory(File.read!(@path), intra_threads: 1)
      end)
      |> Task.await()

    :erlang.garbage_collect()
    for _ <- 1..64, do: assert_reference(model)
  end

  test "invalid and truncated models return errors without breaking later loads" do
    bytes = File.read!(@path)

    for invalid <- [<<>>, "invalid protobuf", binary_part(bytes, 0, div(byte_size(bytes), 2))] do
      assert_raise RuntimeError, fn -> Ortex.load_from_memory(invalid, intra_threads: 1) end
    end

    assert_reference(Ortex.load_from_memory(bytes, intra_threads: 1))
  end

  test "memory loads use the common option validation before parsing the model" do
    assert_raise ArgumentError, ~r/intra_threads/, fn ->
      Ortex.load_from_memory("invalid protobuf", intra_threads: -1)
    end
  end

  test "independent callers can construct and score memory models" do
    1..8
    |> Task.async_stream(
      fn _ ->
        model = Ortex.load_from_memory(File.read!(@path), intra_threads: 1)
        assert_reference(model)
      end,
      max_concurrency: 4
    )
    |> Enum.each(fn result -> assert {:ok, :ok} = result end)
  end

  defp scores(model) do
    {output} = Ortex.run(model, Nx.tensor(@cases.rows, type: :f32))
    output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
  end

  defp assert_reference(model) do
    actual = scores(model)
    assert length(actual) == length(@cases.expected)
    Enum.zip(actual, @cases.expected) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    :ok
  end
end
