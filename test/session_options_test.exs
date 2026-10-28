defmodule Ortex.SessionOptionsTest do
  use ExUnit.Case, async: false
  @path Path.expand("fixtures/xgboost.onnx", __DIR__)
  @cases elem(Code.eval_file(Path.expand("fixtures/xgboost_cases.ex", __DIR__)), 0)

  test "keyword options preserve scores for thread counts, execution modes, and spinning" do
    for option <- [
          [intra_threads: 0],
          [intra_threads: 1],
          [inter_threads: 0],
          [inter_threads: 2],
          [parallel_execution: true],
          [parallel_execution: false],
          [intra_op_spinning: true],
          [intra_op_spinning: false],
          [inter_op_spinning: true],
          [inter_op_spinning: false],
          [optimization_level: 0],
          [optimization_level: 3],
          [memory_pattern: false],
          [deterministic_compute: true],
          [log_id: "test"],
          [environment: Ortex.Environment.get()],
          []
        ] do
      model = Ortex.load(@path, Keyword.merge([intra_threads: 1, inter_threads: 1], option))
      assert_scores(model)
    end
  end

  test "all legacy arities and direct native init remain usable" do
    assert_scores(Ortex.load(@path))
    assert_scores(Ortex.load(@path, [:cpu]))
    assert_scores(Ortex.load(@path, [:cpu], -1))
    reference = Ortex.Native.init(@path, [:cpu], 0)
    assert_scores(%Ortex.Model{reference: reference})
  end

  test "unknown, duplicate, and invalid options fail before loading a file" do
    for options <- [
          [intr_threads: 1],
          [intra_threads: -1],
          [intra_threads: 2_147_483_648],
          [intra_threads: 1.0],
          [intra_threads: nil],
          [inter_threads: "1"],
          [parallel_execution: 1],
          [intra_op_spinning: :yes],
          [inter_op_spinning: nil],
          [intra_threads: 1, intra_threads: 2],
          [providers: [:cpu, "cuda"]],
          [providers: :cpu],
          [optimization_level: 4],
          [environment: :fake],
          [independent_thread_pool: nil],
          [log_id: <<0>>]
        ] do
      assert_raise ArgumentError, fn -> Ortex.load("does-not-exist.onnx", options) end
    end
  end

  test "unknown execution providers never silently become CPU" do
    assert_raise RuntimeError, ~r/unknown or uncompiled execution provider/, fn ->
      Ortex.load(@path, providers: [:cpou])
    end
  end

  test "incorrect input count raises and leaves the session usable" do
    model = Ortex.load(@path, intra_threads: 1)
    assert_raise RuntimeError, ~r/Expected 1 inputs, got 0/, fn -> Ortex.run(model, {}) end
    assert_scores(model)
  end

  test "metadata and inference can be called concurrently" do
    model = Ortex.load(@path, intra_threads: 1)

    1..32
    |> Task.async_stream(
      fn _ ->
        assert {[{"features", _, [-1, 3]}], [_]} = Ortex.Native.show_session(model.reference)
        assert_scores(model)
      end,
      max_concurrency: 8
    )
    |> Enum.each(fn result -> assert {:ok, :ok} = result end)
  end

  test "native binary decoding rejects lengths and handles unaligned sub-binaries" do
    assert {:error, _} = Ortex.Native.from_binary(<<0>>, {2}, {:f, 32})
    binary = :binary.copy(<<0>>, 65)
    sub_binary = :binary.part(binary, 1, 64)
    reference = Ortex.Native.from_binary(sub_binary, {16}, {:f, 32})
    assert Ortex.Native.to_binary(reference, 32, 16) == sub_binary
  end

  defp assert_scores(model) do
    {output} = Ortex.run(model, Nx.tensor(@cases.rows, type: :f32))
    actual = output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
    assert length(actual) == length(@cases.expected)
    Enum.zip(actual, @cases.expected) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    :ok
  end
end
