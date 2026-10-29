defmodule Ortex.InferenceTest do
  use ExUnit.Case, async: false

  @fixture Path.expand("fixtures/xgboost.onnx", __DIR__)
  @cases elem(Code.eval_file(Path.expand("fixtures/xgboost_cases.ex", __DIR__)), 0)

  test "tree-model scores match the independent XGBoost reference" do
    model = Ortex.load(@fixture, [:cpu], 0)
    assert_scores(model)
    for _ <- 1..100, do: assert_scores(model)
  end

  test "concurrent callers can share a model" do
    model = Ortex.load(@fixture, [:cpu], 0)

    1..64
    |> Task.async_stream(fn _ -> assert_scores(model) end, max_concurrency: 8, timeout: 30_000)
    |> Enum.each(fn result -> assert {:ok, :ok} = result end)
  end

  defp assert_scores(model) do
    {output} = Ortex.run(model, Nx.tensor(@cases.rows, type: :f32))
    actual = output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
    assert length(actual) == length(@cases.expected)
    Enum.zip(actual, @cases.expected) |> Enum.each(fn {a, b} -> assert_in_delta(a, b, 1.0e-6) end)
    :ok
  end
end
