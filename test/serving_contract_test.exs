defmodule Ortex.ServingContractTest do
  use ExUnit.Case, async: true

  defp load(name), do: Ortex.load(Path.expand("fixtures/#{name}.onnx", __DIR__), intra_threads: 1)

  test "named tensor batching preserves model names and golden scores" do
    model = load("xgboost")
    serving = Nx.Serving.new(Ortex.Serving, model)

    batch =
      Nx.Batch.stack([
        %{"features" => Nx.tensor([0.0, 0.0, 0.0])},
        %{"features" => Nx.tensor([1.0, 1.0, 1.0])}
      ])

    result = Nx.Serving.run(serving, batch)
    assert map_size(result) == 1
    [output] = Map.values(result)
    [first, second] = Nx.to_flat_list(output)
    assert_in_delta first, 0.02070039138197899, 1.0e-6
    assert_in_delta second, 1.1609605550765991, 1.0e-6
  end

  test "process serving shares a numeric model across batched callers" do
    model = load("xgboost")
    serving = Nx.Serving.new(Ortex.Serving, model)
    name = Module.concat(__MODULE__, ProcessServing)
    start_supervised!({Nx.Serving, serving: serving, name: name, batch_size: 4, batch_timeout: 5})

    1..8
    |> Task.async_stream(
      fn _ ->
        {output} = Nx.Serving.batched_run(name, Nx.Batch.stack([{Nx.tensor([0.0, 0.0, 0.0])}]))
        Nx.to_flat_list(output)
      end,
      max_concurrency: 4
    )
    |> Enum.each(fn {:ok, [score]} -> assert_in_delta score, 0.02070039138197899, 1.0e-6 end)
  end

  test "container, string, and scalar models fail with an actionable batching error" do
    for name <- ["metadata_zipmap", "metadata_string", "numeric_scalar", "sequence_tensors"] do
      serving = Nx.Serving.new(Ortex.Serving, load(name))
      batch = Nx.Batch.stack([{Nx.tensor([1.0, 2.0])}])

      assert_raise ArgumentError, ~r/numeric tensor inputs\/outputs with a batch axis/, fn ->
        Nx.Serving.run(serving, batch)
      end
    end
  end
end
