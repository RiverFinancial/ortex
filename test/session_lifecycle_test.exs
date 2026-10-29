defmodule Ortex.SessionLifecycleTest do
  use ExUnit.Case, async: false
  @path Path.expand("fixtures/xgboost.onnx", __DIR__)

  test "bad input counts return errors without poisoning the session" do
    model = Ortex.load(@path, [:cpu], 0)
    assert_raise RuntimeError, ~r/Expected 1 inputs, got 0/, fn -> Ortex.run(model, {}) end
    assert {_} = Ortex.run(model, Nx.tensor([[0.0, 0.0, 0.0]]))
  end

  test "metadata can be read while callers share a session" do
    model = Ortex.load(@path, [:cpu], 0)

    1..32
    |> Task.async_stream(
      fn _ ->
        assert {[{"features", _, [-1, 3]}], [_]} = Ortex.Native.show_session(model.reference)
        assert {_} = Ortex.run(model, Nx.tensor([[0.0, 0.0, 0.0]]))
        :ok
      end,
      max_concurrency: 8
    )
    |> Enum.each(fn result -> assert {:ok, :ok} = result end)
  end
end
