defmodule Ortex.Scenario do
  def fixture, do: Path.expand("../../models/tinymodel.onnx", __DIR__)

  def score(model) do
    inputs = {Nx.broadcast(0, {1, 100}) |> Nx.as_type(:s32), Nx.broadcast(0.0, {1, 100})}
    Ortex.run(model, inputs) |> Tuple.to_list() |> Enum.map(&Nx.to_flat_list/1)
  end
end

{:ok, _} = Application.ensure_all_started(:ortex)
