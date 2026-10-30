defmodule OrtexTest do
  use ExUnit.Case
  doctest Ortex

  test "Nx.Serving with tinymodel" do
    model = Ortex.load("./models/tinymodel.onnx")

    serving = Nx.Serving.new(Ortex.Serving, model)

    # Create a batch of size 3 with {int32, float32} inputs
    batch =
      Nx.Batch.stack([
        {Nx.broadcast(0, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(0.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(1, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(1.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(2, {100}) |> Nx.as_type(:s32), Nx.broadcast(2.0, {100}) |> Nx.as_type(:f32)}
      ])

    {%Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}} =
      Nx.Serving.run(serving, batch)
  end
end
