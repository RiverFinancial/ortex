defmodule Ortex.ModelsTest do
  use ExUnit.Case, async: false
  @moduletag :model

  for name <- ~w(xgboost iris mnist) do
    test "#{name} matches its prepared reference results" do
      name = unquote(name)
      root = Path.expand("../fixtures", __DIR__)
      cases = File.read!(Path.join(root, name <> ".json")) |> :json.decode()
      model = Ortex.load(Path.join(root, name <> ".onnx"), [:cpu], 0)
      outputs = Ortex.run(model, Nx.tensor(cases["inputs"], type: :f32)) |> Tuple.to_list()
      assert length(outputs) == length(cases["outputs"])

      Enum.zip(outputs, cases["outputs"])
      |> Enum.each(fn {tensor, expected} ->
        actual = Nx.to_flat_list(tensor)
        expected = List.flatten(expected)
        assert length(actual) == length(expected)
        Enum.zip(actual, expected) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-5 end)
      end)
    end
  end
end
