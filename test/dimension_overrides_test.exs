defmodule Ortex.DimensionOverridesTest do
  use ExUnit.Case, async: true
  @path Path.expand("fixtures/metadata_rich.onnx", __DIR__)

  test "symbolic overrides specialize file and memory models with equal results" do
    options = [dimension_overrides: %{"batch" => 2}, intra_threads: 1]
    file = Ortex.load(@path, options)
    memory = Ortex.load_from_memory(File.read!(@path), options)

    for model <- [file, memory] do
      assert [%{type: {:tensor, {:f, 32}, {2, nil, 3}}}] = Ortex.metadata(model).inputs
      input = Nx.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
      {output} = Ortex.run(model, input)
      assert output.shape == {2, 1, 3}
      assert Nx.to_flat_list(output) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end
  end

  test "specialization rejects a mismatching batch without changing another session" do
    fixed = Ortex.load(@path, dimension_overrides: %{"batch" => 2}, intra_threads: 1)
    input = Nx.tensor([[[1.0, 2.0, 3.0]]])
    assert_raise RuntimeError, fn -> Ortex.run(fixed, input) end
    dynamic = Ortex.load(@path, intra_threads: 1)
    assert [%{type: {:tensor, _, {"batch", nil, 3}}}] = Ortex.metadata(dynamic).inputs
    assert {%Nx.Tensor{shape: {1, 1, 3}}} = Ortex.run(dynamic, input)
  end

  test "invalid override names, sizes, and containers fail early" do
    for overrides <- [
          [batch: 2],
          %{batch: 2},
          %{"batch" => -1},
          %{"batch" => 1.5},
          %{"batch" => 9_223_372_036_854_775_808},
          %{"" => 2},
          %{<<0>> => 2}
        ] do
      assert_raise ArgumentError, fn -> Ortex.load(@path, dimension_overrides: overrides) end
    end
  end

  test "native dimension validation remains recoverable" do
    {_, options} = Ortex.Options.validate!(intra_threads: 1)
    options = %{options | dimension_overrides: %{"batch" => -1}}

    assert {:error, message} =
             Ortex.Native.init_with_options(@path, Ortex.Environment.get().reference, options)

    assert message =~ "dimension overrides"
    assert %Ortex.Model{} = Ortex.load(@path, intra_threads: 1)
  end
end
