defmodule Ortex.StringTensorTest do
  use ExUnit.Case, async: true
  alias Ortex.StringTensor

  defp path(name), do: Path.expand("fixtures/#{name}.onnx", __DIR__)
  defp load(name), do: Ortex.load(path(name), intra_threads: 1)

  test "Unicode strings and empty strings round-trip with a matrix shape" do
    input = StringTensor.new(["hello", "π", "", "世界"], {2, 2})
    assert {^input} = Ortex.run(load("string_matrix"), input)

    assert %{"output" => ^input} =
             Ortex.run(load("string_matrix"), %{"input" => input}, outputs: ["output"])
  end

  test "string scalars and empty tensors retain their shape" do
    scalar = StringTensor.new(["π"], {})
    assert {^scalar} = Ortex.run(load("string_scalar"), scalar)
    empty = StringTensor.new([], {0})
    assert {^empty} = Ortex.run(load("metadata_string"), empty)
  end

  test "memory-loaded string result survives source process and model disposal" do
    input = StringTensor.new(["one", "two"], {1, 2})

    result =
      Task.async(fn ->
        model = Ortex.load_from_memory(File.read!(path("string_matrix")), intra_threads: 1)
        Ortex.run(model, input)
      end)
      |> Task.await()

    :erlang.garbage_collect()
    assert result == {input}
  end

  test "numeric and boolean scalars remain Nx tensors" do
    {number} = Ortex.run(load("numeric_scalar"), Nx.tensor(1.25, type: :f64))
    assert number.shape == {}
    assert Nx.to_number(number) == 1.25
    {boolean} = Ortex.run(load("metadata_bool"), Nx.tensor(1, type: :u8))
    assert boolean.shape == {}
    assert boolean.type == {:u, 8}
    assert Nx.to_number(boolean) == 1
  end

  test "invalid values and forged string tensor structs are rejected" do
    for {data, shape} <- [
          {["x"], {2}},
          {[], {}},
          {["x"], {-1}},
          {["x"], [1]},
          {["x"], {1.0}},
          {[<<0>>], {1}},
          {[<<255>>], {1}},
          {[123], {1}}
        ] do
      assert_raise ArgumentError, fn -> StringTensor.new(data, shape) end
    end

    assert_raise ArgumentError, fn ->
      Ortex.run(load("metadata_string"), %StringTensor{data: ["x"], shape: {3}})
    end
  end

  test "native boundary validates string shape and recovers" do
    model = load("metadata_string")

    for input <- [{[2], ["only one"]}, {[9_223_372_036_854_775_808], []}, {[1], [<<0>>]}] do
      assert {:error, message} =
               Ortex.Native.run_values(model.reference, [input], %{outputs: nil, tag: nil})

      assert is_binary(message)
    end

    input = StringTensor.new(["valid"], {1})
    assert {^input} = Ortex.run(model, input)
  end
end
