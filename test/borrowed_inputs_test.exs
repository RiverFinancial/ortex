defmodule Ortex.BorrowedInputsTest do
  use ExUnit.Case, async: true

  defp model,
    do: Ortex.load(Path.expand("fixtures/metadata_rich.onnx", __DIR__), intra_threads: 1)

  defp values(t), do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  test "a resource from an exited owner is reusable across concurrent runs" do
    input =
      Task.async(fn ->
        Nx.tensor([[[1.0, 2.0, 3.0]]]) |> Nx.backend_transfer(Ortex.Backend)
      end)
      |> Task.await()

    :erlang.garbage_collect()
    model = model()

    1..32
    |> Task.async_stream(
      fn _ ->
        {output} = Ortex.run(model, input)
        values(output)
      end,
      max_concurrency: 8
    )
    |> Enum.each(fn result -> assert result == {:ok, [1.0, 2.0, 3.0]} end)

    assert values(input) == [1.0, 2.0, 3.0]
  end

  test "sliced and reshaped resources preserve row-major input values" do
    input = Nx.iota({2, 2, 3}, type: :f32) |> Nx.backend_transfer(Ortex.Backend)
    slice = Nx.slice(input, [0, 1, 0], [2, 1, 3])
    {output} = Ortex.run(model(), slice)
    assert output.shape == {2, 1, 3}
    assert values(output) == [3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
    assert values(input) == Enum.map(0..11, &(&1 * 1.0))
  end

  test "boolean conversion owns its temporary values and leaves the input reusable" do
    model = Ortex.load(Path.expand("fixtures/metadata_bool.onnx", __DIR__), intra_threads: 1)
    input = Nx.tensor(1, type: :u8) |> Nx.backend_transfer(Ortex.Backend)

    for _ <- 1..16 do
      {output} = Ortex.run(model, input)
      assert Nx.to_number(output) == 1
    end

    assert Nx.to_number(input) == 1

    assert_raise RuntimeError, ~r/only 0 or 1/, fn ->
      Ortex.run(model, Nx.tensor(2, type: :u8))
    end

    {output} = Ortex.run(model, input)
    assert Nx.to_number(output) == 1
  end
end
