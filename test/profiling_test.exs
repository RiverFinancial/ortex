defmodule Ortex.ProfilingTest do
  use ExUnit.Case, async: true
  @path Path.expand("fixtures/named_io.onnx", __DIR__)

  @tag :tmp_dir
  test "session profiling flushes a Chrome trace and leaves inference usable", %{tmp_dir: dir} do
    prefix = Path.join(dir, "profile")
    model = Ortex.load(@path, profiling: prefix, intra_threads: 1)
    inputs = {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])}
    {_, _} = Ortex.run(model, inputs, tag: "profile-test")
    path = Ortex.end_profiling(model)
    assert String.starts_with?(path, prefix)
    trace = File.read!(path)
    assert trace |> String.trim() |> String.starts_with?("[")
    assert trace |> String.trim() |> String.ends_with?("]")
    assert trace =~ "model_run"
    assert trace =~ "session_initialization"
    {sum, _} = Ortex.run(model, inputs)
    assert Nx.to_flat_list(sum) == [9.0, 6.0]
  end

  @tag :tmp_dir
  test "memory-loaded profiling supports selected outputs", %{tmp_dir: dir} do
    model =
      Ortex.load_from_memory(File.read!(@path),
        profiling: Path.join(dir, "memory"),
        intra_threads: 1
      )

    result =
      Ortex.run(model, %{"z_left" => Nx.tensor([7.0, 5.0]), "a_right" => Nx.tensor([2.0, 1.0])},
        outputs: ["Δ"]
      )

    assert Map.keys(result) == ["Δ"]
    assert Ortex.end_profiling(model) |> File.read!() =~ "model_run"
  end

  test "ending disabled profiling returns the runtime's empty path" do
    model = Ortex.load(@path, intra_threads: 1)
    assert Ortex.end_profiling(model) == ""
  end
end
