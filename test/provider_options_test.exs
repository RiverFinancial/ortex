defmodule Ortex.ProviderOptionsTest do
  use ExUnit.Case, async: true
  @path Path.expand("fixtures/xgboost.onnx", __DIR__)

  defp scores(model) do
    {output} = Ortex.run(model, Nx.tensor([[0.0, 0.0, 0.0]]))
    output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
  end

  test "CPU arena option maps preserve scores across constructors" do
    expected = scores(Ortex.load(@path, intra_threads: 1))

    for arena <- ["true", "false", "1", "0"] do
      options = [providers: [{:cpu, %{"arena_allocator" => arena}}], intra_threads: 1]
      assert scores(Ortex.load(@path, options)) == expected
      assert scores(Ortex.load_from_memory(File.read!(@path), options)) == expected
    end
  end

  test "unknown keys and malformed values fail even with provider fallback enabled" do
    for options <- [%{"typo" => "true"}, %{"arena_allocator" => "maybe"}] do
      assert_raise RuntimeError, ~r/provider option/, fn ->
        Ortex.load(@path, providers: [{:cpu, options}], error_on_failure: false)
      end
    end

    assert [_] = scores(Ortex.load(@path, intra_threads: 1))
  end

  test "option maps require binary keys and values without NUL bytes" do
    for options <- [
          %{arena_allocator: "true"},
          %{"arena_allocator" => true},
          %{"" => "true"},
          %{("bad" <> <<0>>) => "true"},
          %{"arena_allocator" => <<0>>},
          %{"arena_allocator" => <<255>>}
        ] do
      assert_raise ArgumentError, fn -> Ortex.load(@path, providers: [{:cpu, options}]) end
    end
  end

  test "availability reports independent build and runtime support" do
    providers = Ortex.execution_providers()
    assert providers.cpu == %{compiled: true, available: true}

    assert Enum.sort(Map.keys(providers)) ==
             Enum.sort([:cpu, :cuda, :tensorrt, :coreml, :directml, :acl, :onednn, :rocm])

    for {_name, %{compiled: compiled, available: available}} <- providers do
      assert is_boolean(compiled)
      if compiled, do: assert(is_boolean(available)), else: assert(is_nil(available))
    end
  end

  test "uncompiled providers never silently fall through to CPU" do
    for {name, %{compiled: false}} <- Ortex.execution_providers() do
      assert_raise RuntimeError, ~r/unknown or uncompiled/, fn ->
        Ortex.load(@path, providers: [{name, %{}}], error_on_failure: false)
      end
    end
  end

  test "native provider options reject unknown keys" do
    {_, options} = Ortex.Options.validate!(providers: [:cpu], intra_threads: 1)
    options = %{options | providers: [{:cpu, %{"unknown" => "1"}}]}

    assert {:error, message} =
             Ortex.Native.init_with_options(@path, Ortex.Environment.get().reference, options)

    assert message =~ "unknown provider option"
  end
end
