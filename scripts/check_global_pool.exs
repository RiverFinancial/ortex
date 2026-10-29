# Run in its own VM: MIX_ENV=test mix run --no-start scripts/check_global_pool.exs
Application.load(:ortex)

Application.put_env(:ortex, :environment,
  global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]
)

{:ok, _} = Application.ensure_all_started(:ortex)
path = Path.expand("../test/fixtures/xgboost.onnx", __DIR__)
input = Nx.tensor([[0.0, 0.0, 0.0]])

score = fn model ->
  {output} = Ortex.run(model, input)
  output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
end

for {name, load} <- [
      {:file, fn options -> Ortex.load(path, options) end},
      {:memory, fn options -> Ortex.load_from_memory(File.read!(path), options) end}
    ] do
  model = load.(providers: [:cpu])
  expected = score.(model)

  try do
    load.(intra_threads: 1)
    raise "expected conflicting session/global options to fail"
  rescue
    e in RuntimeError ->
      unless String.contains?(e.message, "independent_thread_pool"),
        do: reraise(e, __STACKTRACE__)
  end

  independent = load.(independent_thread_pool: true, intra_threads: 1)
  ^expected = score.(independent)
  environment = Ortex.Environment.get()
  :ok = Application.stop(:ortex)
  {:ok, _} = Application.ensure_all_started(:ortex)
  ^environment = Ortex.Environment.get()
  ^expected = score.(model)
  IO.puts("Global pool, independent override, and restart checks passed for #{name} loading")
end
