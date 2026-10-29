# Run in its own VM: MIX_ENV=test mix run --no-start scripts/check_global_pool.exs
Application.load(:ortex)

Application.put_env(:ortex, :environment,
  global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]
)

{:ok, _} = Application.ensure_all_started(:ortex)
model = Ortex.load("test/fixtures/xgboost.onnx", providers: [:cpu])
input = Nx.tensor([[0.0, 0.0, 0.0]])
{output} = Ortex.run(model, input)
expected = output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

try do
  Ortex.load("test/fixtures/xgboost.onnx", intra_threads: 1)
  raise "expected conflicting session/global options to fail"
rescue
  e in RuntimeError ->
    unless String.contains?(e.message, "independent_thread_pool"), do: reraise(e, __STACKTRACE__)
end

independent =
  Ortex.load("test/fixtures/xgboost.onnx", independent_thread_pool: true, intra_threads: 1)

{output} = Ortex.run(independent, input)
^expected = output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
environment = Ortex.Environment.get()
:ok = Application.stop(:ortex)
{:ok, _} = Application.ensure_all_started(:ortex)
^environment = Ortex.Environment.get()
{_} = Ortex.run(model, input)
IO.puts("Global pool, independent override, and application restart checks passed")
