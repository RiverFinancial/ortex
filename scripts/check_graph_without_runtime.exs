# MIX_ENV=test mix run --no-start scripts/check_graph_without_runtime.exs
if Process.whereis(Ortex.Environment), do: raise("Ortex unexpectedly started")
path = Path.expand("../test/fixtures/graph_summary.onnx", __DIR__)
summary = Ortex.Graph.from_file(path)

unless summary.node_count == 6 and summary.initializer_bytes == %{known: 94, unknown_count: 1},
  do: raise("graph summary mismatch")

if Process.whereis(Ortex.Environment), do: raise("graph inspection started the runtime")
IO.puts("ONNX graph inspection works without starting an environment or opening external weights")
