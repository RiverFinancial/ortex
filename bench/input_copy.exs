# MIX_ENV=test ORTEX_BUILD=true mix run bench/input_copy.exs [iterations] [rows]
# A synthetic identity graph isolates transfer/input-copy cost from model compute.
iterations = System.argv() |> Enum.at(0, "100") |> String.to_integer()
rows = System.argv() |> Enum.at(1, "1024") |> String.to_integer()
if iterations < 1 or rows < 1, do: raise(ArgumentError, "counts must be positive")
model = Ortex.load(Path.expand("../test/fixtures/metadata_rich.onnx", __DIR__), intra_threads: 1)
binary = Nx.broadcast(1.25, {rows, 256, 3})
resource = Nx.backend_transfer(binary, Ortex.Backend)

for {mode, input} <- [binary_backend: binary, reused_ortex_resource: resource] do
  run = fn ->
    {output} = Ortex.run(model, input)
    if output.shape != input.shape, do: raise("shape mismatch")
  end

  for _ <- 1..10, do: run.()
  :erlang.garbage_collect()
  timings = for(_ <- 1..iterations, do: elem(:timer.tc(run), 0)) |> Enum.sort()

  IO.inspect(%{
    mode: mode,
    input_bytes: Nx.byte_size(input),
    iterations: iterations,
    p50_us: Enum.at(timings, div(iterations, 2)),
    p99_us: Enum.at(timings, min(iterations - 1, trunc(iterations * 0.99)))
  })
end
