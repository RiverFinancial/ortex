# MIX_ENV=test ORTEX_BUILD=true mix run bench/lifecycle.exs [runs] [load_dispose_cycles]
# RSS includes BEAM/allocator retention. This is a diagnostic, not a leak proof.
runs = System.argv() |> Enum.at(0, "100000") |> String.to_integer()
cycles = System.argv() |> Enum.at(1, "500") |> String.to_integer()
if runs < 10 or cycles < 10, do: raise(ArgumentError, "counts must be at least ten")
path = Path.expand("../test/fixtures/xgboost.onnx", __DIR__)
bytes = File.read!(path)
{%{rows: rows, expected: expected}, _} = Code.eval_file(Path.rootname(path) <> "_cases.ex")
input = Nx.tensor(rows, type: :f32) |> Nx.backend_transfer(Ortex.Backend)
options = [intra_threads: 1, intra_op_spinning: false]

score = fn model ->
  {output} = Ortex.run(model, input)
  actual = output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

  unless length(actual) == length(expected) and
           Enum.all?(Enum.zip(actual, expected), fn {a, b} -> abs(a - b) < 1.0e-6 end),
         do: raise("golden-score mismatch")
end

sample = fn phase, count ->
  :erlang.garbage_collect()

  rss =
    case System.cmd("ps", ["-o", "rss=", "-p", System.pid()]) do
      {value, 0} -> value |> String.trim() |> String.to_integer() |> Kernel.*(1024)
      _ -> nil
    end

  IO.inspect(%{phase: phase, count: count, rss_bytes: rss, beam_bytes: :erlang.memory(:total)})
end

model = Ortex.load(path, options)
for _ <- 1..1000, do: score.(model)
sample.(:warmup, 1000)

for i <- 1..runs do
  score.(model)
  if rem(i, max(div(runs, 10), 1)) == 0, do: sample.(:repeated_run, i)
end

# Each load is owned by a fresh short-lived process so resources become eligible
# for destruction at a known boundary. Alternate both copying constructors.
for i <- 1..cycles do
  Task.async(fn ->
    loaded =
      if rem(i, 2) == 0,
        do: Ortex.load(path, options),
        else: Ortex.load_from_memory(bytes, options)

    score.(loaded)
  end)
  |> Task.await(60_000)

  if rem(i, max(div(cycles, 10), 1)) == 0, do: sample.(:load_dispose, i)
end

IO.inspect(%{
  runs: runs,
  load_dispose_cycles: cycles,
  constructors: [:file, :memory],
  note: "Golden scores checked on every call; RSS is diagnostic, not a leak assertion."
})
