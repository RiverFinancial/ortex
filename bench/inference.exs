# MIX_ENV=test mix run bench/inference.exs [iterations] [concurrency]
# Measures the whole public call, including Nx conversions and mutex wait time.
iterations = System.argv() |> Enum.at(0, "2000") |> String.to_integer()
concurrency = System.argv() |> Enum.at(1, "8") |> String.to_integer()
if iterations < 1 or concurrency < 1, do: raise(ArgumentError, "counts must be positive")

fixture = Path.expand("../test/fixtures/xgboost.onnx", __DIR__)

%{rows: rows} =
  fixture |> Path.rootname() |> Kernel.<>("_cases.ex") |> Code.eval_file() |> elem(0)

options = System.get_env("ORTEX_BENCH_OPTIONS", "[[:cpu], 0]") |> Code.eval_string() |> elem(0)
model = apply(Ortex, :load, [fixture | options])
input = Nx.tensor(rows, type: :f32)

run = fn ->
  {output} = Ortex.run(model, input)
  output |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
end

rss = fn ->
  case System.cmd("ps", ["-o", "rss=", "-p", System.pid()]) do
    {text, 0} -> text |> String.trim() |> String.to_integer() |> Kernel.*(1024)
    _ -> nil
  end
end

for _ <- 1..200, do: run.()
:erlang.garbage_collect()
before = rss.()
{serial_us, timings} = :timer.tc(fn -> for _ <- 1..iterations, do: elem(:timer.tc(run), 0) end)

{concurrent_us, _} =
  :timer.tc(fn ->
    1..iterations
    |> Task.async_stream(fn _ -> run.() end, max_concurrency: concurrency, timeout: 60_000)
    |> Enum.each(fn {:ok, _} -> :ok end)
  end)

samples =
  for _ <- 1..5 do
    for _ <- 1..iterations, do: run.()
    :erlang.garbage_collect()
    rss.()
  end

sorted = Enum.sort(timings)

result = %{
  elixir: System.version(),
  otp: System.otp_release(),
  arch: :erlang.system_info(:system_architecture),
  iterations: iterations,
  concurrency: concurrency,
  options: options,
  serial_calls_per_second: iterations * 1_000_000 / serial_us,
  concurrent_calls_per_second: iterations * 1_000_000 / concurrent_us,
  p50_us: Enum.at(sorted, div(iterations, 2)),
  p99_us: Enum.at(sorted, trunc(iterations * 0.99)),
  rss_before_bytes: before,
  rss_samples_bytes: samples,
  note: "RSS includes BEAM and allocator retention; growth is diagnostic, not proof of a leak."
}

IO.inspect(result, pretty: true, limit: :infinity)
