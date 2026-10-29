# Compare with four normal, four dirty CPU, and four dirty I/O schedulers.
# Build/run each mode in fresh VMs; see docs/schedulers.md.
mode = Ortex.Native.inference_scheduler()
expected = if System.get_env("ORTEX_BENCHMARK_DIRTY_CPU") == "1", do: :dirty_cpu, else: :dirty_io
if mode != expected, do: raise("stale NIF: force a native rebuild before measuring")
options = [intra_threads: 1, intra_op_spinning: false]
path = Path.expand("../test/fixtures/xgboost.onnx", __DIR__)
{%{rows: rows, expected: scores}, _} = Code.eval_file(Path.rootname(path) <> "_cases.ex")
model = Ortex.load(path, options)
input = Nx.tensor(rows, type: :f32) |> Nx.backend_transfer(Ortex.Backend)

run = fn ->
  {output} = Ortex.run(model, input)
  actual = Nx.to_flat_list(output)

  unless Enum.all?(Enum.zip(actual, scores), fn {a, b} -> abs(a - b) < 1.0e-6 end),
    do: raise("score mismatch")
end

for _ <- 1..200, do: run.()
{serial_us, timings} = :timer.tc(fn -> for _ <- 1..2000, do: elem(:timer.tc(run), 0) end)

{concurrent_us, _} =
  :timer.tc(fn ->
    1..2000
    |> Task.async_stream(fn _ -> run.() end, max_concurrency: 8)
    |> Enum.each(fn {:ok, _} -> :ok end)
  end)

percentile = fn values, fraction ->
  values |> Enum.sort() |> Enum.at(min(length(values) - 1, trunc(length(values) * fraction)))
end

# Native.from_binary always uses dirty CPU. Its latency reveals whether queued
# inference mutex waiters can monopolize that pool and delay unrelated work.
binary = :binary.copy(<<0>>, 2_097_152)

probe = fn ->
  case Ortex.Native.from_binary(binary, {524_288}, {:f, 32}) do
    {:error, error} -> raise error
    _resource -> :ok
  end
end

for _ <- 1..5, do: probe.()
copy_baseline = for _ <- 1..10, do: elem(:timer.tc(probe), 0)
loop = Ortex.load(Path.expand("../test/fixtures/cancellable_loop.onnx", __DIR__), options)
iterations = 20_000

refs =
  for value <- [iterations, 0] do
    %Nx.Tensor{data: %Ortex.Backend{ref: reference}} =
      Nx.tensor(value, type: :s64) |> Nx.backend_transfer(Ortex.Backend)

    reference
  end

pool_width =
  case mode do
    :dirty_cpu -> :erlang.system_info(:dirty_cpu_schedulers_online)
    :dirty_io -> :erlang.system_info(:dirty_io_schedulers)
  end

workers = max(pool_width * 4, 16)
owner = self()

wait_claimed = fn recurse, controls, deadline ->
  if Enum.count(controls, &Ortex.RunControl.started?/1) < pool_width do
    if System.monotonic_time(:millisecond) >= deadline,
      do: raise("inference pool did not become occupied")

    Process.sleep(1)
    recurse.(recurse, controls, deadline)
  end
end

contended =
  for _ <- 1..10 do
    controls = for _ <- 1..workers, do: Ortex.RunControl.new(loop)

    tasks =
      Enum.map(controls, fn control ->
        Task.async(fn ->
          send(owner, {:ready, self()})

          receive do
            :go -> :ok
          end

          Ortex.Native.run_controlled(loop.reference, refs, control.reference)
        end)
      end)

    for task <- tasks do
      pid = task.pid

      receive do
        {:ready, ^pid} -> :ok
      after
        5000 -> raise("worker did not become ready")
      end
    end

    for task <- tasks, do: send(task.pid, :go)
    wait_claimed.(wait_claimed, controls, System.monotonic_time(:millisecond) + 10_000)
    {latency, _} = :timer.tc(probe)

    for task <- tasks do
      [{:tensor, {reference, [], :s, 64}}] = Task.await(task, 30_000)

      unless Ortex.Native.to_binary(reference, 64, 1) == <<iterations::signed-native-64>>,
        do: raise("loop result mismatch")
    end

    latency
  end

IO.inspect(
  %{
    mode: mode,
    schedulers: :erlang.system_info(:schedulers_online),
    dirty_cpu: :erlang.system_info(:dirty_cpu_schedulers_online),
    dirty_io: :erlang.system_info(:dirty_io_schedulers),
    tree_serial_calls_per_second: 2_000_000_000 / serial_us,
    tree_concurrent_calls_per_second: 2_000_000_000 / concurrent_us,
    tree_p50_us: percentile.(timings, 0.5),
    tree_p99_us: percentile.(timings, 0.99),
    copy_baseline_p50_us: percentile.(copy_baseline, 0.5),
    contended_copy_p50_us: percentile.(contended, 0.5),
    contended_copy_p99_us: percentile.(contended, 0.99),
    contended_copy_samples_us: contended,
    queued_calls_per_round: workers,
    loop_iterations: iterations,
    note: "Diagnostic on this host; one contended CPU session, no accelerator or placement claim."
  },
  limit: :infinity,
  charlists: :as_lists
)
