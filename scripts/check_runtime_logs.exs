# Run in a separate VM, before :ortex starts, to configure its one environment.
defmodule OrtexLogCheck do
  def log(event, %{config: %{owner: owner}}), do: send(owner, {:runtime_log, event})

  def await(expected_id, minimum) do
    receive do
      {:runtime_log,
       %{
         level: severity,
         meta: %{ort_category: category, ort_log_id: ^expected_id, ort_code_location: location}
       }} ->
        unless is_binary(category) and is_binary(location), do: raise("invalid log metadata")

        if minimum == :warning and severity not in [:warning, :error, :critical],
          do: raise("runtime level filter failed")

        :ok

      {:runtime_log, _} ->
        await(expected_id, minimum)
    after
      2000 -> raise "runtime log bridge delivered no native log event for #{expected_id}"
    end
  end
end

minimum =
  case System.argv() do
    ["warning"] -> :warning
    _ -> :info
  end

Application.load(:ortex)

Application.put_env(:ortex, :environment,
  runtime_logging: true,
  runtime_log_level: minimum,
  global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]
)

{:ok, _} = Application.ensure_all_started(:ortex)
Logger.configure(level: :info)

:ok =
  :logger.add_handler(:ortex_runtime_check, OrtexLogCheck, %{
    level: :info,
    config: %{owner: self()}
  })

path = Path.expand("../test/fixtures/logging_warning.onnx", __DIR__)

for constructor <- [:file, :memory] do
  model =
    case constructor do
      :file -> Ortex.load(path, log_id: "log-file")
      :memory -> Ortex.load_from_memory(File.read!(path), log_id: "log-memory")
    end

  {sum, _} = Ortex.run(model, {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])})
  unless Nx.to_flat_list(sum) == [9.0, 6.0], do: raise("score mismatch")
  :ok = Ortex.Environment.flush_logs()
  Logger.flush()
  OrtexLogCheck.await("log-#{constructor}", minimum)
  :ok = Application.stop(:ortex)
  {:ok, _} = Application.ensure_all_started(:ortex)
end

:ok = :logger.remove_handler(:ortex_runtime_check)
IO.puts("Runtime Logger bridge and environment restart passed for both constructors")

# Suspend only the BEAM drain process. Explicit handles keep session loading
# independent while real runtime callbacks fill the bounded native queue.
environment = Ortex.Environment.get()
:ok = :sys.suspend(Ortex.Environment)

try do
  for _ <- 1..300 do
    Ortex.load(path, environment: environment)
  end

  {entries, dropped} = Ortex.Native.take_runtime_logs(environment.reference)

  unless length(entries) == 256 and dropped > 0,
    do: raise("runtime queue did not enforce its bound")

  IO.puts("Runtime log buffer bounded at 256 entries; overflow counted")
after
  :ok = :sys.resume(Ortex.Environment)
end
