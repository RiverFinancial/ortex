# A fresh VM proves config-path precedence, real provider inheritance, and restart.
Application.load(:ortex)
runtime = System.fetch_env!("ORT_DYLIB_PATH")

options = [
  name: "River Ortex tests",
  runtime_path: runtime,
  providers: [{:cpu, %{"arena_allocator" => "true"}}],
  runtime_logging: true,
  runtime_log_level: :info,
  global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]
]

Application.put_env(:ortex, :environment, options)
System.put_env("ORT_DYLIB_PATH", "/does-not-exist/config-path-must-win")
{:ok, _} = Application.ensure_all_started(:ortex)
environment = Ortex.Environment.get()
path = Path.expand("../test/fixtures/named_io.onnx", __DIR__)

:ok = :sys.suspend(Ortex.Environment)

try do
  for load <- [
        fn opts -> Ortex.load(path, opts) end,
        fn opts -> Ortex.load_from_memory(File.read!(path), opts) end
      ] do
    for {overrides, expected_arena} <- [
          {[], true},
          {[providers: [{:cpu, %{"arena_allocator" => "false"}}]], false}
        ] do
      Ortex.Native.take_runtime_logs(environment.reference)
      model = load.([environment: environment] ++ overrides)
      {sum, _} = Ortex.run(model, {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])})
      unless Nx.to_flat_list(sum) == [9.0, 6.0], do: raise("score mismatch")
      {entries, _dropped} = Ortex.Native.take_runtime_logs(environment.reference)

      arena =
        Enum.any?(entries, fn {_, _, _, _, message} -> String.contains?(message, "BFCArena") end)

      unless arena == expected_arena,
        do: raise("environment provider inheritance/override failed")
    end
  end
after
  :ok = :sys.resume(Ortex.Environment)
end

:ok = Application.stop(:ortex)
{:ok, _} = Application.ensure_all_started(:ortex)

unless Ortex.Environment.get().reference == environment.reference,
  do: raise("environment resource changed")

# The legacy path arity still requests CPU explicitly.
model = Ortex.load(path)
{_, _} = Ortex.run(model, {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])})
:ok = Application.stop(:ortex)
Application.put_env(:ortex, :environment, Keyword.put(options, :name, "changed"))
{:error, reason} = Application.ensure_all_started(:ortex)

unless inspect(reason) =~ "configuration changed",
  do: raise("changed environment was not rejected")

Application.put_env(:ortex, :environment, options)
{:ok, _} = Application.ensure_all_started(:ortex)

IO.puts(
  "Environment path/name/providers, constructor overrides, legacy load, and restart checks passed"
)
