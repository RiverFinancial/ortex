Code.require_file(Path.expand("../support/scenario.exs", __DIR__))
model = Task.async(fn -> Ortex.load(Ortex.Scenario.fixture(), [:cpu]) end) |> Task.await(30_000)
:erlang.garbage_collect()
expected = Ortex.Scenario.score(model)

1..16
|> Task.async_stream(fn _ -> Ortex.Scenario.score(model) end, max_concurrency: 4)
|> Enum.each(fn result ->
  unless result == {:ok, expected}, do: raise("result changed after model owner exited")
end)
