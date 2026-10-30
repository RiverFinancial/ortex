iterations = System.argv() |> List.first("2000") |> String.to_integer()
if iterations < 1, do: raise("iterations must be positive")
root = Path.expand("../test/fixtures", __DIR__)
cases = File.read!(Path.join(root, "xgboost.json")) |> :json.decode()
model = Ortex.load(Path.join(root, "xgboost.onnx"), [:cpu], 0)
input = Nx.tensor(cases["inputs"], type: :f32)
expected = List.flatten(cases["outputs"])

run = fn ->
  {output} = Ortex.run(model, input)
  actual = Nx.to_flat_list(output)

  unless length(actual) == length(expected) and
           Enum.all?(Enum.zip(actual, expected), fn {a, b} -> abs(a - b) < 1.0e-6 end),
         do: raise("benchmark reference mismatch")
end

Enum.each(1..100, fn _ -> run.() end)
{elapsed, _} = :timer.tc(fn -> Enum.each(1..iterations, fn _ -> run.() end) end)

IO.inspect(%{
  iterations: iterations,
  calls_per_second: iterations * 1_000_000 / elapsed,
  elixir: System.version(),
  otp: System.otp_release(),
  runtime: System.fetch_env!("ORT_DYLIB_PATH"),
  workload: "synthetic XGBoost; public call including transfers; CPU; optimization disabled"
})
