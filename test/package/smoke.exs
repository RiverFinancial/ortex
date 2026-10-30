model = Ortex.load(System.fetch_env!("ORTEX_FIXTURE_PATH"), [:cpu], 0)
cases = File.read!(System.fetch_env!("ORTEX_CASES_PATH")) |> :json.decode()
{result} = Ortex.run(model, Nx.tensor(cases["inputs"], type: :f32))
actual = Nx.to_flat_list(result)
expected = List.flatten(cases["outputs"])

unless length(actual) == length(expected) and
         Enum.all?(Enum.zip(actual, expected), fn {a, b} -> abs(a - b) < 1.0e-6 end),
       do: raise("installed package produced incorrect reference scores")

IO.puts("Installed package passed #{System.fetch_env!("ORTEX_PACKAGE_MODE")} inference")
