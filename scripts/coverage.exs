Mix.ensure_application!(:tools)
{:ok, _} = :cover.start()
:ok = :cover.import(~c"cover/library.coverdata")

rows =
  :cover.imported_modules()
  |> Enum.reject(&(&1 == Ortex.Native))
  |> Enum.sort()
  |> Enum.map(fn module ->
    {:ok, lines} = :cover.analyse(module, :coverage, :line)

    # Match Mix's line metric: exclude generated line 0 and combine counters
    # sharing a source line. Any covered counter makes that line covered.
    unique =
      Enum.reduce(lines, %{}, fn
        {{_, 0}, _}, acc -> acc
        {{_, line}, {hit, _}}, acc -> Map.update(acc, line, hit > 0, &(&1 or hit > 0))
      end)

    covered = Enum.count(unique, fn {_, hit} -> hit end)
    missed = map_size(unique) - covered
    {:ok, _} = :cover.analyse_to_file(module, ~c"cover/#{module}.html", [:html])
    {module, covered, missed}
  end)

if rows == [], do: raise("no library coverage data")

percentage = fn hit, total ->
  if total == 0, do: "n/a", else: :erlang.float_to_binary(hit / total * 100, decimals: 2) <> "%"
end

covered = Enum.sum(for {_, hit, _} <- rows, do: hit)
total = Enum.sum(for {_, hit, missed} <- rows, do: hit + missed)

report =
  "# Elixir library coverage\n\nExcludes the NIF interface. Rust and fresh-VM scenarios are outside this report.\n\n" <>
    "| Module | Covered | Executable | Coverage |\n|---|---:|---:|---:|\n" <>
    Enum.map_join(rows, "\n", fn {module, hit, missed} ->
      "| #{inspect(module)} | #{hit} | #{hit + missed} | #{percentage.(hit, hit + missed)} |"
    end) <>
    "\n| **Total** | #{covered} | #{total} | **#{percentage.(covered, total)}** |\n"

File.write!("cover/summary.md", report)
IO.puts(report)
