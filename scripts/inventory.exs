components =
  Mix.Dep.cached()
  |> Enum.map(fn dep ->
    {:hex, name, version, _, _, _, _, _} = dep.opts[:lock]

    %{
      type: "library",
      name: to_string(name),
      version: version,
      purl: "pkg:hex/#{name}@#{version}"
    }
  end)

File.mkdir_p!("dist")
File.write!("dist/hex-components.json", :json.encode(components))
