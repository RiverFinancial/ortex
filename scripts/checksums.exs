# elixir scripts/checksums.exs dist [--all]
[directory | options] = System.argv()
files = directory |> Path.join("*.tar.gz") |> Path.wildcard() |> Enum.sort()
if files == [], do: raise("no NIF archives found")
version = Regex.run(~r/@version "([^"]+)"/, File.read!("mix.exs")) |> Enum.at(1)

targets =
  ~w(aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-unknown-linux-musl aarch64-unknown-linux-musl x86_64-pc-windows-msvc)

expected =
  Enum.map(targets, fn target ->
    {prefix, suffix} =
      if String.contains?(target, "windows"), do: {"", "dll"}, else: {"lib", "so"}

    "#{prefix}ortex-v#{version}-nif-2.16-#{target}.#{suffix}.tar.gz"
  end)

names = Enum.map(files, &Path.basename/1)
if names -- expected != [], do: raise("unexpected target or version in artifact directory")
if "--all" in options and expected -- names != [], do: raise("expected all seven target archives")

checksums =
  Map.new(files, fn path ->
    digest = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    {Path.basename(path), "sha256:" <> digest}
  end)

File.write!(
  "checksum-Elixir.Ortex.Native.exs",
  inspect(checksums, pretty: true, limit: :infinity) <> "\n"
)

IO.puts("Generated checksums for #{map_size(checksums)} archives")
