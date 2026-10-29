defmodule Ortex.Native do
  @moduledoc false

  version = Mix.Project.config()[:version]
  config = Application.compile_env(:ortex, __MODULE__, [])
  features = Keyword.get(config, :features, ["load-dynamic", "api-17"])

  use RustlerPrecompiled,
    otp_app: :ortex,
    crate: :ortex,
    version: version,
    base_url: "https://github.com/RiverFinancial/ortex/releases/download/v#{version}",
    targets:
      ~w(aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-unknown-linux-musl aarch64-unknown-linux-musl x86_64-pc-windows-msvc),
    nif_versions: ["2.16"],
    features: features,
    default_features: false,
    force_build:
      System.get_env("ORTEX_BUILD") in ["1", "true"] or
        features != ["load-dynamic", "api-17"]

  # When loading a NIF module, dummy clauses for all NIF function are required.
  # NIF dummies usually just error out when called when the NIF is not loaded, as that should never normally happen.
  # Compatibility wrapper for consumers that called this internal entry point.
  def init(path, providers, optimization) do
    {_, options} = Ortex.Options.legacy!(providers, optimization)
    init_with_options(path, Ortex.Environment.get().reference, options)
  end

  def init_with_options(_path, _environment, _options), do: :erlang.nif_error(:nif_not_loaded)
  def init_from_memory(_bytes, _environment, _options), do: :erlang.nif_error(:nif_not_loaded)
  def environment(_global_thread_pool), do: :erlang.nif_error(:nif_not_loaded)

  def run(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def from_binary(_bin, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def to_binary(_reference, _bits, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def show_session(_model), do: :erlang.nif_error(:nif_not_loaded)

  def slice(_tensor, _start_indicies, _lengths, _strides),
    do: :erlang.nif_error(:nif_not_loaded)

  def reshape(_tensor, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def concatenate(_tensors_refs, _type, _axis), do: :erlang.nif_error(:nif_not_loaded)
end
