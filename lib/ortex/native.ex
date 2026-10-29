defmodule Ortex.Native do
  @moduledoc false
  config = Application.compile_env(:ortex, __MODULE__, [])

  use Rustler,
    otp_app: :ortex,
    crate: :ortex,
    default_features: false,
    features: Keyword.get(config, :features, ["load-dynamic", "api-17"])

  def init(path, providers, optimization) do
    {_, options} = Ortex.Options.legacy!(providers, optimization)
    init_with_options(path, Ortex.Environment.get().reference, options)
  end

  def init_with_options(_path, _environment, _options), do: :erlang.nif_error(:nif_not_loaded)
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
