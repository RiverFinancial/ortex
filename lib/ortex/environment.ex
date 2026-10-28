defmodule Ortex.Environment do
  @moduledoc """
  The VM's ONNX Runtime environment. Configure it before starting `:ortex`:

      config :ortex, :environment,
        global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]

  Sessions share this pool by default. A session may explicitly request
  `independent_thread_pool: true`. Without global pool configuration, ONNX Runtime
  creates its usual per-session pools. Session thread-count and spinning controls
  require an independent pool when a global pool is configured.

  ort rc.15 permits one environment creation per process lifetime. Its resource is
  retained in the BEAM across application restarts, rather than in a Rust global.
  Changing environment configuration requires a fresh VM. `get/0` returns the
  environment handle for explicit `Ortex.load(path, environment: env)` calls.
  Set `ORT_DYLIB_PATH` before starting the VM to select the runtime shared library.
  """
  use GenServer
  @enforce_keys [:reference]
  defstruct [:reference]

  @doc "Returns this VM's environment handle."
  def get, do: GenServer.call(__MODULE__, :get)

  @doc false
  def start_link(_opts), do: GenServer.start_link(__MODULE__, [], name: __MODULE__)

  @impl true
  def init(_) do
    opts = Application.get_env(:ortex, :environment, [])
    validate!(opts)
    key = {__MODULE__, :environment}

    case :persistent_term.get(key, nil) do
      nil ->
        pool =
          if opts[:global_thread_pool],
            do:
              Map.merge(
                %{intra_threads: nil, inter_threads: nil, spinning: nil},
                Map.new(opts[:global_thread_pool])
              )

        case Ortex.Native.environment(pool) do
          {:error, message} ->
            {:stop, message}

          reference ->
            environment = %__MODULE__{reference: reference}
            :persistent_term.put(key, {opts, environment})
            {:ok, environment}
        end

      {^opts, environment} ->
        {:ok, environment}

      _ ->
        {:stop, "Ortex environment configuration changed; restart the VM"}
    end
  end

  @impl true
  def handle_call(:get, _from, environment), do: {:reply, environment, environment}

  defp validate!(opts) do
    unless Keyword.keyword?(opts) and Keyword.keys(opts) in [[], [:global_thread_pool]] do
      raise ArgumentError, "expected environment options: global_thread_pool: keyword_list"
    end

    if pool = opts[:global_thread_pool] do
      unless Keyword.keyword?(pool) and
               length(Keyword.keys(pool)) == length(Enum.uniq(Keyword.keys(pool))) and
               Keyword.keys(pool) -- [:intra_threads, :inter_threads, :spinning] == [] do
        raise ArgumentError, "invalid global thread pool options"
      end

      for {key, value} <- pool do
        valid =
          if key == :spinning,
            do: is_boolean(value),
            else: is_integer(value) and value >= 0 and value <= 2_147_483_647

        unless valid,
          do: raise(ArgumentError, "invalid global thread pool option #{inspect(key)}")
      end
    end
  end
end
