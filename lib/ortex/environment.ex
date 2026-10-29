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

  `:name` sets the environment log name (default `"Ortex"`). `:runtime_path`
  selects a dynamic runtime library ahead of `ORT_DYLIB_PATH`. `:providers`
  configures default provider atoms/option maps, with strict registration.
  Keyword sessions inherit these providers when `:providers` is omitted; an
  explicit list replaces them, including `[]` to request runtime CPU fallback.
  Positional loading keeps its explicit CPU/provider defaults. Without configured
  providers the environment defaults to `[:cpu]`, preserving CPU arena behavior.

  Set `runtime_logging: true` and optionally `runtime_log_level: :info` in the
  environment config to forward runtime logs into Elixir Logger. Levels are
  `:debug`, `:info`, `:warning` (default), `:error`, and `:critical`. A bounded
  native queue is drained every 100 ms. It holds 256 entries; each field is
  truncated after 16 KiB at a UTF-8 boundary. Overflow drops new entries and is
  reported as a warning when drained. Logger metadata includes `:ort_category`,
  `:ort_log_id`, and `:ort_code_location`. Logging is disabled by default.

  The queue is owned by the retained environment and survives application
  restart. Native resource layout V2 requires a fresh VM after upgrading.
  """
  use GenServer
  require Logger
  @enforce_keys [:reference]
  defstruct [:reference, runtime_logging: false]

  @doc "Returns this VM's environment handle."
  def get, do: GenServer.call(__MODULE__, :get)

  @doc "Drains buffered runtime logs into Logger synchronously."
  def flush_logs, do: GenServer.call(__MODULE__, :flush_logs)

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

        logging = Keyword.get(opts, :runtime_logging, false)

        level =
          if logging,
            do:
              Enum.find_index(
                [:debug, :info, :warning, :error, :critical],
                &(&1 == Keyword.get(opts, :runtime_log_level, :warning))
              )

        options = %{
          global_thread_pool: pool,
          log_level: level,
          name: Keyword.get(opts, :name, "Ortex"),
          runtime_path: opts[:runtime_path],
          providers: Keyword.get(opts, :providers, [:cpu])
        }

        case Ortex.Native.environment_with_options(options) do
          {:error, message} ->
            {:stop, message}

          reference ->
            environment = %__MODULE__{reference: reference, runtime_logging: logging}
            :persistent_term.put(key, {opts, environment})
            ready(environment)
        end

      {^opts, environment} ->
        ready(environment)

      _ ->
        {:stop, "Ortex environment configuration changed; restart the VM"}
    end
  end

  @impl true
  def handle_call(:get, _from, environment), do: {:reply, environment, environment}

  def handle_call(:flush_logs, _from, environment) do
    flush(environment)
    {:reply, :ok, environment}
  end

  @impl true
  def handle_info(:flush_runtime_logs, environment) do
    flush(environment)
    schedule(environment)
    {:noreply, environment}
  end

  defp ready(environment) do
    schedule(environment)
    {:ok, environment}
  end

  defp schedule(%{runtime_logging: true}),
    do: Process.send_after(self(), :flush_runtime_logs, 100)

  defp schedule(_), do: :ok

  defp flush(environment) do
    {entries, dropped} = Ortex.Native.take_runtime_logs(environment.reference)

    for {level, category, id, location, message} <- entries do
      Logger.log(level, "ONNX Runtime: " <> message,
        ort_category: category,
        ort_log_id: id,
        ort_code_location: location
      )
    end

    if dropped > 0, do: Logger.warning("ONNX Runtime log buffer dropped #{dropped} messages")
  end

  defp validate!(opts) do
    unless Keyword.keyword?(opts) and
             length(Keyword.keys(opts)) == length(Enum.uniq(Keyword.keys(opts))) and
             Keyword.keys(opts) --
               [
                 :global_thread_pool,
                 :runtime_logging,
                 :runtime_log_level,
                 :name,
                 :runtime_path,
                 :providers
               ] ==
               [] do
      raise ArgumentError, "invalid Ortex environment options"
    end

    unless is_boolean(Keyword.get(opts, :runtime_logging, false)),
      do: raise(ArgumentError, "runtime_logging must be a boolean")

    unless Keyword.get(opts, :runtime_log_level, :warning) in [
             :debug,
             :info,
             :warning,
             :error,
             :critical
           ],
           do: raise(ArgumentError, "invalid runtime_log_level")

    for key <- [:name, :runtime_path], Keyword.has_key?(opts, key) do
      value = opts[key]

      unless is_binary(value) and value != "" and String.valid?(value) and
               not String.contains?(value, <<0>>),
             do: raise(ArgumentError, "invalid environment #{key}")
    end

    if Keyword.has_key?(opts, :providers),
      do: Ortex.Options.validate!(providers: opts[:providers])

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
