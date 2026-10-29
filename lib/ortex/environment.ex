defmodule Ortex.Environment do
  @moduledoc """
  Owns this VM's ONNX Runtime environment. ort permits one environment creation
  per process lifetime, so the handle is retained across application restarts.
  Set `ORT_DYLIB_PATH` before startup; changing the runtime requires a fresh VM.
  """
  use GenServer
  @enforce_keys [:reference]
  defstruct [:reference]
  def get, do: GenServer.call(__MODULE__, :get)
  def start_link(_opts), do: GenServer.start_link(__MODULE__, [], name: __MODULE__)

  @impl true
  def init(_) do
    key = {__MODULE__, :environment}

    case :persistent_term.get(key, nil) do
      nil ->
        case Ortex.Native.environment() do
          {:error, message} ->
            {:stop, message}

          reference ->
            environment = %__MODULE__{reference: reference}
            :persistent_term.put(key, environment)
            {:ok, environment}
        end

      environment ->
        {:ok, environment}
    end
  end

  @impl true
  def handle_call(:get, _from, environment), do: {:reply, environment, environment}
end
