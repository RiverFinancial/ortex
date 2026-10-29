defmodule Ortex.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    Supervisor.start_link([Ortex.Environment], strategy: :one_for_one, name: Ortex.Supervisor)
  end
end
