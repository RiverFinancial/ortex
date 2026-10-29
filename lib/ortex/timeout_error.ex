defmodule Ortex.TimeoutError do
  @moduledoc "Raised when a bounded inference call exceeds its caller deadline."
  defexception [:timeout]
  @impl true
  def message(%{timeout: timeout}), do: "Ortex inference timed out after #{timeout}ms"
end
