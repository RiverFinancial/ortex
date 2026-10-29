defmodule Ortex.Runner do
  @moduledoc false

  # The caller owns a reply alias; late responses are dropped when it expires.
  # A separate watcher owns the native worker and cancels it on caller death.
  def run(control, timeout, fun) do
    deadline = if timeout == :infinity, do: :infinity, else: now() + timeout
    if timeout == 0, do: timed_out(control, timeout)
    owner = self()
    reply = :erlang.alias()
    {watcher, monitor} = spawn_monitor(fn -> watch(owner, reply, control, fun) end)

    try do
      receive do
        {^reply, outcome} ->
          if expired?(deadline), do: abort(watcher, reply, control, timeout)

          case outcome do
            {:return, value} -> value
            {:raise, kind, reason, stack} -> :erlang.raise(kind, reason, stack)
          end

        {:DOWN, ^monitor, :process, ^watcher, reason} ->
          Ortex.RunControl.cancel(control)
          exit({:ortex_run_watcher, reason})
      after
        remaining(deadline) -> abort(watcher, reply, control, timeout)
      end
    after
      :erlang.unalias(reply)
      Process.demonitor(monitor, [:flush])

      receive do
        {^reply, _} -> :ok
      after
        0 -> :ok
      end
    end
  end

  defp watch(owner, reply, control, fun) do
    owner_monitor = Process.monitor(owner)
    watcher = self()

    {worker, worker_monitor} =
      spawn_monitor(fn ->
        outcome =
          try do
            {:return, fun.()}
          catch
            kind, reason -> {:raise, kind, reason, __STACKTRACE__}
          end

        send(watcher, {reply, self(), outcome})
      end)

    try do
      action =
        receive do
          {^reply, ^worker, outcome} ->
            send(reply, {reply, outcome})
            :complete

          {:DOWN, ^owner_monitor, :process, ^owner, _reason} ->
            :cancel

          {:DOWN, ^worker_monitor, :process, ^worker, reason} ->
            send(reply, {reply, {:raise, :exit, {:ortex_run_worker, reason}, []}})
            :cancel

          {:abort, ^reply} ->
            :cancel
        end

      if action == :cancel, do: cancel_worker(control, worker)
    catch
      kind, reason ->
        stack = __STACKTRACE__
        cancel_worker(control, worker)
        :erlang.raise(kind, reason, stack)
    after
      Process.demonitor(owner_monitor, [:flush])
      Process.demonitor(worker_monitor, [:flush])
    end
  end

  defp cancel_worker(control, worker) do
    # An executing NIF retains native resources even after its BEAM worker dies.
    try do
      Ortex.RunControl.cancel(control)
    after
      Process.exit(worker, :kill)
    end
  end

  defp abort(watcher, reply, control, timeout) do
    send(watcher, {:abort, reply})
    timed_out(control, timeout)
  end

  defp timed_out(control, timeout) do
    Ortex.RunControl.cancel(control)
    raise Ortex.TimeoutError, timeout: timeout
  end

  defp now, do: System.monotonic_time(:millisecond)
  defp expired?(:infinity), do: false
  defp expired?(deadline), do: now() >= deadline
  defp remaining(:infinity), do: :infinity
  defp remaining(deadline), do: max(deadline - now(), 0)
end
