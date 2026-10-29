defmodule Ortex.ManagedRunsTest do
  use ExUnit.Case, async: true
  alias Ortex.RunControl

  defp load(name), do: Ortex.load(Path.expand("fixtures/#{name}.onnx", __DIR__), intra_threads: 1)
  defp loop_inputs(n), do: {Nx.tensor(n, type: :s64), Nx.tensor(0, type: :s64)}

  defp wait_started(control, deadline) do
    unless RunControl.started?(control) do
      if System.monotonic_time(:millisecond) >= deadline, do: flunk("native run was not claimed")
      Process.sleep(1)
      wait_started(control, deadline)
    end
  end

  test "bounded named runs preserve output selection and tensor ownership" do
    result =
      Task.async(fn ->
        model = load("named_io")

        Ortex.run(model, %{"z_left" => Nx.tensor([7.0, 5.0]), "a_right" => Nx.tensor([2.0, 1.0])},
          outputs: ["Δ"],
          tag: "bounded",
          timeout: 5000
        )
      end)
      |> Task.await()

    :erlang.garbage_collect()
    assert Map.keys(result) == ["Δ"]
    assert Nx.to_flat_list(result["Δ"]) == [5.0, 4.0]
  end

  test "a real loop times out and releases the session for subsequent inference" do
    model = load("cancellable_loop")

    error =
      assert_raise Ortex.TimeoutError, fn ->
        Ortex.run(model, loop_inputs(1_000_000_000_000), timeout: 5)
      end

    assert error.timeout == 5
    {result} = Ortex.run(model, loop_inputs(10), timeout: 5000)
    assert Nx.to_number(result) == 10
  end

  test "caller death cancels its native worker and releases the model mutex" do
    model = load("cancellable_loop")
    control = RunControl.new(model)
    on_exit(fn -> RunControl.cancel(control) end)

    {owner, monitor} =
      spawn_monitor(fn -> Ortex.run(model, loop_inputs(1_000_000_000_000), control: control) end)

    wait_started(control, System.monotonic_time(:millisecond) + 5000)
    Process.exit(owner, :kill)
    assert_receive {:DOWN, ^monitor, :process, ^owner, :killed}
    {result} = Ortex.run(model, loop_inputs(10), timeout: 5000)
    assert Nx.to_number(result) == 10
  end

  test "a queued deadline returns while another attempt still owns the session" do
    model = load("cancellable_loop")
    control = RunControl.new(model)
    on_exit(fn -> RunControl.cancel(control) end)

    owner =
      Task.async(fn ->
        try do
          Ortex.run(model, loop_inputs(1_000_000_000_000), control: control)
        rescue
          error -> {:error, error}
        end
      end)

    wait_started(control, System.monotonic_time(:millisecond) + 5000)
    assert_raise Ortex.TimeoutError, fn -> Ortex.run(model, loop_inputs(10), timeout: 5) end
    assert Process.alive?(owner.pid)
    :ok = RunControl.cancel(control)
    assert {:error, %RuntimeError{}} = Task.await(owner, 5000)
    {result} = Ortex.run(model, loop_inputs(10), timeout: 5000)
    assert Nx.to_number(result) == 10
  end

  test "worker exceptions retain their error and inference stack" do
    model = load("named_io")

    try do
      Ortex.run(model, %{"wrong" => Nx.tensor([1.0, 2.0]), "a_right" => Nx.tensor([3.0, 4.0])},
        timeout: 5000
      )

      flunk("expected invalid input")
    rescue
      error in RuntimeError ->
        assert error.message =~ "Unknown input"
        assert Enum.any?(__STACKTRACE__, fn {module, _, _, _} -> module == Ortex.Model end)
    end
  end

  test "rejecting a reused control does not cancel its original in-flight attempt" do
    model = load("cancellable_loop")
    control = RunControl.new(model)
    on_exit(fn -> RunControl.cancel(control) end)

    owner =
      Task.async(fn ->
        try do
          Ortex.run(model, loop_inputs(1_000_000_000_000), control: control)
        rescue
          error -> {:error, error}
        end
      end)

    wait_started(control, System.monotonic_time(:millisecond) + 5000)

    assert_raise RuntimeError, ~r/already been used/, fn ->
      Ortex.run(model, loop_inputs(10), control: control, timeout: 5000)
    end

    reference = owner.ref
    refute_receive {^reference, _}, 20
    assert Process.alive?(owner.pid)
    :ok = RunControl.cancel(control)
    assert {:error, %RuntimeError{}} = Task.await(owner, 5000)
  end

  test "zero and invalid deadlines are explicit and control construction remains separate" do
    model = load("named_io")
    input = {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])}
    control = RunControl.new(model)

    assert_raise Ortex.TimeoutError, fn ->
      Ortex.run(model, input, control: control, timeout: 0)
    end

    refute RunControl.started?(control)
    assert_raise RuntimeError, ~r/terminate/, fn -> Ortex.run(model, input, control: control) end

    for timeout <- [-1, 4_294_967_296, 1.5, nil, :later] do
      assert_raise ArgumentError, fn -> Ortex.run(model, input, timeout: timeout) end
    end

    assert_raise ArgumentError, fn -> RunControl.new(model, timeout: 5) end
    assert {_, _} = Ortex.run(model, input, timeout: :infinity)
  end

  test "timed-out protocol workers terminate and leave no late reply or monitor messages" do
    model = load("named_io")
    caller = self()
    control = RunControl.new(model)

    assert_raise Ortex.TimeoutError, fn ->
      Ortex.Runner.run(control, 50, fn ->
        send(caller, {:worker, self()})
        Process.sleep(:infinity)
      end)
    end

    assert_receive {:worker, worker}
    monitor = Process.monitor(worker)
    assert_receive {:DOWN, ^monitor, :process, ^worker, _}, 5000
    assert_no_late_replies()
  end

  test "successful and deadline-racing replies do not accumulate in the caller mailbox" do
    model = load("named_io")

    for _ <- 1..25 do
      control = RunControl.new(model)

      try do
        assert :completed =
                 Ortex.Runner.run(control, 1, fn ->
                   Process.sleep(1)
                   :completed
                 end)
      rescue
        Ortex.TimeoutError -> :ok
      end
    end

    assert_no_late_replies()
  end

  defp assert_no_late_replies do
    receive do
      {reference, _} = message when is_reference(reference) ->
        flunk("late reply: #{inspect(message)}")

      {:DOWN, _, :process, _, _} = message ->
        flunk("late monitor: #{inspect(message)}")
    after
      20 -> :ok
    end
  end
end
