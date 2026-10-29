defmodule Ortex.RunControlTest do
  use ExUnit.Case, async: true
  alias Ortex.RunControl

  defp load(name), do: Ortex.load(Path.expand("fixtures/#{name}.onnx", __DIR__), intra_threads: 1)
  defp inputs, do: {Nx.tensor([7.0, 5.0]), Nx.tensor([2.0, 1.0])}
  defp loop_inputs(iterations), do: {Nx.tensor(iterations, type: :s64), Nx.tensor(0, type: :s64)}

  defp start_loop(model, control) do
    on_exit(fn -> RunControl.cancel(control) end)

    task =
      Task.async(fn ->
        try do
          {:ok, Ortex.run(model, loop_inputs(1_000_000_000_000), control: control)}
        rescue
          error -> {:error, error}
        end
      end)

    wait_for_start(control, System.monotonic_time(:millisecond) + 5000)
    task
  end

  defp wait_for_start(control, deadline) do
    if RunControl.started?(control) do
      :ok
    else
      if System.monotonic_time(:millisecond) >= deadline,
        do: flunk("controlled inference did not enter its NIF")

      Process.sleep(1)
      wait_for_start(control, deadline)
    end
  end

  test "cancelling before execution is idempotent and leaves normal inference usable" do
    model = load("named_io")
    control = RunControl.new(model)
    assert :ok = RunControl.cancel(control)
    assert :ok = RunControl.cancel(control)

    assert_raise RuntimeError, ~r/terminate/, fn ->
      Ortex.run(model, inputs(), control: control)
    end

    {sum, _} = Ortex.run(model, inputs())
    assert Nx.to_flat_list(sum) == [9.0, 6.0]
  end

  test "a control selects outputs for one attempt and cannot be reused" do
    model = load("named_io")
    control = RunControl.new(model, outputs: ["Δ"], tag: "controlled")
    {difference} = Ortex.run(model, inputs(), control: control)
    assert Nx.to_flat_list(difference) == [5.0, 4.0]

    assert_raise RuntimeError, ~r/already been used/, fn ->
      Ortex.run(model, inputs(), control: control)
    end

    assert :ok = RunControl.cancel(control)
    {sum, _} = Ortex.run(model, inputs())
    assert Nx.to_flat_list(sum) == [9.0, 6.0]
  end

  test "controls are bound to their model and reject ambiguous options" do
    model = load("named_io")
    other = load("named_io")
    control = RunControl.new(model)

    assert_raise RuntimeError, ~r/another model/, fn ->
      Ortex.run(other, inputs(), control: control)
    end

    assert_raise ArgumentError, fn ->
      Ortex.run(model, inputs(), control: control, tag: "override")
    end

    assert_raise ArgumentError, fn -> RunControl.new(model, control: control) end

    assert_raise RuntimeError, ~r/Unknown output/, fn ->
      RunControl.new(model, outputs: ["unknown"])
    end

    assert {_, _} = Ortex.run(model, inputs(), control: control)
  end

  test "failed native validation consumes the attempt" do
    model = load("named_io")
    control = RunControl.new(model)

    assert_raise RuntimeError, ~r/Expected 2 inputs/, fn ->
      Ortex.run(model, {}, control: control)
    end

    assert_raise RuntimeError, ~r/already been used/, fn ->
      Ortex.run(model, inputs(), control: control)
    end
  end

  test "in-flight loops terminate from concurrent cancellation requests" do
    model = load("cancellable_loop")
    control = RunControl.new(model)
    task = start_loop(model, control)

    1..16
    |> Task.async_stream(fn _ -> RunControl.cancel(control) end, max_concurrency: 8)
    |> Enum.each(fn result -> assert result == {:ok, :ok} end)

    assert {:error, %RuntimeError{message: message}} = Task.await(task, 5000)
    assert message =~ "terminate"
    {output} = Ortex.run(model, loop_inputs(10))
    assert Nx.to_number(output) == 10
  end

  test "a queued run can be cancelled while another run holds the session mutex" do
    model = load("cancellable_loop")
    first_control = RunControl.new(model)
    first = start_loop(model, first_control)
    queued_control = RunControl.new(model)
    queued = start_loop(model, queued_control)
    assert :ok = RunControl.cancel(queued_control)
    assert Process.alive?(first.pid)
    assert :ok = RunControl.cancel(first_control)

    for task <- [first, queued] do
      assert {:error, %RuntimeError{message: message}} = Task.await(task, 5000)
      assert message =~ "terminate"
    end
  end

  test "controlled named runs keep Map/Sequence and string conversion" do
    model = load("metadata_zipmap")
    control = RunControl.new(model, outputs: ["mapped"])

    assert %{"mapped" => [%{0 => 0.25, 1 => 0.75}]} =
             Ortex.run(model, %{"probabilities" => Nx.tensor([[0.25, 0.75]])}, control: control)

    string_model = load("metadata_string")
    strings = Ortex.StringTensor.new(["π"], {1})
    assert {^strings} = Ortex.run(string_model, strings, control: RunControl.new(string_model))
  end

  test "a control survives its creating process and model owner" do
    control = Task.async(fn -> RunControl.new(load("named_io")) end) |> Task.await()
    :erlang.garbage_collect()
    assert :ok = RunControl.cancel(control)
  end
end
