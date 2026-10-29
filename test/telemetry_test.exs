defmodule Ortex.TelemetryTest do
  use ExUnit.Case, async: false

  def handle(event, measurements, metadata, pid), do: send(pid, {event, measurements, metadata})

  setup do
    id = {__MODULE__, make_ref()}

    events =
      for prefix <- [[:ortex, :session, :load], [:ortex, :run]],
          suffix <- [:start, :stop, :exception],
          do: prefix ++ [suffix]

    :ok = :telemetry.attach_many(id, events, &__MODULE__.handle/4, self())
    on_exit(fn -> :telemetry.detach(id) end)
    :ok
  end

  defp fixture(name), do: Path.expand("fixtures/#{name}.onnx", __DIR__)
  defp load, do: Ortex.load(fixture("named_io"), intra_threads: 1)

  test "file load emits source/options at start and model reference at stop" do
    path = fixture("named_io")
    model = Ortex.load(path, providers: [:cpu], optimization_level: 1, intra_threads: 1)
    assert_receive {[:ortex, :session, :load, :start], %{system_time: time}, start}
    assert is_integer(time)
    assert start.source == :file
    assert start.path == path
    assert start.providers == [:cpu]
    assert start.optimization_level == 1
    assert_receive {[:ortex, :session, :load, :stop], %{duration: duration}, stop}
    assert duration >= 0
    assert stop.model == model.reference
    assert stop.path == start.path
  end

  test "memory load reports byte count without model contents" do
    bytes = File.read!(fixture("named_io"))
    model = Ortex.load_from_memory(bytes, intra_threads: 1)
    assert_receive {[:ortex, :session, :load, :start], _, start}
    assert start.source == :memory
    assert start.byte_size == byte_size(bytes)
    refute bytes in Map.values(start)
    assert_receive {[:ortex, :session, :load, :stop], _, %{model: ref}}
    assert ref == model.reference
  end

  test "load failures emit exception and preserve raised error" do
    assert_raise RuntimeError, fn -> Ortex.load(fixture("missing"), intra_threads: 1) end
    assert_receive {[:ortex, :session, :load, :start], _, _}
    assert_receive {[:ortex, :session, :load, :exception], %{duration: d}, metadata}
    assert d >= 0
    assert metadata.kind == :error
    assert %RuntimeError{} = metadata.reason
    assert is_list(metadata.stacktrace)
    refute_receive {[:ortex, :session, :load, :stop], _, _}
  end

  test "named inference emits tensor summaries and selected output tag" do
    model = load()
    inputs = %{"z_left" => Nx.tensor([7.0, 5.0]), "a_right" => Nx.tensor([2.0, 1.0])}
    result = Ortex.run(model, inputs, outputs: ["Δ"], tag: "test-score")
    assert Map.keys(result) == ["Δ"]
    assert_receive {[:ortex, :run, :start], _, start}
    assert start.model == model.reference
    assert start.tag == "test-score"
    assert start.selected_outputs == ["Δ"]

    assert start.inputs == %{
             "z_left" => %{shape: {2}, type: {:f, 32}},
             "a_right" => %{shape: {2}, type: {:f, 32}}
           }

    assert_receive {[:ortex, :run, :stop], %{duration: d}, stop}
    assert d >= 0
    assert stop.outputs == %{"Δ" => %{shape: {2}, type: {:f, 32}}}
    assert stop.inputs == start.inputs
  end

  test "run failure emits exception without a stop and leaves subsequent inference usable" do
    model = load()

    assert_raise RuntimeError, ~r/Unknown input/, fn ->
      Ortex.run(model, %{"wrong" => Nx.tensor([1.0, 2.0]), "a_right" => Nx.tensor([3.0, 4.0])})
    end

    assert_receive {[:ortex, :run, :exception], %{duration: d},
                    %{kind: :error, reason: %RuntimeError{}}}

    assert d >= 0
    refute_receive {[:ortex, :run, :stop], _, _}

    assert {%Nx.Tensor{}, %Nx.Tensor{}} =
             Ortex.run(model, {Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])})
  end

  test "string and sequence summaries contain shape or container size" do
    string = Ortex.load(fixture("metadata_string"), intra_threads: 1)
    Ortex.run(string, Ortex.StringTensor.new(["hello"], {1}))
    assert_receive {[:ortex, :run, :start], _, %{inputs: [%{shape: {1}, type: :string}]}}
    assert_receive {[:ortex, :run, :stop], _, %{outputs: [%{shape: {1}, type: :string}]}}
    sequence = Ortex.load(fixture("sequence_empty"), intra_threads: 1)
    Ortex.run(sequence, {})
    assert_receive {[:ortex, :run, :stop], _, %{outputs: [%{kind: :sequence, size: 0}]}}
  end
end
