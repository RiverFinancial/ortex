defmodule Ortex.RunControl do
  @moduledoc """
  Immutable output selection and tag for one cancellable inference attempt.

  Create a control for a specific model and pass it to `Ortex.run/3` with
  `control: control`. Another process can call `cancel/1` while inference is in
  progress or waiting for the session mutex. Cancellation is cooperative: the
  runtime observes a termination flag, and some kernels may finish first.

      control = Ortex.RunControl.new(model, outputs: ["scores"], tag: "risk-score")
      # In the inference process:
      Ortex.run(model, inputs, control: control)
      # From another process:
      Ortex.RunControl.cancel(control)

  Cancellation before execution prevents the attempt from succeeding. A control
  can be used for one inference attempt only, including failed attempts. Repeated
  cancellation is harmless and never cancels later runs using another control.
  A control retains its model and cannot be used with a different model. Selected
  output buffers are not preallocated or shared between runs.

  Controlled calls use a monitored worker and request cancellation if their
  caller exits. Add `timeout: milliseconds` to `Ortex.run/3` to bound caller wait;
  the constructor accepts only output selection and tag options.
  """
  @enforce_keys [:reference]
  defstruct [:reference, :outputs, :tag]

  @doc "Creates a control with `:outputs` and `:tag` options for one model."
  def new(%Ortex.Model{} = model, opts \\ []) do
    options = Ortex.RunOptions.validate!(opts)

    if Keyword.has_key?(opts, :control) or Keyword.has_key?(opts, :timeout),
      do: raise(ArgumentError, "run control options must contain only outputs and tag")

    case Ortex.Native.new_run_control(model.reference, options) do
      {:error, message} -> raise message
      reference -> %__MODULE__{reference: reference, outputs: options.outputs, tag: options.tag}
    end
  end

  @doc "Requests cooperative termination of this control's inference attempt."
  def cancel(%__MODULE__{reference: reference}) do
    case Ortex.Native.cancel_run(reference) do
      {:error, message} -> raise message
      :ok -> :ok
    end
  end

  @doc "Returns whether the single attempt has been claimed, including waiting for the session mutex."
  def started?(%__MODULE__{reference: reference}), do: Ortex.Native.run_started(reference)
end
