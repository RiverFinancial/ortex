# Separate VM; call with ORT_DYLIB_PATH pointing to a nonexistent file.
case Application.ensure_all_started(:ortex) do
  {:error, reason} ->
    message = inspect(reason)
    unless String.contains?(message, "Failed to load ONNX Runtime"), do: raise(message)
    if String.contains?(message, "nif_panicked"), do: raise(message)
    IO.puts("Missing runtime returned a descriptive startup error")

  other ->
    raise("expected missing-runtime failure, got: #{inspect(other)}")
end
