# Main's ort rc.8 reports a failed dynamic load as a caught NIF panic. The runtime
# migration replaces this legacy behavior with a descriptive loader error.
try do
  Ortex.load(System.fetch_env!("ORTEX_FIXTURE_PATH"), [:cpu], 0)
  raise "missing runtime unexpectedly allowed inference"
rescue
  error in ErlangError ->
    unless error.original == :nif_panicked, do: reraise(error, __STACKTRACE__)
end

IO.puts("Missing runtime was rejected in a separate VM")
