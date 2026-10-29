import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

# CPU-only by default. Accelerator features require an explicit source build and
# a matching ONNX Runtime library; never infer GPU support from the host OS.

# Test-only source-build diagnostic, separate from distributed precompiled NIFs.
if config_env() == :test and System.get_env("ORTEX_BENCHMARK_DIRTY_CPU") == "1" do
  config :ortex, Ortex.Native, features: ["load-dynamic", "api-17", "benchmark-dirty-cpu"]
end
