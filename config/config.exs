import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

# CPU-only by default. Accelerator features require an explicit source build and
# a matching ONNX Runtime library; never infer GPU support from the host OS.
