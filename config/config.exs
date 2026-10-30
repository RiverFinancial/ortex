import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

# Do not infer accelerator support from the operating system. Runtime selection
# stays explicit through ORT_DYLIB_PATH; compilation needs no runtime headers.
config :ortex, Ortex.Native, features: ["ort/load-dynamic"]
