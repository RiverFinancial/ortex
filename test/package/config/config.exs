import Config

config :ortex, Ortex.Native,
  features: ["ort/load-dynamic"],
  skip_compilation?: System.fetch_env!("ORTEX_PACKAGE_MODE") == "prebuilt"
