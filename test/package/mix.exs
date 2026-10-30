defmodule Ortex.PackageCheck.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex_package_check,
      version: "0.1.0",
      elixir: "~> 1.19",
      deps: [{:ortex, path: System.fetch_env!("ORTEX_PACKAGE_PATH")}]
    ]
  end
end
