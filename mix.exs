defmodule Ortex.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex,
      version: "0.1.10",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      test_ignore_filters: [~r{test/(system|package|fixtures|support)/}],
      test_coverage: [ignore_modules: [Ortex.Native], summary: [threshold: 0], export: "library"],
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: "https://github.com/RiverFinancial/ortex",
      homepage_url: "https://github.com/RiverFinancial/ortex",
      docs: [
        main: "readme",
        extras: ["README.md"]
      ],
      package: package()
    ]
  end

  def cli do
    [preferred_envs: [docs: :docs]]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.29.1"},
      {:nx, "~> 0.6"},
      {:credo, "~> 1.7.19", only: :test, runtime: false},
      {:ex_doc, "~> 0.40.4", only: :docs, runtime: false}
    ]
  end

  defp package do
    [
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* native/ortex/src/ config/config.exs
        native/ortex/Cargo.lock native/ortex/Cargo.toml native/ortex/.cargo/config.toml),
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/RiverFinancial/ortex"},
      description: "ONNX Runtime bindings for Elixir"
    ]
  end
end
