defmodule Ortex.Examples.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex_examples,
      version: "0.1.0",
      elixir: "~> 1.19",
      deps: [
        {:ortex, path: ".."},
        {:tokenizers, "~> 0.4"},
        {:jason, "~> 1.4"}
      ]
    ]
  end
end
