Code.require_file(Path.expand("../support/scenario.exs", __DIR__))
model = Ortex.load(Ortex.Scenario.fixture(), [:cpu])
before = Ortex.Scenario.score(model)
:ok = Application.stop(:ortex)
{:ok, _} = Application.ensure_all_started(:ortex)

unless Ortex.Scenario.score(model) == before,
  do: raise("model changed across application restart")
