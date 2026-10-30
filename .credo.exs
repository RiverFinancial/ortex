%{
  configs: [
    %{
      name: "default",
      files: %{included: ["lib/", "test/"], excluded: [~r"/fixtures/", ~r"/package/"]},
      checks: %{
        enabled: [
          {Credo.Check.Warning.IExPry, []},
          {Credo.Check.Warning.IoInspect, []},
          {Credo.Check.Warning.UnusedEnumOperation, []},
          {Credo.Check.Consistency.TabsOrSpaces, []},
          {Credo.Check.Readability.TrailingBlankLine, []}
        ]
      }
    }
  ]
}
