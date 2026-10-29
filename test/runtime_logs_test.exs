defmodule Ortex.RuntimeLogsTest do
  use ExUnit.Case, async: true

  test "disabled runtime logging has no buffered events" do
    environment = Ortex.Environment.get()
    assert environment.runtime_logging == false
    assert {[], 0} = Ortex.Native.take_runtime_logs(environment.reference)
    assert :ok = Ortex.Environment.flush_logs()
  end

  test "invalid native log levels fail before touching the existing environment" do
    assert {:error, message} = Ortex.Native.environment_with_logs(nil, 99)
    assert message =~ "runtime log level"
    assert %Ortex.Environment{} = Ortex.Environment.get()
  end
end
