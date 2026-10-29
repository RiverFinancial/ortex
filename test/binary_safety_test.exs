defmodule Ortex.BinarySafetyTest do
  use ExUnit.Case, async: true

  test "malformed shapes and binary sizes return errors" do
    assert {:error, "binary size does not match tensor shape and type"} =
             Ortex.Native.from_binary(<<0>>, {2}, {:f, 32})

    assert {:error, "tensor shape overflow"} =
             Ortex.Native.from_binary(<<>>, {18_446_744_073_709_551_615, 2}, {:f, 32})
  end

  test "sub-binaries round trip without alignment assumptions" do
    binary = :binary.copy(<<0>>, 4097)
    sub_binary = :binary.part(binary, 1, 4096)
    reference = Ortex.Native.from_binary(sub_binary, {1024}, {:f, 32})
    assert Ortex.Native.to_binary(reference, 32, 1024) == sub_binary
  end
end
