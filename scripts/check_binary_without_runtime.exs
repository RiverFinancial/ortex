# Run in a fresh VM with an invalid ORT_DYLIB_PATH and --no-start.
{:error, "binary size does not match tensor shape and type"} =
  Ortex.Native.from_binary(<<0>>, {2}, {:f, 32})

{:error, "tensor shape overflow"} =
  Ortex.Native.from_binary(<<>>, {18_446_744_073_709_551_615, 2}, {:f, 32})

{:error, "unsupported tensor type"} = Ortex.Native.from_binary(<<>>, {0}, {:f, 7})
binary = <<1.0::float-native-32>>
reference = Ortex.Native.from_binary(binary, {1}, {:f, 32})
^binary = Ortex.Native.to_binary(reference, 32, 1)
IO.puts("Tensor validation and conversion work without loading ONNX Runtime")
