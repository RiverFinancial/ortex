import ExUnit.Assertions

# A separately supplied runtime must load before the first inference in this VM.
path = System.fetch_env!("ORT_DYLIB_PATH")
assert File.regular?(path)
model = Ortex.load(Path.expand("../../models/tinymodel.onnx", __DIR__), [:cpu])
{inputs, outputs} = Ortex.Native.show_session(model.reference)
assert length(inputs) == 2
assert length(outputs) == 3
