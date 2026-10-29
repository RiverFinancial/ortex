"""A harmless unused initializer causes a deterministic ONNX Runtime warning."""
from pathlib import Path
import onnx
from onnx import helper, TensorProto

model = onnx.load("test/fixtures/named_io.onnx")
model.graph.initializer.append(helper.make_tensor("unused_logging_fixture", TensorProto.FLOAT, [1], [0.0]))
onnx.checker.check_model(model)
Path("test/fixtures/logging_warning.onnx").write_bytes(model.SerializeToString())
