"""Synthetic two-input graph whose declaration order differs from sorted names."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Add", ["z_left", "a_right"], ["sum"]),
     helper.make_node("Sub", ["z_left", "a_right"], ["Δ"])],
    "named_io",
    [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2]) for n in ["z_left", "a_right"]],
    [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2]) for n in ["sum", "Δ"]],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
model.ir_version = 8
onnx.checker.check_model(model)
Path("test/fixtures/named_io.onnx").write_bytes(model.SerializeToString())
