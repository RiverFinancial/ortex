"""Introspection-only ONNX fixture with nested graphs/functions and initializer forms.
The external initializer intentionally names a missing file: inspection must not open it.
"""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

weights = helper.make_tensor("weights", TensorProto.FLOAT, [3, 2], [1.0] * 6)
bias = helper.make_tensor("unused_scalar", TensorProto.INT64, [], [7])
words = helper.make_tensor("unused_words", TensorProto.STRING, [2], ["hello", "π"])
packed = TensorProto(name="unused_packed", data_type=TensorProto.UINT4, dims=[3], raw_data=b"\x21\x03")
external = TensorProto(name="external", data_type=TensorProto.INT64, dims=[2], data_location=TensorProto.EXTERNAL)
for key, value in [("location", "../../missing.weights"), ("offset", "0"), ("length", "16")]:
    item = external.external_data.add()
    item.key, item.value = key, value
sparse = helper.make_sparse_tensor(
    helper.make_tensor("sparse", TensorProto.FLOAT, [2], [1.0, 2.0]),
    helper.make_tensor("indices", TensorProto.INT64, [2, 2], [0, 1, 3, 3]), [4, 4])
branches = []
for op, name in [("Identity", "then"), ("Neg", "else")]:
    local = [helper.make_tensor("branch_scalar", TensorProto.FLOAT, [], [1.0])] if name == "then" else []
    branches.append(helper.make_graph([helper.make_node(op, ["double_y"], ["result"])], name, [],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 2])], initializer=local))
function = helper.make_function("local.tests", "Double", ["input"], ["output"],
    [helper.make_node("Add", ["input", "input"], ["output"])], [helper.make_opsetid("", 15)])
graph = helper.make_graph([
    helper.make_node("MatMul", ["x", "weights"], ["y"]),
    helper.make_node("Double", ["y"], ["double_y"], domain="local.tests"),
    helper.make_node("If", ["condition"], ["final"], then_branch=branches[0], else_branch=branches[1])],
    "graph_summary", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
                      helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
    [helper.make_tensor_value_info("final", TensorProto.FLOAT, [1, 2])],
    initializer=[weights, bias, words, packed, external], sparse_initializer=[sparse])
model = helper.make_model(graph, functions=[function],
    opset_imports=[helper.make_opsetid("", 15), helper.make_opsetid("local.tests", 1)],
    producer_name="River Ortex tests", producer_version="fixture-1", domain="org.riverfinancial.ortex.tests", model_version=7)
model.ir_version = 13
Path("test/fixtures/graph_summary.onnx").write_bytes(model.SerializeToString())
