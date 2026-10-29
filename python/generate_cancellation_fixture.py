"""A scalar counting Loop supports deterministic cooperative cancellation tests."""
from pathlib import Path
import onnx
from onnx import helper, TensorProto

i64 = lambda name: helper.make_tensor_value_info(name, TensorProto.INT64, [])
boolean = lambda name: helper.make_tensor_value_info(name, TensorProto.BOOL, [])
body = helper.make_graph([
    helper.make_node("Identity", ["condition_in"], ["condition_out"]),
    helper.make_node("Add", ["state_in", "one"], ["state_out"])], "counting_body",
    [i64("iteration"), boolean("condition_in"), i64("state_in")],
    [boolean("condition_out"), i64("state_out")],
    initializer=[helper.make_tensor("one", TensorProto.INT64, [], [1])])
graph = helper.make_graph([helper.make_node("Loop", ["iterations", "condition", "initial"], ["result"], body=body)],
    "cancellable_loop", [i64("iterations"), i64("initial")], [i64("result")],
    initializer=[helper.make_tensor("condition", TensorProto.BOOL, [], [True])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
model.ir_version = 8
onnx.checker.check_model(model)
Path("test/fixtures/cancellable_loop.onnx").write_bytes(model.SerializeToString())
