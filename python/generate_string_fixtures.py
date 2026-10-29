"""Synthetic string tensor/scalar and numeric scalar identity graphs."""
from pathlib import Path
import onnx
from onnx import helper, TensorProto

for name, dtype, shape in [
    ("string_matrix", TensorProto.STRING, [None, 2]),
    ("string_scalar", TensorProto.STRING, []),
    ("numeric_scalar", TensorProto.DOUBLE, []),
]:
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], name,
                              [helper.make_tensor_value_info("input", dtype, shape)],
                              [helper.make_tensor_value_info("output", dtype, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    Path(f"test/fixtures/{name}.onnx").write_bytes(model.SerializeToString())
