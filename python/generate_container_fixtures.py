"""Synthetic ZipMap and tensor-sequence graphs, compatible with runtime 1.17."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper


def save(name, nodes, inputs, outputs, ml=False):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    if ml:
        model.opset_import.append(helper.make_opsetid("ai.onnx.ml", 1))
    onnx.checker.check_model(model)
    Path(f"test/fixtures/{name}.onnx").write_bytes(model.SerializeToString())


map_type = helper.make_map_type_proto(TensorProto.STRING, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
save("zipmap_strings",
     [helper.make_node("ZipMap", ["probabilities"], ["mapped"], domain="ai.onnx.ml", classlabels_strings=["safe", "risk π"])],
     [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 2])],
     [helper.make_value_info("mapped", helper.make_sequence_type_proto(map_type))], ml=True)

sequence_type = helper.make_sequence_type_proto(helper.make_tensor_type_proto(TensorProto.FLOAT, [2]))
save("sequence_tensors", [helper.make_node("SequenceConstruct", ["x", "y"], ["sequence"])],
     [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2]) for n in ["x", "y"]],
     [helper.make_value_info("sequence", sequence_type)])
save("sequence_empty", [helper.make_node("SequenceEmpty", [], ["sequence"], dtype=TensorProto.FLOAT)],
     [], [helper.make_value_info("sequence", sequence_type)])
