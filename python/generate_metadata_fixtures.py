"""Small synthetic ONNX fixtures; regenerate with fixture-requirements.txt."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

destination = Path("test/fixtures")


def save(name, nodes, inputs, outputs, rich=False, ml=False):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8  # supported by the oldest test runtime, 1.17.3
    if ml:
        model.opset_import.append(helper.make_opsetid("ai.onnx.ml", 1))
    if rich:
        model.producer_name = "River Ortex tests"
        model.domain = "org.riverfinancial.ortex.tests"
        model.doc_string = "Synthetic model description"
        model.graph.doc_string = "Synthetic graph description"
        model.model_version = 42
        helper.set_model_props(model, {"feature.order": '["a","b","c"]', "unicode.π": "value π"})
    onnx.checker.check_model(model)
    (destination / (name + ".onnx")).write_bytes(model.SerializeToString())


for name, dtype, shape in [
    ("metadata_rich", TensorProto.FLOAT, ["batch", None, 3]),
    ("metadata_string", TensorProto.STRING, [None]),
    ("metadata_bool", TensorProto.BOOL, []),
]:
    save(name, [helper.make_node("Identity", ["features"], ["prediction"])],
         [helper.make_tensor_value_info("features", dtype, shape)],
         [helper.make_tensor_value_info("prediction", dtype, shape)], rich=name == "metadata_rich")

map_type = helper.make_map_type_proto(TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
save("metadata_zipmap",
     [helper.make_node("ZipMap", ["probabilities"], ["mapped"], domain="ai.onnx.ml", classlabels_int64s=[0, 1])],
     [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 2])],
     [helper.make_value_info("mapped", helper.make_sequence_type_proto(map_type))], ml=True)
