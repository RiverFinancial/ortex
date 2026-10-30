"""Generate the small model suite and its reference cases with pinned exporters."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "test/fixtures"
MNIST_REVISION = "1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5"
MNIST_SHA = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"


def xgboost():
    import onnxmltools
    import xgboost as xgb
    from onnxmltools.convert.common.data_types import FloatTensorType

    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 3)).astype(np.float32)
    y = (2 * (x[:, 0] > 0) - (x[:, 1] > 0.5) + x[:, 2] * 0.25).astype(np.float32)
    model = xgb.XGBRegressor(
        n_estimators=8, max_depth=3, learning_rate=0.25, n_jobs=1, random_state=42
    )
    model.fit(x, y)
    converted = onnxmltools.convert_xgboost(
        model, initial_types=[("features", FloatTensorType([None, 3]))], target_opset=15
    )
    rows = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.25, 0.75, -0.5]], dtype=np.float32)
    return (
        converted,
        rows,
        [model.predict(rows).reshape(-1, 1)],
        {
            "source": "deterministic synthetic training data; seed 42",
            "license": "MIT",
            "reference": "XGBoost predict, independently checked against Python ORT",
        },
    )


def iris():
    from skl2onnx import to_onnx
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = load_iris()
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=500))
    model.fit(dataset.data, dataset.target)
    rows = dataset.data[[0, 50, 100, 149]].astype(np.float32)
    converted = to_onnx(model, rows, options={id(model[-1]): {"zipmap": False}}, target_opset=15)
    converted.ir_version = 8
    return (
        converted,
        rows,
        [model.predict(rows), model.predict_proba(rows)],
        {
            "source": "scikit-learn load_iris; StandardScaler + LogisticRegression",
            "license": "CC-BY-4.0 (Iris dataset; see NOTICE.md)",
            "reference": "scikit-learn predict and predict_proba",
        },
    )


def mnist():
    url = f"https://huggingface.co/onnxmodelzoo/mnist-12/resolve/{MNIST_REVISION}/mnist-12.onnx"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != MNIST_SHA:
        raise ValueError("MNIST download digest mismatch")
    converted = onnx.load_model_from_string(data)
    # A real trained network with synthetic pixels tests binding/layout/numerics,
    # not classification accuracy or ORT operator correctness.
    rows = np.random.default_rng(42).random((1, 1, 28, 28), dtype=np.float32)
    return (
        converted,
        rows,
        None,
        {
            "source": url,
            "license": "Apache-2.0 (model repository); upstream model documentation also records MIT",
            "reference": "Python ONNX Runtime 1.24.4; synthetic pixels, seed 42",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=["xgboost", "iris", "mnist"])
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_path = DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.models or ["xgboost", "iris", "mnist"]:
        model, inputs, expected, provenance = globals()[name]()
        onnx.checker.check_model(model)
        data = model.SerializeToString()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: inputs})
        if expected is None:
            expected = actual
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        (DIRECTORY / f"{name}.onnx").write_bytes(data)
        cases = {
            "inputs": inputs.tolist(),
            "outputs": [output.tolist() for output in expected],
            "types": [str(output.dtype) for output in actual],
        }
        case_data = (json.dumps(cases, indent=2) + "\n").encode()
        (DIRECTORY / f"{name}.json").write_bytes(case_data)
        manifest[name] = {
            **provenance,
            "sha256": hashlib.sha256(data).hexdigest(),
            "cases_sha256": hashlib.sha256(case_data).hexdigest(),
            "ir": model.ir_version,
            "opsets": {x.domain: x.version for x in model.opset_import},
            "generator": "fixtures/generate.py",
            "requirements": "fixtures/requirements.txt",
            "atol": 1e-5,
        }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
