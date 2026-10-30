# Model fixture notices

The XGBoost fixture is trained on synthetic data using fixtures/generate.py.
Its model and reference inputs are provided under this repository's MIT license.

The Iris fixture adapts the dataset distributed with scikit-learn into a trained
StandardScaler / LogisticRegression pipeline. Attribution: Fisher, R. (1936).
Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
The UCI dataset is licensed under Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/. Training, export, and selection of
reference rows are the modifications made here.

The MNIST model is copied unchanged from onnxmodelzoo/mnist-12 at the revision
recorded in manifest.json. That model repository declares Apache-2.0; its license
text is in LICENSE-MNIST. The original ONNX Model Zoo MNIST documentation also
records MIT. We retain the Apache notice for the exact artifact source used here.
Reference pixels are synthetic; no MNIST image dataset is redistributed.

These fixtures and notices are test inputs, excluded from the Hex package.
