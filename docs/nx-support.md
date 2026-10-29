# Nx storage and Serving support

Keep `Ortex.Backend` as numeric tensor storage and transfer support. Inference
results already use it, and reusing its aligned storage avoids repeated transfer
and input copies. Its shape, slice, and concatenate operations also support
Nx.Serving batching. General arithmetic, Nx.Defn compilation, and use as a default
compute backend remain outside its contract. Transfer values to BinaryBackend or
an appropriate compute backend before those operations.

Keep `Ortex.Serving` for numeric/boolean tensor models with a leading batch axis.
Both positional tuples and binary-keyed named maps work. Synthetic tree-model
tests cover named inline batching and simultaneous callers to a supervised Serving
process, checking independent golden scores. Existing positional tiny-model tests
remain. The optional external ResNet model is excluded when absent.

The adapter validates structured model descriptors when initialized and rejects
strings, rank-zero scalar tensors, Map, Sequence, and Optional values with an
explicit batching error. Direct `Ortex.run/2` supports string tensors, scalars,
ZipMap maps, and tensor sequences; these do not acquire Nx batching semantics just
because they can be returned by direct inference. Optional value conversion is
still unsupported. `Ortex.StringTensor` is separate from Nx numeric tensors.

This preserves the useful, exercised integration without expanding Ortex into a
second tensor computation engine. CI uses the locked Nx dependency; compatibility
with an untested Nx release is not implied by the dependency's allowed range.
