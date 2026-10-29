# ONNX summary schema

`onnx_summary.proto` is a wire-compatible subset of ONNX 1.20.1's Apache-2.0
`onnx-ml.proto`. Field tags and scalar/message wire types match the original.
The original license is retained in `LICENSE-ONNX`. See the source link in the
schema. Tensor payload fields are deliberately absent: Protox retains their
unknown bytes temporarily but does not decode them into numeric lists, and the
summary does not return them. This is not a full semantic ONNX checker.

The committed codec is generated with Protox 2.0.10. Regeneration needs protoc;
ordinary source and precompiled consumers do not. From the repository root:

```sh
MIX_ENV=test mix protox.generate --namespace Ortex.Graph \
  --output-path lib/ortex/graph_proto.ex priv/proto/onnx_summary.proto
mix format lib/ortex/graph_proto.ex
```

The schema/codec are private implementation details. Public graph summaries use
binary model names and operator/domain keys, and copy returned binaries so a
small summary does not retain the source model's tensor data.
