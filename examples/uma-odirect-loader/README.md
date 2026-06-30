# UMA O_DIRECT Loader Examples

These examples show the environment variables used by the experimental UMA
O_DIRECT loader branch.

They intentionally use placeholders such as `/models/model.gguf`,
`10.0.0.2`, and `/path/to/llama.cpp`. Adjust them before use.

Examples:

- `single-node-local.env`: single-node UMA-safe local O_DIRECT read path.
- `rpc-server.env`: remote RPC server with stream extension enabled.
- `rpc-main-fast.env`: main server with RPC tensor streaming and faster local
  O_DIRECT reads.
- `rpc-main-quiet.env`: main server with a shared weighted read budget.

Typical usage:

```bash
set -a
. examples/uma-odirect-loader/rpc-main-fast.env
set +a

./build/bin/llama-server \
  -m "$MODEL" \
  --rpc "$RPC_ENDPOINT" \
  -ngl 999 \
  --split-mode layer \
  --tensor-split "$TENSOR_SPLIT" \
  --uma-loader-safe \
  --uma-loader-buffer-load-order "$UMA_BUFFER_LOAD_ORDER" \
  --uma-loader-buffer-gate "$UMA_BUFFER_GATE" \
  --uma-loader-buffer-min-available-gib "$UMA_BUFFER_MIN_AVAILABLE_GIB" \
  --uma-loader-slice-mib "$UMA_SLICE_MIB" \
  --uma-loader-psi-gate "$UMA_PSI_GATE" \
  --uma-loader-min-available-gib "$UMA_MIN_AVAILABLE_GIB"
```

Always check RAM, swap, and memory PSI before loading large models on UMA
systems.
