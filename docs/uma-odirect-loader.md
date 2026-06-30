# UMA O_DIRECT Loader

This branch adds an experimental model loader mode for UMA systems where file
cache, CPU staging buffers, and accelerator-visible memory share the same
physical RAM pool.

The goal is not to replace the default loader on ordinary dGPU systems. The
goal is to provide a fail-closed loading path for systems where the default
Linux page-cache and mmap behavior can cause memory pressure stalls while large
GGUF models are being loaded.

## Why This Exists

On typical dGPU systems, system RAM and VRAM are separate pools. During model
loading, page cache and CPU-side staging usually compete with system RAM, while
final weights live in VRAM.

On UMA systems, those pools are effectively the same RAM. A large model load can
temporarily overlap:

- file-backed page cache or mmap pages
- final backend tensor buffers
- RPC transfer buffers
- repack or upload staging buffers
- KV / prompt-cache reservations

This branch adds controls to reduce full-buffer staging and to fail rather than
silently falling back to mmap or ordinary buffered reads when an explicit UMA
loader path was requested.

## Primary Mode

Enable safe mode with:

```bash
--uma-loader-safe
```

Safe mode implies `--no-mmap` and enables stricter loader behavior. If an
explicit O_DIRECT/RPC streaming path cannot be used, the loader should fail
closed instead of silently falling back to a page-cache-heavy path.

Useful loader arguments:

```text
--uma-loader-safe
--uma-loader-buffer-load-order default|round-robin|remote-first|parallel
--uma-loader-slice-mib N
--uma-loader-psi-gate N
--uma-loader-min-available-gib N
--uma-loader-buffer-gate N
--uma-loader-buffer-min-available-gib N
--uma-loader-upload-chunk-mib N
```

## In-Process Local O_DIRECT Reads

For a single-node local load, use local O_DIRECT reads without an external
streamer:

```bash
GGML_LOCAL_ODIRECT_STREAM_ENDPOINT=local \
GGML_LOCAL_ODIRECT_STREAM_MODE=local \
./build/bin/llama-server \
  -m /models/model.gguf \
  -ngl 999 \
  --uma-loader-safe \
  --uma-loader-slice-mib 1024 \
  --uma-loader-psi-gate 30 \
  --uma-loader-min-available-gib 20
```

The `ENDPOINT=local` value is just a non-empty marker for in-process local
reads when `GGML_LOCAL_ODIRECT_STREAM_MODE=local` is selected.

Optional local read limiter:

```bash
GGML_LOCAL_ODIRECT_READ_RATE_LIMIT_MIBPS=3000
```

## RPC Loading

This branch adds an RPC tensor streaming command with protocol/capability
gating. New clients must not send the stream command to servers that do not
advertise support.

For the RPC server, enable the O_DIRECT stream extension explicitly:

```bash
GGML_RPC_ODIRECT_STREAM_ENABLE=1 \
GGML_RPC_ODIRECT_STREAM_ALLOW_ENDPOINT=127.0.0.1:50152 \
GGML_RPC_ODIRECT_STREAM_PATH_PREFIX=/models/ \
./build/bin/rpc-server -H 0.0.0.0 -p 50052
```

`GGML_RPC_ODIRECT_STREAM_ALLOW_ENDPOINT` and
`GGML_RPC_ODIRECT_STREAM_PATH_PREFIX` are required for the server-side
`SET_TENSOR_FROM_FILE` stream path. Keep them narrow. The RPC server is still an
unauthenticated server and should only be exposed on a trusted private network.

For a main server that reads locally and streams the tensor payload through RPC,
use:

```bash
GGML_RPC_ODIRECT_STREAM_ENDPOINT=local \
GGML_RPC_ODIRECT_STREAM_MODE=local \
GGML_LOCAL_ODIRECT_STREAM_ENDPOINT=local \
GGML_LOCAL_ODIRECT_STREAM_MODE=local \
./build/bin/llama-server \
  -m /models/model.gguf \
  --rpc 10.0.0.2:50052 \
  -ngl 999 \
  --split-mode layer \
  --tensor-split 1,1 \
  --uma-loader-safe \
  --uma-loader-buffer-load-order remote-first \
  --uma-loader-buffer-gate 30 \
  --uma-loader-buffer-min-available-gib 20 \
  --uma-loader-slice-mib 1024 \
  --uma-loader-psi-gate 30 \
  --uma-loader-min-available-gib 20
```

## Read Scheduling

The loader can rate-limit local and RPC O_DIRECT reads.

Independent per-path limits:

```bash
GGML_ODIRECT_READ_SCHEDULER=none
GGML_RPC_ODIRECT_READ_RATE_LIMIT_MIBPS=0
GGML_LOCAL_ODIRECT_READ_RATE_LIMIT_MIBPS=3000
```

Weighted shared budget:

```bash
GGML_ODIRECT_READ_SCHEDULER=weighted
GGML_ODIRECT_READ_RATE_LIMIT_MIBPS=3600
GGML_RPC_ODIRECT_READ_WEIGHT=1
GGML_LOCAL_ODIRECT_READ_WEIGHT=1
```

Use `weighted` when load-time responsiveness matters more than raw load time.
Use `none` with explicit local/RPC limits when you want to tune the two paths
separately.

## External O_DIRECT Streamer

The server can optionally spawn external O_DIRECT streamer child processes
during model load:

```bash
GGML_ODIRECT_STREAM_SPAWN=1
GGML_ODIRECT_STREAM_BIN=/path/to/odirect-stream
GGML_ODIRECT_STREAM_HOST=0.0.0.0
GGML_RPC_ODIRECT_STREAM_ENDPOINT=127.0.0.1:50152
GGML_RPC_ODIRECT_STREAM_MODE=stream
```

External streamer mode is mainly useful for experiments. Prefer in-process
`local` mode when possible, because it has fewer moving parts and avoids
leaving helper processes behind.

## Operational Checks

Before loading a very large model on UMA:

```bash
free -h
cat /proc/pressure/memory
cat /proc/pressure/io
pgrep -af 'llama-server|rpc-server|vllm|sglang|python.*api_server' || true
```

Treat any nonzero memory PSI during load as a failed load, even if
`MemAvailable` still looks high. On UMA systems, memory PSI usually means the
loader is forcing reclaim/stall in the same RAM pool needed by the model.

Do not use:

- `--mlock` for very large mmap-backed loads on memory-tight UMA systems
- RPC local cache (`rpc-server -c`) unless you intentionally want file-cache
  residency on the RPC node
- fail-open gate settings for production loads

## Current Limitations

- Linux is the primary target for O_DIRECT and PSI-gated behavior.
- The public C struct currently carries additional UMA loader fields; treat this
  branch as ABI-incompatible with upstream unless all consumers are rebuilt.
- The O_DIRECT paths are conservative and fail-closed by design. If they fail,
  fix the configuration rather than expecting automatic fallback to the default
  loader.
- This is an experimental branch. Validate with your own model, filesystem,
  backend, and RAM pressure before relying on it for unattended loads.
