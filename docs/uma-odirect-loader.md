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

This problem is easy to underestimate because the usual `free` output separates
`used` from `buff/cache`. On Linux, file-backed page cache is often reclaimable,
so `MemAvailable` is usually the right high-level signal. On memory-tight UMA
systems, however, a large GGUF load can create a short window where page cache,
backend tensor allocations, and staging buffers all overlap in the same physical
RAM pool. If you only look at `used`, you can miss the real pressure.

For load testing, record at least:

- peak `used`
- peak `buff/cache`
- peak `used + buff/cache`
- minimum `available`
- swap usage
- memory PSI and IO PSI

If memory PSI rises during model load, treat that run as unsafe even when
`MemAvailable` still looks large. PSI means the kernel is already stalling tasks
on reclaim or memory pressure. On a desktop UMA machine this can feel like a
temporary freeze; at larger sizes it can become unrecoverable without a hard
reset.

## What To Try Before This Branch

If you want to stay within upstream llama.cpp behavior first, the practical
mitigations are:

- reduce `--ctx-size`, `--parallel`, batch/ubatch, and KV cache size
- disable warmup for load tests with `--no-warmup`
- avoid `--mlock` for memory-tight UMA model loads
- avoid RPC local cache (`ggml-rpc-server -c`) unless you explicitly want the
  RPC node to keep file-cache residency
- compare `mmap`, `--no-mmap`, and `--direct-io` with RAM/PSI monitoring
- drop page cache before controlled benchmarks only when you understand the
  system-wide impact
- stop other model servers before loading a large model

These are still worth testing. For some 20-30 GiB GGUF loads, standard mmap can
be fast and safe if there is enough headroom. The issue appears when the hidden
page-cache-inclusive peak approaches the UMA memory limit, or when a multi-node
load overlaps local reads, RPC transfer, and accelerator allocations.

## Measured Examples

The following measurements are from a DGX Spark-style UMA setup and are intended
as shape-of-behavior examples, not universal performance claims. Always validate
on your own filesystem, kernel, backend, and model.

### Single-node 35B-class GGUF

Model size was about 22 GiB. The test used `llama-server`, `ctx 8192`,
`parallel 1`, `cache-ram 0`, full GPU offload, and no warmup. The file cache was
advised away before each run.

| loader path | model loaded | peak used | peak buff/cache | peak used + buff/cache | min available | memory PSI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard mmap | `0:12.727` | `30.28 GiB` | `23.50 GiB` | `53.78 GiB` | `89.35 GiB` | `0.00/0.00` |
| standard `--no-mmap` | `0:19.742` | `31.22 GiB` | `24.07 GiB` | `55.29 GiB` | `88.41 GiB` | `0.00/0.00` |
| UMA local O_DIRECT | `0:11.305` | `31.21 GiB` | `2.94 GiB` | `34.14 GiB` | `88.42 GiB` | `0.00/0.00` |

The standard mmap run was already safe in this case if judged by `available`
and memory PSI. But the page-cache-inclusive peak was about 20 GiB higher than
the UMA local O_DIRECT path. This is the class of difference that matters when
you are about to load another model or increase context on a machine with less
headroom.

### Two-node 200B-class GGUF over RPC

A much larger 200B-class sharded GGUF load was tested with two UMA nodes and
RPC offload. The model was Q4-quantized and the loaded API metadata reported
about 199B parameters with about 117 GiB of GGUF payload. Early
standard-leaning loader paths were in the roughly `5:51` to `6:02` range and
produced severe local memory PSI in at least one cache-disabled trial. An early
safer path with RPC cache disabled but before the O_DIRECT streaming work
loaded in about `3:12`.

The current public UMA O_DIRECT path, using in-process local O_DIRECT reads,
RPC tensor streaming, RPC cache disabled, `remote-first` buffer loading, and a
local read limit of `3000 MiB/s`, loaded the same class of model in `0:51.612`
with swap at `0` and memory PSI at `0.00/0.00` on both nodes.

That result is not just a speedup. The important change is that the loader does
not silently return to mmap/page-cache/full-buffer fallback paths after the user
has explicitly requested the UMA-safe path.

## Primary Mode

Build with RPC enabled when you want multi-node loading:

```bash
cmake -S . -B build-uma-rpc -DGGML_RPC=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-uma-rpc --target llama-server ggml-rpc-server -j
```

For single-node testing, RPC is not required:

```bash
cmake -S . -B build-uma-local -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-uma-local --target llama-server -j
```

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
./build-uma-local/bin/llama-server \
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
./build-uma-rpc/bin/ggml-rpc-server -H 0.0.0.0 -p 50052
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
./build-uma-rpc/bin/llama-server \
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
pgrep -af 'llama-server|ggml-rpc-server|vllm|sglang|python.*api_server' || true
```

Treat any nonzero memory PSI during load as a failed load, even if
`MemAvailable` still looks high. On UMA systems, memory PSI usually means the
loader is forcing reclaim/stall in the same RAM pool needed by the model.

Do not use:

- `--mlock` for very large mmap-backed loads on memory-tight UMA systems
- RPC local cache (`ggml-rpc-server -c`) unless you intentionally want file-cache
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
