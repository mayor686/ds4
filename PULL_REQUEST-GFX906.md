> **AI implementation disclosure:** this gfx906 port, its tests, profiling,
> long-context work, and pull-request preparation were implemented with
> OpenAI Codex under human direction and validated on the hardware below.

## Summary

This change enables and tunes the ROCm backend for gfx906 GPUs: Radeon VII,
Radeon Pro VII, Instinct MI50, and MI60. These devices use fixed 64-lane
wavefronts and cannot use the gfx11 WMMA builtin or the rocWMMA architectures
used by the existing gfx1151 path.

The port provides a wave64-safe compatibility implementation for the exact
16x16 half x half -> float WMMA subset used by ds4, fixes logical-wave32 masks
and broadcasts, and adds native gfx906 paths for Q8 decode, F16-pair decode,
router evaluation, and short-cache attention scoring.

The final revision also removes two distributed-memory bottlenecks:

- each process allocates persistent KV only for its local layer slice;
- ROCm SSD expert streaming uses bounded selected-address caches on worker
  slices as well as on the coordinator.

It also stabilizes live KV reuse across tool-call continuations. Tool schemas
are now rendered deterministically when clients send the same tool set in a
different order or serialize semantically identical JSON object keys in a
different order. Schema property order remains intact because ds4 uses it for
DSML argument ordering, while malformed duplicate names/keys are deliberately
not normalized.

On the tested six-GPU host, resident weights now support at least 750K context
without changing KV precision. With a 2 GiB routed-expert cache per process,
the server reaches the model's native 1,000,000-token context limit.

## Validation system

| Component | Tested configuration |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 3955WX, 16 cores / 32 threads |
| RAM | Approximately 96 GiB |
| GPUs | 6 x gfx906 / 60 CU: 5 x 16 GiB plus 1 x 32 GiB |
| GPU mix | 2 x Radeon VII, 3 x Radeon Pro VII, 1 x 32 GiB Vega 20 |
| Desktop GPU | One 16 GiB Radeon VII remains attached to the GUI |
| Interconnect | PCIe x16; no XGMI links between the six cards |
| Software | ROCm 7.2.4, `--offload-arch=gfx906` |
| Storage | Samsung 990 Pro NVMe |
| Model | DeepSeek-V4-Flash IQ2XXS chat-v2 imatrix, approximately 81 GiB |

The capacity PP6 layout gives the 32 GiB coordinator layers 0:12 plus
embedding/output and gives the five 16 GiB workers six layers each.  The speed
profile uses a balanced 8 + 7x5 split at 32K context.  `run.sh` expresses this
as a configurable device/layer route rather than assuming six GPUs or a fixed
enumeration.  Inter-stage activations use 16 bits.

## Performance and capacity

| Workload | Measured result |
|---|---:|
| Balanced PP6, 2,418-token prefill | 109.28 token/s, up from 72.80 (+50.1%) |
| Balanced PP6, profiled decode | 12.61 token/s, up from 12.52 (+0.7%) |
| Balanced PP6, end-to-end benchmark | 27.21 s, down from 38.33 s (-29.0%) |
| Resident short decode, repeated | 14.31-14.35 token/s; median 14.33 |
| 21,745-token coherent prompt | 70.86 prompt token/s |
| Decode after the 21,745-token prompt | 9.79 token/s |
| Initial fully serialized PP6 setup | approximately 6.8 token/s |
| SSD streaming, 2 GiB cold cache | 2.67 decode token/s |
| gfx906 WMMA compatibility | `max_err=0`, 8,192 values |
| Router wave32 -> native wave64 microbenchmark | 89.4 -> 31.4 us |

The controlled legacy-vs-gfx906 workgroup A/B improved end-to-end decode by
5.0%. Isolated Q8 decode improved 18.5%; F16-pair results were bit-identical
and improved at the real 256/512/1024-row shapes. Removing global kernel
serialization accounts for the larger gain over the original setup.

Balancing the resident PP6 route removes the coordinator prefill bottleneck:
the same 2,418-token prompt improved by 50.1%, while decode remained neutral
within run-to-run variance.  The balanced split is therefore the default speed
example; the conservative capacity split remains available for large contexts.

Long-context allocation and live-session results:

| Mode | Context | Worker VRAM after inference | Coordinator VRAM |
|---|---:|---:|---:|
| Resident | 300,000 | 12.92-13.16 GiB | 29.42 GiB |
| Resident | 393,216 | 13.11-13.31 GiB | 29.78 GiB |
| Resident | 750,000 | 13.80-14.00 GiB | 31.11 GiB |
| SSD selected-address, 2 GiB cache | 1,000,000 | 8.26-8.62 GiB | 14.08 GiB |

Before local-slice KV allocation, 290K opened the HTTP server but failed on
the first worker session with an out-of-memory error. After the change, all
contexts in the table created a distributed session and completed inference.
The 1M result validates capacity and a live request; it is not a claim that a
full one-million-token prefill was timed.

SSD streaming uses the original quantized expert bytes, so it changes capacity
and latency rather than model precision. A deterministic `temperature=0`,
no-reasoning comparison returned the same output (`391`) in resident and SSD
modes for the same arithmetic prompt.

## Changes

- Add `rocm/ds4_rocm_wmma_gfx906.cuh`, selected only for gfx906.
- Make subgroup masks and broadcasts correct for logical wave32 groups inside
  a hardware wave64.
- Exclude the gfx11-only raw WMMA builtin from gfx906 builds.
- Add `make gfx906`, architecture-selectable ROCm targets, and real-device
  WMMA, Q8, F16-pair, router, attention, and long-context regressions.
- Pack two Q8 logical rows per wave64 and tune F16-pair workgroups for 60 CUs.
- Map a router row to one native wave64, retaining existing paths elsewhere.
- Use cooperative score dots only for the measured short-cache range.
- Increase one-token IQ2 gate/up occupancy from 96 to 384 workgroups on gfx906
  while preserving each output row's reduction order.
- Transpose only the selected indexed-attention K rows for coalesced Q·K reads,
  retaining the original V layout and FP32 accumulation order.
- Fuse indexed-attention inverse RoPE into the final attention workgroup; the
  ROCm regression verifies bit-identical output against the separate kernel.
- Make the ROCm weight-arena chunk configurable.
- Allocate raw, compressed-attention, indexer, and compressor-frontier KV only
  for the layer range owned by the local distributed process.
- Use selected-address expert streaming for non-zero worker slices instead of
  mapping their full routed layers.
- Reject distributed ROCm expert caches smaller than one complete expert table
  instead of silently falling back to an unsafe full-layer mapping.
- Make the distributed launcher accept configurable coordinator and worker
  device/layer routes, including GPU counts other than the validated PP6 host.
- Document and provide balanced resident-speed and 1M-context PP6 examples.
- Canonicalize tool-schema JSON object keys and tool-list order so equivalent
  tool continuations retain the live KV prefix instead of cold-prefilling the
  conversation again.
- Preserve real invalidations for schema changes, DSML property-order changes,
  omitted reasoning, and interleaved requests; avoid canonicalizing ambiguous
  duplicate tool names or JSON keys.
- Allow launchers to enable the existing server trace using `TRACE_FILE`
  without changing the default runtime behavior.

## Validation

```text
make gfx906 -j16                                      PASS
ROCR_VISIBLE_DEVICES=2 make rocm-regression \
    ROCM_ARCH=gfx906                                 PASS
  top-k n_comp=32768, n_tokens=32                   PASS
  attention ring reference max_abs=2.60e-7          PASS
  indexed attention 16K: 1.347 ms -> 0.382 ms       PASS
  indexed stable/transpose max_abs=9.31e-10         PASS
  indexed fused-RoPE max_abs=0, RMS=0               PASS
  gfx906 WMMA: 256 tiles / 8192 values, max_err=0   PASS
  Q8 wave64 packing: 4097 rows, 0 mismatches        PASS
  F16-pair sizing: 513 rows, 0 mismatches           PASS
  router wave64: 256/384 experts, 0 mismatches      PASS
./ds4_test --server                                  PASS
ASan + UBSan focused schema/tool-replay tests        PASS
```

Distributed end-to-end checks used `run-speed.sh` and `run-context.sh`.
`run-context.sh` passes the same 2 GiB cache policy to every process. A negative
test with 1 GiB / 151 experts exits cleanly and reports the required minimum of
256 experts / approximately 1.69 GiB for this model.

A final resident test allocated a 300,000-token server context and asked the
loaded DeepSeek-V4-Flash model to implement a single-file terminal task manager.
The API returned HTTP 200 with `finish_reason=stop` after generating 1,322
tokens at 13.5 token/s.  The unmodified generated source passed Python compile
and functional tests for add/list/done/stats/remove, stable IDs, JSON output,
invalid IDs, corrupt databases, invalid priorities, and atomic-save cleanup.

Live tool-continuation regression testing used a 300,000-token resident server
and real SSE assembly. An 18-call `read` response generated 751 DSML tokens;
the continuation changed only the JSON key order of an otherwise identical
schema and reused all 1,417 live-prefix tokens, prefilling only the 239-token
tool-result suffix. A larger test reversed all 30 tools in a 4,392-token
continuation and reused 4,374 tokens, writing only 18 new tokens. Before schema
canonicalization, both reorderings caused a full cache miss.

Negative controls confirmed that actual description/tool changes, DSML
property-order changes, omitted thinking content, and an interleaved unrelated
request still invalidate the live KV. Parsing and canonicalizing 30 tool schemas
measured approximately 0.23 ms per request over 5,000 iterations.

The complete seven-commit series was additionally applied with `git am` to a
fresh clone of `antirez/ds4` at `80ebbc3`; that clone passed `make gfx906 -j16`,
the ROCm long-context regression, and all gfx906 wave64 shim tests.

## Reproduction

```sh
make gfx906 -j16
ROCR_VISIBLE_DEVICES=<gfx906-index> make rocm-regression ROCM_ARCH=gfx906
```

Edit `MODEL_PATH` inside the desired launcher; no external environment setup is
required:

```sh
./run-speed.sh    # resident weights, balanced 32K speed profile
./run-context.sh  # 2 GiB expert cache per GPU, native 1M context
```

The included ROCR device map is specific to the validated 5 x 16 GiB plus
1 x 32 GiB topology and must be adjusted when HSA enumeration differs.

## Known limitations

- The gfx906 WMMA shim prioritizes correctness because gfx906 has no MFMA.
- SSD streaming is a capacity mode: the tested 2 GiB cache is substantially
  slower than resident weights. Larger caches trade VRAM for hit rate.
- A full 1M-token quality/prefill benchmark is prohibitively long on this PP6
  PCIe topology; the test covers allocation, route creation, and live decode.
- The GUI-attached card can thermally throttle during sustained prefills.
- MTP did not fit reliably beside the resident coordinator layout and did not
  improve quality-adjusted throughput in this configuration.
- Live KV is a single mutable session: an unrelated interleaved request still
  replaces it, and thinking-mode continuations must replay required reasoning.
