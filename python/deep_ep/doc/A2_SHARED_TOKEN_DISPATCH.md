# Experimental A2 shared-token dispatch

Enable with `-DDEEPEP_A2_DEDUP=ON` when configuring `csrc/deepep/ops2`.
The default is OFF. Every EP rank must use the same build: the packet format
is incompatible with native receivers. API signatures and window sizes stay
unchanged.

## Protocol

For each source token and destination rank, the first active top-k assignment
sends the full input vector. Other assignments send a 44-byte descriptor and
their own source/token/top-k triple at the existing per-expert window positions.
The unused 32-byte scale area stores a version marker and the owner's local
expert slot and row ordinal. Masked assignments cannot own a payload.

The receiver resolves the descriptor in its local HCCL window and emits the
original expert-grouped row. Existing send barriers precede readiness flags,
including payloads written by another sender core. Expert counts, compute
interfaces and combine are unchanged. The send loop and per-expert readiness
messages remain; this removes redundant payload copies, not all communication.

## Supported scope

Unquantized single-node A2, communication TP size 1, no shared-expert ranks,
no internal TP all-gather, and no one-dimensional token mask. Two-dimensional
expert masks are supported. Other modes retain the original path. Hardware
validation covers BF16 hidden size 2048. SGLang model TP=8 uses communication
TP=1 here. Normal/prefill dispatch is unchanged; use DeepEP auto for serving.

## Validation

The hardware check validates exact received rows/triples, expert counts,
weighted combine outputs, masks, empty sources, reused windows and graph replay:

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  tests/python/deepep/check_a2_dedup_hardware.py --tokens 6 --out /tmp/dedup-check
```

Use a fresh process group, Buffer and output directory per shape. Select the
package before startup through `PYTHONPATH`, `LD_LIBRARY_PATH` and
`ASCEND_CUSTOM_OPP_PATH`. If vendor API library names collide, preload the chosen
DeepEP `vendors/hwcomputing/op_api/lib/libcust_opapi.so`. Set `HCCL_BUFFSIZE=1024`,
`MOE_ENABLE_TOPK_NEG_ONE=1`, `MOE_EXPERT_TOKEN_NUMS_TYPE=1`; unset
`HCCL_OP_EXPANSION_MODE`. Run only on reserved devices through the local scheduler.

Four-device checks passed T=1/6/128, 24 cases and 48 graph replays per rank and
shape. Combine tolerance is BF16 rtol=2%, atol=0.08; fully masked output rows
are unspecified by native combine and ignored. The ON dispatch target builds
with CANN 9.0.1; OFF retains the original compiler flags.

## Performance limits

Qwen3-30B-A3B, TP=EP=8, DeepEP auto and graph replay; identical 144 requests/run,
concurrency 48, 512 input / 128 output tokens, seed 1234. Two pilots discarded,
two process repeats with reversed order per condition. Placements are static;
startup and solving are outside ITL. All 1,728 requests and probes passed.

| Placement | Control ITL (ms) | Shared-copy (ms) | Change |
|---|---:|---:|---:|
| Default contiguous | 25.022 | 25.079 | +0.23% |
| ILP, no replicas | 25.450 | 25.241 | -0.82% |
| ILP, +16 slots | 26.059 | 25.703 | -1.37% |

Default control remains fastest. The +16 effect repeated at -1.25%/-1.49%,
with only +0.21% output throughput. Two repeats do not establish a general gain.
Exact text agreement across kernels (103–105/144) resembles within-backend
variation (100–109/144), but does not prove tensor-level equivalence. An
inconsistent card-0 utilization counter and common startup forkserver exceptions
remain environment limitations; independent idle checks and probes passed.

Synthetic EP=4 dispatch+combine tests also show a regression: at T=128/source,
shared-copy changes grouped traffic by -6.27% but spread traffic by +8.04%.
Decode-sized T=6 shows no reliable benefit. Descriptor reads, owner lookup,
local expansion, synchronization and unchanged combine can outweigh saved bytes.
