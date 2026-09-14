# Experimental A2 shared-token dispatch

When a token selects several experts on one destination NPU, the existing A2
single-node low-latency dispatch copies the input vector once per expert. This
prototype sends one payload per source token and destination, followed by small
references for the other experts. The receiver expands the rows using local HBM
reads, preserving the expert GEMM and combine interfaces.

This is an opt-in experiment, disabled in ordinary builds. The matched serving
study below finds a small latency reduction with one replicated placement,
while default placement with the control kernel remains fastest. This does not
establish a general speedup or justify enabling the protocol by default.

## Enable and build

Configure the `csrc/deepep/ops2` CMake project with `-DDEEPEP_A2_DEDUP=ON`
in addition to its usual build options. The kernel option is
`-DDEEPEP_A2_DEDUP=1`; with the CMake option OFF, the original flags are retained.
Set `OPS_PROJECT_NAME=aclnnInner` before configuring when building the private
vendor package used by DeepEP. Build and install into a separate package first.

Every rank in an EP group must load the same protocol build. The new packet
format is incompatible with native receivers. The API and buffer reservations
are unchanged; changing this option is not an online routing choice.

## Packet protocol

1. Find the first active top-k assignment for the same token and destination.
   With an expert mask, scan compact active entries and resolve their original
   indices, so masked assignments cannot own a payload.
2. The owner copies the full existing row. Other assignments copy only its
   44-byte tail, retaining their existing per-expert window positions.
3. The unused 32-byte scale field stores a version marker, the owner's local
   expert slot, and its row ordinal. The original 12-byte source/token/top-k
   triple is retained for every assignment.
4. The existing all-core send barrier precedes per-expert readiness publication.
   Payload and descriptor writes must both complete before receiver reads.
5. The receiver reads the descriptor, resolves the owner's row in its own HCCL
   window, and reads the vector locally. It emits the original expert-grouped
   input row with that assignment's combine triple.

The send loop and per-expert count/readiness messages remain. This does not merge
all assignments into one physical DMA operation. It removes redundant payload
copies. Combine still returns separate expert contributions; local reduction of
those contributions is a separate change.

## Scope and limits

The protocol applies to unquantized single-node A2 routed experts, without shared
expert ranks, internal TP all-gather, or one-dimensional token masks, and with
communication TP size 1. Other configurations retain the original path.
Two-dimensional expert masks are supported. BF16 hidden size 2048 has been
hardware checked; other unquantized types/widths are not hardware validated here.
SGLang model TP=8 is compatible with the internal communication TP size of 1.

The normal/prefill dispatch path is unchanged. Use SGLang `--deepep-mode auto`;
forcing low-latency dispatch for prefill can exceed its decode-sized buffer.

The receiver now has a descriptor-first address dependency and the sender must
find the owner ordinal. These costs and unchanged synchronization/combine work
can offset the payload reduction. Existing HCCL window sizes are retained.

## Correctness checks

CPU-only packet oracle:

```bash
python tests/python/deepep/test_a2_dedup_protocol.py
```

It checks 144 cases and 129,588 assignments across EP=2/4/8 and 1/6/32/128
source tokens: masks, empty sources, reused windows, shuffled completed writes,
payloads/triples, and expert-specific weighted output reconstruction. It models
the protocol and does not prove device synchronization.

Manual hardware check (submit via the local NPU scheduler, using only free cards):

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  tests/python/deepep/check_a2_dedup_hardware.py --tokens 6 --out /tmp/dedup-check
```

Set the chosen package's `PYTHONPATH`, vendor `LD_LIBRARY_PATH` and
`ASCEND_CUSTOM_OPP_PATH` before starting workers, plus `HCCL_BUFFSIZE=1024`,
`MOE_ENABLE_TOPK_NEG_ONE=1`, and `MOE_EXPERT_TOKEN_NUMS_TYPE=1`.
Unset `HCCL_OP_EXPANSION_MODE` for this A2 setup. Optional
`--expected-module-root` verifies the imported package location.
Use a fresh output directory, process group and Buffer for each shape.

When another imported vendor package also exposes `libcust_opapi.so`, preload
the selected DeepEP package's `vendors/hwcomputing/op_api/lib/libcust_opapi.so`
through `LD_PRELOAD` before starting workers. The DeepEP helper resolves that
generic library name; selecting the Python package alone does not ensure the
matching operator API is loaded. The serving comparison applies this equally
to control and shared-copy builds and verifies symbol resolution on every rank
before graph capture.

Hardware validation on four Ascend 910B2 devices passed T=1/6/128, 24 routing
cases and 48 graph replays per rank per shape, checking 220,791 expert rows.
Input rows and triples are exact; weighted affine/identity outputs use BF16
rtol=2%, atol=0.08. Fully masked token outputs are unspecified by native combine
and are ignored, while absence of received rows and zero counts are checked.
The functions validate routing semantics, not real expert GEMMs.

## Measured kernel performance

Matched control and shared-copy packages differed only in dispatch kernel
binaries/metadata. Other package files, including combine, were identical.
EP=4, E=64, top-k=8, hidden=2048 BF16; all assignments remote, with identical
expert/device row counts. Two discarded backend pilots precede 16 measured
process runs, with configuration and backend order reversed for the second
repeat. No FFNs, attention, TP collectives or ILP solve are timed.

| Tokens/source | Destinations/token | Control round trip (us) | Shared-copy (us) | Change |
|---:|---:|---:|---:|---:|
| 6 | 1 | 119.508 | 121.555 | +1.71% |
| 6 | 3 | 123.208 | 122.041 | -0.95% |
| 128 | 1 | 795.977 | 746.044 | -6.27% |
| 128 | 3 | 578.107 | 624.589 | +8.04% |

These are unprofiled amortized dispatch+combine rates: median device-event graph
block duration divided by eight pairs, maximum across rank medians, then median
across two process runs. Negative is faster. The small T=6 spread result changed
sign between repeats (+4.39% and -6.17%); it is not a reliable decode gain.
T=128 is synthetic low-latency traffic, not a prefill measurement.

Fewer destinations alone is not an adequate placement objective: the original
spread case at T=128 is faster than the modified grouped case despite identical
expert loads. Source-destination traffic and synchronization also matter.
Requested dispatch copy lengths fall from 33,120 bytes/token to 4,448 (grouped)
or 12,640 (spread), but these are not measured wire-byte counters.

## TP=EP=8 serving validation

Qwen3-30B-A3B on eight Ascend 910B2 NPUs, SGLang TP=EP=8, DeepEP auto,
graph replay, 144 requests per run, concurrency 48, target 512 input / 128
output tokens, seed 1234. Default is contiguous placement; the two ILP maps
are fixed at startup, with zero or 16 extra physical expert slots per layer.
No runtime solver or online routing coordinator is used. Model startup, map
loading, graph capture and correctness probes are outside the ITL measurement.
No profiler or utilization sampler is enabled.

Two separate backend pilots were discarded. Twelve measured process runs use
forward configuration/backend order followed by reverse order. All generated
request objects match SHA-256
`4d65322a63ecbbe2a69bd7b7d1a040175ab18ec393b96449e88f7822fef8e96a`.
Every process passed Paris and arithmetic probes, all 1,728 measured requests
completed with 128 output tokens each, and all 96 rank startup audits selected
the intended package and unquantized operator API. Each task reserved all eight
cards through the scheduler and finished in 216–240 seconds (300-second cap).

Median ITL is the median within each run, then across two independent process
runs. Negative change means shared-copy is faster.

| Placement | Control ITL (ms) | Shared-copy ITL (ms) | Change | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|---:|---:|
| Default contiguous | 25.022 | 25.079 | +0.23% | -0.03% | +0.48% |
| ILP, no replicas | 25.450 | 25.241 | -0.82% | -0.23% | -1.41% |
| ILP, +16 slots | 26.059 | 25.703 | -1.37% | -1.25% | -1.49% |

The +16 result is a small, consistent reduction in these two repeats: 0.356 ms,
or 1.37%. Output throughput changes only +0.21% for that placement. The
no-replica effect varies more between repeats, and the default effect changes
sign. There are only two process repeats per condition; no confidence interval
or general performance guarantee is claimed.

Default with the control kernel remains fastest. The shared-copy ILP maps are
0.88% (no replicas) and 2.72% (+16) slower than default control. This study tests
the kernel at existing fixed maps; it does not establish an ILP advantage over
default or a production greedy placement. No new placement objective is solved.

Exact generated-text agreement across backends is 103–105 of 144 requests;
within-backend repeats agree on 100–109. Similar agreement rates are compatible
with batching/numerical variation but do not prove tensor-level equivalence or
model quality. The synthetic hardware test above separately checks received
rows and combine results. Output serialization occurs after benchmark timing.

### Environment qualifications

Card 0 reported an inconsistent AI-core counter despite low HBM, zero overall
utilization, no device processes and healthy status. Scheduler-reserved vector
and matrix-multiply correctness diagnostics passed. Each serving arm required
a scheduler reservation and two stable low-HBM polls; only card 0 used the
independent availability indicators when that counter was nonzero. The control
pilot reproduced the historical 25.14 ms baseline at 25.058 ms. These checks do
not prove that every hardware performance characteristic was normal.

All measured runs logged the same forkserver signal-handler exceptions during
startup, then completed graph capture, probes and serving. That startup issue
was not changed or suppressed during measurement. Both backend packages use
the library-selection procedure above; no audit hook runs during timed replay.

The branch's ON dispatch target compiles with CANN 9.0.1; OFF configuration
retains the original compile flags. Its kernel header matches the tested
prototype byte-for-byte. CPU protocol and repository Python formatting checks
pass. The original source checkout and installed serving package are unchanged.
