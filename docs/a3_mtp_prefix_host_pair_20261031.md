# A3 MTP3 operator and shared-prefix framework pair

## Source pair

This branch preserves the tested Q4 LIM partial-chunk fix at `6d28bd6`, including
its regression test and validation record. See
[the Q4 validation record](lim_q4_partial_chunk_validation_20261029.md).

Use it with
[JinyanYi/sglang:feat/mtp3-prefix-kv-fused-metadata-20261031](https://github.com/JinyanYi/sglang/tree/feat/mtp3-prefix-kv-fused-metadata-20261031).
The corresponding framework validation is in
[test/manual/npu/MTP_PREFIX_HOST_VALIDATION.md](https://github.com/JinyanYi/sglang/blob/feat/mtp3-prefix-kv-fused-metadata-20261031/test/manual/npu/MTP_PREFIX_HOST_VALIDATION.md).

## Responsibilities

The Q4 operator fix corrects partial 512-element eviction masks and duplicate
slot-zero victim ownership. It does not implement radix-prefix host-payload
lifetime or framework pre-metadata selection.

The paired framework stores host payloads at native physical token/page IDs,
mirrors accepted-token relocation and adapts pre/post metadata to Q4 shared
prefixes. Both metadata fusions are implemented in the framework. No further
C++ operator change was made for the 2026-10-31 integration run.

For long Q4 with TopK 2048, use C in [8192, 32640], aligned to 128 tokens. This is
an interval, not a whitelist. Integration cases sampled 11392, 12288, 16384 and
16768; they do not exhaustively establish every capacity.

## Validation scope

The paired framework passed 20 CPU regressions and 14 selected real NPU cases on
A3/CANN 9.1, including Q1/Q4 eager/graph metadata, padding changes, rewrite and
rollback, registered-host overlap moves and mathematical attention references.
Full-model TP16/DP1 W4A8/BF16-KV NEXTN 3/1/4 shared-prefix checks returned the
expected distant answer for 7 cold/hot requests, followed by 17 cold/hot16
responses. The auxiliary hot16 client exit code was not retained.

Separate N16/C16 timing and profiling replays, each 65536 input / 1024 output,
completed 16/16 requests with client exit 0. The target trace contains 21 LIM and
78 CopySFA calls per complete decode cycle, with 21 pre-metadata and 78
post-metadata calls. NPU test scope and model scope remain distinct.

The clean HiSparse service result was 189.44 output tokens/s and 56.88 ms mean
TPOT, compared with 226.28 tokens/s and 22.77 ms for a native N6/C6 control with
appropriate target bucket 8/draft 16. This run did not establish an end-to-end
throughput gain or broad model accuracy. Prefix reuse was 65408/65536 tokens;
the selected-KV C-pool hit rate was not measured.

The existing Q4 tail-fix operator package was used in the run; no binary rebuild
was performed solely to publish this paired branch. Follow normal device/build
compatibility checks before reusing a binary in another environment.
