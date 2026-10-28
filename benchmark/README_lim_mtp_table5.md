# Full-visible Q4 LIM operator benchmark

`bench_lim_mtp_table5.py` compares Ascend vendor LightningIndexer (LI) with
native full-visible LIM for S=65536, batch=16, Q=4, K=2048, and C=12288.
It requires an A3-compatible `sgl-kernel-npu` build with the `full_visible`
operator argument, BF16 index inputs, and an idle NPU. It does not measure
CopySFA, KV payload transfers, model inference, or end-to-end serving.

```bash
python3 benchmark/bench_lim_mtp_table5.py --misses 410 --warmup 20 --iterations 200 --rounds 3 --output lim95.json
python3 benchmark/bench_lim_mtp_table5.py --misses 737 --warmup 20 --iterations 200 --rounds 3 --output lim91.json
```

For each request the fixture uses vendor LI's causal TopK union and builds a
full resident map with exactly the requested number of unique selected sources
absent. It excludes the four rewritten positions from initial residency, then
restores the original map **before each LIM start event**. It checks causal
selection, LI/LIM selected-set equality, per-route and union miss counts,
copy-plan uniqueness, and final source-to-slot consistency. The reported hit
rate is `1 - unique union misses / (4 * 2048)`, not the per-query access hit
rate. Raw per-iteration samples, input hashes, environment and loaded library
hashes are saved in each JSON file. Use `--only-lim` to time only LIM while
still computing the LI reference outside the measured intervals.

One A3/BF16 single-device run produced the following paired eager NPU-event
means (20 warmups, three rounds of 200 samples per operator). Events may
include device idle time while waiting for host submission; these are not
isolated device-kernel durations.

| Union misses per request | Defined hit rate | LI mean | LIM mean | Mean difference |
| ---: | ---: | ---: | ---: | ---: |
| 410 | 94.9951171875% | 481.215 us | 785.881 us | 304.666 us |
| 737 | 91.00341796875% | 507.830 us | 816.257 us | 308.427 us |

The paper's Table 5 first row reports 395.812 mean misses, LI 138.814 us,
LIM 167.721 us and 28.907 us difference. Its C8/FP8 interface differs from
this A3/BF16 measurement; exact hardware and timing equivalence is not
established.
