# DSPark Triton-Ascend operators

This port is paired with SGLang branch `feat/npu-dspark-triton-ops` and builds on
the Markov sampler in commit `998f742`. GPU sources and torch test oracles were
taken from SGLang commit `7036f4b0ad`.

## Ownership and dispatch

The package contains compute kernels and tensor launchers, without importing
SGLang. Runtime sampling metadata, probability preparation for top-k/top-p, and
chain-buffer ownership remain in SGLang. The SGLang NPU integration imports the
NPU operators with the same names as the GPU interfaces. Operator `execute`
methods call Triton directly, without testing `Tensor.is_cuda` or falling back to
native torch compute. Torch is still used for allocation, dtype/layout conversion
and random-number buffers; this does not replace all torch operations in the
model or scheduler.

| Interface | Package module |
| --- | --- |
| Markov sampling | `sample.markov_sample` (existing) |
| Markov greedy projection/selection | `sample.dspark_draft` |
| Commit KV projection | `sample.dspark_kv_proj` |
| Greedy verification, row argmax | `sample.dspark_greedy` |
| Temperature softmax, bonus gathers, mixed accept, accept lengths | `sample.dspark_accept` |
| Chain rejection sampling | `sample.chain_speculative_sampling` (existing) |
| Confidence-based verify scheduling | `sample.dspark_schedule` |
| Compact indexing, token ids, ragged verify inputs | `sample.dspark_compact` |
| Compact scatter, commit layout, output token assembly | `sample.dspark_verify_window` |
| Paged draft attention metadata | `attention.dspark_metadata` |
| Per-request ring metadata | `attention.dspark_ring` |
| Causal prefill metadata and page tables | `attention.dspark_causal_metadata` |
| Ragged bucket lengths and query indptr | `sample.dspark_ragged` |
| Committed cache mapping | `attention.dspark_commit_swa` |
| Fused K norm, RoPE and KV write | `attention.dspark_kv_write` |
| Simulated bonus sampling | `sample.dspark_simulated_bonus` |

Vector kernels use persistent one-dimensional grids capped at the detected vector
core count and `num_warps=1`. Logical GPU grid coordinates are reconstructed
inside each persistent loop. KV matmul uses the detected AI core count. Schedule
selection assigns a request to one program to avoid cross-program atomics.

Argmax explicitly preserves the first index among ties and the first NaN index.
Output token assembly uses integer bit selection so token ids above `2**24` are
not rounded through FP32. Compact cache locations retain the existing NPU int32
dtype (the GPU helper uses int64); positions and token ids remain int64.

Commit KV projection supports floating weights/activations (FP16, BF16, FP32) and
standard FP8 E4M3 weights with floating 128x128 block scales. INT8 weights and
packed/MXFP8 scale formats are not supported by this new projection launcher;
they raise rather than silently using the old native Linear path. These formats
need a separate implementation and device validation before enabling them.
NPU block verification is unchanged and remains unsupported.

## Tests

The following tests require the rebuilt/installed NPU package, CANN,
Triton-Ascend and real Ascend hardware:

```bash
python -m pytest -q \
  tests/python/sgl_kernel_npu/test_dspark_kernel_parity.py \
  tests/python/sgl_kernel_npu/test_dspark_simulated_bonus.py \
  tests/python/sgl_kernel_npu/test_dspark_npu_edges.py
python -m pytest -q \
  tests/python/sgl_kernel_npu/test_markov_sample.py \
  tests/python/sgl_kernel_npu/test_chain_speculative_sampling.py
```

`test_dspark_kernel_parity.py` preserves all 20 `_case_*` functions from the CUDA
parity suite, including original shapes, dtypes, assertions and tolerances. Each
case is exposed as an individual unittest. Imports/device setup and the parity
adapter change; the torch oracle code is copied into test-only
`dspark_reference/`, without a dependency on an installed SGLang checkout.

`test_dspark_simulated_bonus.py` preserves all three original simulated-bonus
cases. NPU edge tests additionally cover strided argmax, NaNs/infinities/ties,
every greedy prefix length, ring-window floors, large integer token ids, partial
matmul tiles and floating dtypes, graph replay, and masked KV writes in both RoPE
styles. The existing Markov and chain-sampling tests remain in place.

Hardware tests skip when torch_npu or an NPU is unavailable. A skip is not a
numerical validation result. No NPU compilation, numerical, performance or GPQA
result has been obtained for this complete port yet; the earlier Markov-only
GPQA result does not validate these changes.

Host-only API/launch checks can run without CANN or Triton:

```bash
python -m pytest -q tests/host/test_dspark_port.py
```

These checks intercept kernel launches to validate argument binding, output
shape/dtype, persistent grid bounds and absence of selected native fallbacks.
They do not execute or simulate device kernel arithmetic.

## Installation and integration

Source changes do not update an already installed wheel. In an Ascend build
environment, rebuild and install the package before running the paired SGLang
branch:

```bash
bash build.sh -a kernels
pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
```

The paired SGLang change cannot be deployed with the old `2026.9.0.post9` wheel.
Before merging/publishing the integration, release a wheel containing these
operators and update the Docker/CI pins to that actual release. No future version
number is invented or pinned by this development branch.
