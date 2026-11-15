# qsa_prefill

## Hardware support

arch22 is split because the two members of that family are validated
differently. A3 means Ascend910_93 series, A2 means Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `ascendc` | Supported | Unverified | Not implemented |
| `torch_native` | Supported | Unverified | Unverified |
| `triton` | Not implemented | Not implemented | Not implemented |
| `tilelang` | Not implemented | Not implemented | Not implemented |

The `ascendc` implementation is registered for arch22, which covers both A2
and A3, and `SGL_KERNEL_ENABLE_A3_ONLY_OPS` compiles its kernels for every
SoC outside the Ascend950 series. Nothing excludes A2: the kernels use
`__CCE_AICORE__ == 220`, which both families report, the host guard rejects
only Ascend950, and the launch geometry follows `GetCoreNumAic()`. A2 is
marked unverified rather than unsupported because no A2 measurement exists
yet, not because a limit is known. Only an Ascend950 build lacks the native
helpers, and calling the operator there raises `AttributeError` on
`torch.ops.npu.qsa_prefill_prepare`.

Hardware validation used Ascend910_9382 and Ascend910_9362 (910C) with
CANN 9.0.0/9.1.0, torch 2.10 and torch_npu 2.10. arch35 is not implemented.

## Signature

The default option is `auto`, registered to the same entry point as `ascendc`.
Omitting `impl` or passing `"auto"` selects the Ascend C pipeline.

```python
from sgl_kernel_npu.operators.attention import qsa_prefill

def qsa_prefill(
    q,
    k,
    v,
    blocks,
    request_table,
    request_row,
    length,
    base,
    scale=1 / 16,
    *,
    impl="auto",
):
    ...
```

## Description

Computes BF16 main attention for ratio-4 compressed QSA prefill with head
dimension 256, over 512 preselected logical blocks per query. The caller
supplies the indexer's selection; this operator computes neither indexer scores
nor Top-K, and it performs no distributed communication.

For query row $r$ the visible logical positions are the four token positions of
each nonnegative selected block, followed by that row's own causal remainder:

$$
P_r=\{4b+j \mid b \in \mathrm{blocks}_r,\ b \ge 0,\ 0 \le j < 4,\ 4b+j < L\}
\ \cup\ \{p \mid v_r-(v_r \bmod 4) \le p < v_r,\ 0 \le p < L\},
\quad v_r=\mathrm{base}+r+1
$$

Here $L$ is `length`. Selection order and duplicates are preserved; each
position is mapped to a physical slot $s=T[\mathrm{request\_row},p]$ through the
request table and dropped when $s \notin [0,\mathrm{capacity})$. With local
query head $h$ belonging to KV head $\lfloor h/(H/KV) \rfloor$:

$$
Y_{r,h,:}=\sum_{p \in P_r}\mathrm{softmax}_p
\left(\alpha \sum_{d} Q_{r,h,d}K_{s(p),\lfloor h/(H/KV)\rfloor,d}\right)
V_{s(p),\lfloor h/(H/KV)\rfloor,:}
$$

$\alpha$ is `scale`. Rows whose $P_r$ contains no valid position produce zeros
rather than NaN. Logits, softmax and the value accumulation use float32; the
result is rounded to BF16 once.

## Parameters and return value

The public operator validates the shared contract listed below before
selecting an implementation, so both backends reject the same inputs.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[R, H, 256]` | `torch.bfloat16` | Required | Queries; contiguous, on an NPU device. `R == 0` returns an empty result. `(H, KV)` must be `(16, 2)`, `(8, 1)`, `(6, 1)`, `(3, 1)`, `(2, 1)`, `(24, 2)` or `(12, 1)`. |
| `k` | `torch.Tensor` | `[capacity, KV, 256]` | `torch.bfloat16` | Required | Key cache; contiguous, on q's device. `capacity < 2**31`, and must be nonempty when `length > 0`. |
| `v` | `torch.Tensor` | `[capacity, KV, 256]` | `torch.bfloat16` | Required | Value cache; same shape, dtype and device as `k`, contiguous. |
| `blocks` | `torch.Tensor` | `[R, 512]` | `torch.int32` | Required | Selected logical blocks per query; contiguous, on q's device. Negative entries and blocks whose positions reach `length` are ignored. |
| `request_table` | `torch.Tensor` | `[requests, max_length]` | `torch.int32` | Required | Logical-to-physical slot table on q's device; `requests > 0` and unit column stride. Row stride may include padding. |
| `request_row` | `torch.Tensor` | `[1]` | `torch.int64` | Required | Contiguous NPU tensor holding the caller's valid row index into `request_table`. Read on the device only. |
| `length` | `int` | — | — | Required | Sequence length in tokens; `0 <= length <= request_table.shape[1]`. CPU metadata; never read back from the device. |
| `base` | `int` | — | — | Required | Logical position preceding q's first row, normally `length - R`. Requires `-2**31 <= base` and `base + R < 2**31`. |
| `scale` | `float` | — | — | `1 / 16` | Logit multiplier; must be finite and positive. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"` or `"torch_native"`. `auto` selects Ascend C. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, H, 256]` | `torch.bfloat16` | Newly allocated output on q's device. Rows without visible tokens are zero. Inputs are not modified. |

Invalid metadata raises `ValueError` before any kernel launch. The caller
handles head sharding and KV replication: at a TP degree above the global KV
head count, pass the replicated KV head for that rank.

For a 24 Q / 2 KV model such as Qwen3.8-Flash-Next, the local layouts are
`(24, 2)` at TP1, `(12, 1)` at TP2, `(6, 1)` at TP4 and `(3, 1)` at TP8; all
four are supported. For a 16 Q / 2 KV model such as Whittle-Next they are
`(16, 2)`, `(8, 1)` and `(2, 1)` at TP1, TP2 and TP8.

## ascendc backend

- The QK, online-softmax, PV and rescale pipeline and the physical run
  preparation are Ascend C, registered as `torch.ops.npu.qsa_prefill_prepare`
  and `torch.ops.npu.qsa_prefill_runs`. These are internal helpers; pass only
  metadata produced by this operator, not hand-written run descriptors.
- Query packing, the full 512-entry selection equality check and the per-query
  causal tail merge are Triton NPU kernels, so this backend needs both
  runtimes. Base attention is shared only within exact 16/8/4/2/1 query groups
  whose selected block lists are equal entry by entry, in the same order.
- Base output and LSE stay in float32 through the tail merge. Results are not
  bitwise equal to other attention implementations.
- A 16-query sharing group occupies `16 * (H / KV)` rows of the per-stage GM
  workspace block, which holds 128 rows, so one launch requires `H / KV <= 8`.
  `(16, 2)`, `(8, 1)`, `(6, 1)`, `(3, 1)` and `(2, 1)` meet that and run as a
  single launch. Attention is independent per query head, so `(24, 2)` and
  `(12, 1)`, which need twelve queries per KV head, are
  evaluated as four and two launches over consecutive six-head slices and
  concatenated in head order. Those slices are copied, and `(24, 2)` also
  copies each KV head once, so these two layouts are slower than a layout the
  kernel takes directly.
- Tiling is immutable and cached by full shape, scale, device and both local
  head counts. Warm a shape before capturing it in an NPU graph.
- Long shared prefills are considerably faster than independent-query
  evaluation; short or fully unshared selections can be slower. This operator
  is an explicit API, not a default dispatch path.

## torch_native backend

- Expands the selection, maps it through the request table and evaluates dense
  softmax attention with float32 accumulation. It shares no code with the
  Ascend C path: no packing, no sharing-group detection, no run descriptors and
  no separate tail merge, which makes it usable as an accuracy reference.
- Rows are processed in chunks because the gathered key and value tensors grow
  with `R x 2052 x KV x 256`. It is a reference, not a performance path.
- Reduction order differs from the kernel, so bitwise equality is not expected.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.attention import qsa_prefill

rows, heads, kv_heads, length = 17, 16, 2, 8192
q = torch.randn(rows, heads, 256, device="npu", dtype=torch.bfloat16)
k = torch.randn(length, kv_heads, 256, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
blocks = torch.arange(512, device="npu", dtype=torch.int32).repeat(rows, 1)
request_table = torch.arange(length, device="npu", dtype=torch.int32)[None]
request_row = torch.zeros(1, device="npu", dtype=torch.int64)

print(list_impls(qsa_prefill))  # ['ascendc', 'auto', 'torch_native']
args = (q, k, v, blocks, request_table, request_row, length, length - rows)
out = qsa_prefill(*args)  # Default auto selects Ascend C
out_native = qsa_prefill(*args, impl="torch_native")
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
