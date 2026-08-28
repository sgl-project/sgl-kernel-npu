# torch.ops.npu.lightning_indexer_v2

## Origin

Ported from `vllm-ascend/csrc/attention/lightning_indexer`. It is a newer revision of the
kernel behind [`lightning_indexer`](../lightning_indexer/README.md), which remains in the
tree unchanged; the two ops are independent.

Compared to `lightning_indexer` this version adds:

- an **arch35** kernel path for <term>Ascend 950PR/950DT</term> (VF top-k), alongside the
  existing **arch22** path for <term>Atlas A2/A3</term>. `lightning_indexer` is A3-only;
  `lightning_indexer_v2` is therefore *not* gated on `SGL_KERNEL_ENABLE_A3_ONLY_OPS`.
- an optional second output **`sparse_values`** (`return_values`).
- **fp32 `weights`** while `query`/`key` stay fp16/bf16 (upstream `DT_W_FLAG`).
- the `pre_tokens` / `next_tokens` attributes.

## Product Support Status

| Product                                                    | Supported |
| ---------------------------------------------------------- | :-------: |
| <term>Ascend 950PR / Ascend 950DT</term>                     |     √     |
| <term>Atlas A3 Training/Inference Product Series</term>      |     √     |
| <term>Atlas A2 Training/Inference Product Series</term>      |     √     |

## Function Description

`LightningIndexer` computes the Top-$k$ positions corresponding to each token. For an Index
Query $Q_{index}\in\R^{g\times d}$ corresponding to a certain token, given the context Index
Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$, where $g$ is the group size for GQA,
$d$ is the dimension of each head, and $S_{k}$ is the context length:

$$
Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

## Function Prototype

```
torch.ops.npu.lightning_indexer_v2(
    query, key, weights,
    actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None,
    layout_query='BSND', layout_key='BSND',
    sparse_count=2048, sparse_mode=3,
    pre_tokens=None, next_tokens=None, return_values=False,
) -> (Tensor, Tensor)
```

Returns `(sparse_indices, sparse_values)`. When `return_values` is `False`, `sparse_values`
is an empty tensor.

## Parameter Description

Dimension meanings: B (Batch Size), S (Sequence Length), H (Head Size), N (Head Num),
D (Head Dim, D=H/N), T (cumulative sum of sequence lengths across the batch). S1/N1 refer to
`query`, S2/N2 to `key`.

- **query** (`Tensor`): required, contiguous, ND. `bfloat16` or `float16`.
- **key** (`Tensor`): required, contiguous, ND. `bfloat16` or `float16`. When `layout_key` is
  `'PA_BSND'` the shape is `[block_count, block_size, N2, D]`.
- **weights** (`Tensor`): required, contiguous, ND. Same dtype as `query`, **or** `float32`.
- **actual_seq_lengths_query** (`Tensor`, optional): `int32`, 1-D of length B. Required when
  `layout_query` is `'TND'`, where each element is the prefix sum of tokens up to and
  including that batch (non-decreasing, non-negative), and its length defines B.
- **actual_seq_lengths_key** (`Tensor`, optional): `int32`, 1-D of length B.
- **block_table** (`Tensor`, optional): `int32`, ND. Required for PageAttention; 2-D, first
  dim B, second dim at least maxBlockNumPerSeq.
- **layout_query** (`str`, optional): `'BSND'` or `'TND'`. Default `'BSND'`.
- **layout_key** (`str`, optional): `'PA_BSND'`, `'BSND'` or `'TND'`. Default `'BSND'`.
  Outside PageAttention it must equal `layout_query`.
- **sparse_count** (`int`, optional): number of positions kept by the topK phase, 1-2048.
- **sparse_mode** (`int`, optional): `0` (defaultMask) or `3` (rightDownCausal). Default `3`.
- **pre_tokens** / **next_tokens** (`int`, optional): currently only `INT64_MAX` is accepted,
  matching upstream.
- **return_values** (`bool`, optional): also return the Top-$k$ scores. Default `False`.

## Return Value

- **sparse_indices** (`Tensor`): `int32`, ND.
- **sparse_values** (`Tensor`): same dtype as `query`, ND. Empty when `return_values=False`.

## Constraints

- Inference scenarios; graph mode supported.
- N in `query` supports 64, N in `key` supports 1.
- D of `query` and `key` must both be 128.
- `query` and `key` must share a dtype; `weights` matches it or is `float32`.
- `block_size` must be a multiple of 16, up to 1024.

## Port notes

The upstream sources are reused as close to verbatim as practical so they stay diffable
against `vllm-ascend`. The deviations are:

| Upstream | Here | Why |
| --- | --- | --- |
| `ASCENDC_TPL_ARGS_DECL` / `GET_TPL_TILING_KEY` codegen | `op_kernel/lightning_indexer_v2_tiling_key.h` + an explicit `switch` in the kernel entry | sgl-kernel-npu emits a single kernel binary, so the template combinations upstream lists in `ASCENDC_TPL_SEL` are expanded by hand. |
| `BEGIN_TILING_DATA_DEF(LITilingData)` | POD in `op_host/tiling/lightning_indexer_v2_tiling_data.h` | No CANN op-project codegen; the struct is shared by host and device. |
| `err/ops_err.h` (`OP_CHECK_IF`, `OP_LOGE`, ...) | `op_host/tiling/lightning_indexer_v2_ops_compat.h` | Those macros only exist inside a CANN op project. The shim maps them onto `TORCH_CHECK`, which keeps the ~800 lines of tiling checks byte-identical to upstream. |
| `aclnn` + `EXEC_NPU_CMD` | `EXEC_KERNEL_CMD` + `ge_helper` | This repo uses kernel-direct-launch, not a registered CANN operator. |
| `context->SetBlockDim/SetTilingKey/GetRawTilingData` | carried in the tiling POD | Not available on the `ge_helper::TilingContext` shim. |
| `CeilDiv(T, T)` | `CeilDiv(T1, T2)` | CANN 9.1.0's kernel headers have no second `CeilDiv` overload, so mixed-type calls in the service layer fail to deduce. |
| `-mllvm -cce-aicore-hoist-movemask=false` applied unconditionally | applied only for `Ascend950*` | The A2/A3 bisheng rejects the flag as an unknown argument. |
| — | host-only `EVENT_ID4..7` fallbacks in `lightning_indexer_v2_common.h` | The `--cce-host-only` stub pass defines no aicore target macros, so the compiler builtins `EVENT_ID4..7` are missing there. |

Namespaces are nested under `sglang::npu_kernel::liv2` (device) and `sglang::LIV2Host`
(host) so nothing collides with the older `lightning_indexer` op in the same library.

## Usage Example

See [test_lightning_indexer_v2.py](../../tests/python/sgl_kernel_npu/test_lightning_indexer_v2.py).
