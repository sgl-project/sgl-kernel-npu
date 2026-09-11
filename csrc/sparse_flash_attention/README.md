# sparse_flash_attention

Ported from vLLM-Ascend's vendored `csrc/attention/sparse_flash_attention`.

**Provenance:** last upstream change to that directory was
`21382607d15a835728d9ea493ab101e7209799b5` (2026-07-29); copied from a tree at vllm-ascend
`748acedfea31b795e507f9c8175585f824328d8c`.

## Why this is vendored rather than called through torch_npu

CANN ships its own build of this operator as `torch_npu.npu_sparse_flash_attention`, and SGLang's
Ascend MLA decode calls it today. That build refuses to return the log-sum-exp under a paged KV
layout:

```cpp
OP_CHECK_IF(*opParamInfo_.returnSoftmaxLse && kvLayout_ == SFALayout::PA_BSND,
            … "When return_softmax_lse is true, layout_kv does not support PA_BSND"),
            return ge::GRAPH_FAILED);
```
— `ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp:1994`

`PA_BSND` is that operator's only paged layout, so paged and LSE are mutually exclusive there.
Decode Context Parallelism needs a per-rank LSE to weight the cross-rank merge, which makes that
refusal the blocker. The vendored build has no such clause — its layout validation checks dimensions
and layout pairing only — and vLLM-Ascend runs `PA_BSND` together with `return_softmax_lse=True` in
production (`vllm_ascend/device/device_op.py:429-445`, feeding the merge at
`vllm_ascend/attention/context_parallel/sfa_cp.py:1250`).

It is registered as **`torch.ops.npu.npu_sparse_flash_attention_lse`** — a distinct name — so the
operator CANN provides is untouched and the non-DCP serving path is unaffected.

## Returned values

`(attention_out, softmax_max, softmax_sum)`. The LSE is natural-log and the caller combines the two
halves as `lse = softmax_max + log(softmax_sum)`, because `softmax_sum` is `sum(exp(qk - max))`.

With `layout_query="TND"` the softmax outputs are shaped `(N2, T, G)` — kv heads, tokens, query heads
per kv head. With `layout_query="BSND"` they are `(B, N2, S, G)`. Those shapes are not a choice: the
tiling derives the expected layout from `layout_query` and rejects the call if the tensors disagree
(`SFAInfoParser::GetSoftmaxMaxAndSumLayout`). When `return_softmax_lse` is false both are empty.

### A row with no valid sparse indices

Measured on A3 (Ascend910_9382, CANN 9.1.0), 2026-09-09, by
`glm5.2_testing/p3b_sparse_flash_attention_lse_probe.py`:

| | |
|---|---|
| `softmax_max` | `-2e38` — the kernel's `SOFTMAX_MIN_NUM` sentinel, **not** `-inf` |
| `softmax_sum` | the slot count (`topk`), **not** `0` |
| `softmax_max + log(softmax_sum)` | about `-2e38`, a large finite negative |

`softmax_sum` is the slot count because with every index masked every slot equals the max, so each
contributes `exp(0) = 1`.

**This needs no guard in the consumer.** Every LSE combine centres on the global max and
exponentiates the difference, and `exp(-2e38 - O(1))` underflows to exactly `0.0` — the same weight
`-inf` would have produced. Checked in SGLang against all three implementations:
`cp_lse_ag_out_rs_{mha,mla}` (`srt/layers/dcp/comm.py:94-98`, which also already has a
`nan_to_num` on the scale), the Triton combine (`kernels/ops/attention/dcp_kernels.py:540-570`)
and its torch reference (`:655-680`). No LSE is ever multiplied by a base-conversion constant —
`is_lse_base_on_e` only selects `exp` vs `exp2` — so the magnitude cannot overflow either.

Because `softmax_sum` is never `0`, nothing evaluates `log(0)`; an epsilon on it would be pointless.

Note the sentinel is in fact *safer* here than `-inf` would be. If every rank were empty, `-inf`
drives `lse_max` into its `-inf` special case, every weight to `0`, and `acc / weight_sum` to
`0/0 = NaN`; `-2e38` gives a centred `0`, weights of `1`, and a clean zero output. That case should
not arise — the replicated indexer always selects at least one position per token and each position
has exactly one owner — but it is the better failure mode to have.

## What differs from upstream, and why

Every difference comes from building as part of a plain shared library instead of a CANN custom-op
package. Nothing here changes what the kernel computes; the `arch22/` kernel bodies are untouched.

| upstream | here | reason |
|---|---|---|
| `op_host/sparse_flash_attention_def.cpp` (`ops::OpDef` + `OP_ADD`) | `op_host/sparse_flash_attention_def.h` (`ge_helper::OpDef`) | Registering the op would create a custom vendor-package op, which shadows CANN's build of the same name via `load_priority` in `opp/vendors/config.ini`. Same interface table, consumed by `SetToContext()` instead. |
| `sparse_flash_attention_torch_adpt.h` → `EXEC_NPU_CMD(aclnnSparseFlashAttention, …)` | `op_host/sparse_flash_attention.cpp` → `EXEC_KERNEL_CMD` | No ACLNN wrapper without an op package. Output-shape logic is upstream's, unchanged. |
| `op_host/op_api/aclnn_*`, `op_host/sparse_flash_attention_infershape.cpp` | dropped | The launcher allocates the outputs itself. |
| `register/tilingdata_base.h` macros | `op_host/sparse_flash_attention_tiling_data.h` (POD structs) | Macros are op-package-only. Field order preserved — the kernel reads the struct back from GM. |
| `GET_TILING_DATA_WITH_STRUCT(...)` in the kernel entry | `reinterpret_cast<const __gm__ ...*>(tiling)`, and `__gm__` added to the tiling pointer in `arch22/`'s `Init()` signatures | That macro belongs to the op-packer's tiling plumbing and will not take a plain POD struct type: it rejects the type name as "does not refer to a value" and never declares the variable. The tiling buffer is viewed in place instead and its scalars read from GM, exactly as `csrc/sparse_attn_sharedkv` does. Every use is a scalar field read, so no local copy is needed. |
| `err/ops_err.h` | `op_host/sparse_flash_attention_ops_compat.h` | Same; keeps the formatted messages so a rejected call says *why*. |
| `platform/soc_spec.h` `NpuArch` / `GetCurNpuArch()` | `SFANpuArch`, from `GetSocVersion()` | Neither is used anywhere else in this repo; `GetSocVersion()` is. |
| compile-time template selection (`GET_TPL_TILING_KEY` + one binary per combination + injected `ORIG_DTYPE_*`) | runtime `dispatchKey` switch in `op_kernel/sparse_flash_attention.cpp` | One binary is built here, reached through `aclrtlaunch_sparse_flash_attention`. Same axes; the dtype axis moves into the key because the `ORIG_DTYPE_*` macros only exist under the packer. |
| `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` blocks | the switch cases | Same combination list, expressed where it is now used. |
| `-mllvm -cce-aicore-hoist-movemask=false` | dropped | CANN 9.1.0's bisheng rejects it as an unknown argument. Restore if a later CANN accepts it. |
| `op_kernel/arch35/` (A5 / Ascend 950) | not ported | No A5 hardware on this project, so it could not be tested. `GetNpuInfo()` rejects non-A2/A3 SOCs rather than letting a call through to a kernel that was never built. Adding it means copying `arch35/` back from upstream, restoring the `#if (__CCE_AICORE__ == 310)` branch in the kernel entry, and extending the switch with the `IS_SPLIT_G` axis — which is why that axis is still in the dispatch key. |

## Sparse-indices contract

`-1` marks an invalid/padding slot, and valid entries must be **left-aligned** with the padding as a
suffix — upstream states the requirement in its own README: 需要保证每行有效值均在前半部分、无效值均在后半部分.
