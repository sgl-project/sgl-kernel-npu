# BF16 hyper-connection epilogue (Ascend C / CANN ACLNN)

Native `AICORE` (`KERNEL_TYPE_AIV_ONLY`) operators behind the explicit
`impl="ascendc"` backend of `sgl_kernel_npu.operators.norm.grouped_gemma_rmsnorm`,
`sgl_kernel_npu.operators.hyper_conn.hc_mix`, and
`sgl_kernel_npu.operators.hyper_conn.hc_combine`.

## Registered CANN operators (`SUPPORTED_ARCHS 2201`)

- `hyper_conn_grouped_norm` (`SglHyperConnGroupedNorm`) -> `torch.ops.npu.hyper_conn_grouped_norm`
- `hyper_conn_scaled_silu` (`SglHyperConnScaledSilu`) -> `torch.ops.npu.hyper_conn_scaled_silu`
- `hyper_conn_branch_mix` (`SglHyperConnBranchMix`) -> `torch.ops.npu.hyper_conn_branch_mix`
- `hyper_conn_branch_combine` (`SglHyperConnBranchCombine`) -> `torch.ops.npu.hyper_conn_branch_combine`

## Numerics

Each vector kernel evaluates its intermediate arithmetic in float32 and rounds
to BF16 once when writing to global memory (`CAST_RINT`). Inside `hc_mix`, the
projections execute via BF16 `F.linear`, so `low`, `activated`, and
`gate_logits` cross global memory in BF16 (unlike the default Triton path from
#807, which keeps intermediate projections in float32).

## Tiling tables

`op_host/hyper_conn_tiling.h` preserves the Ascend910_9382 measured sweep over
`hc = 4`, `width` in `{2048, 2560}`, and decode/prefill row counts, clamped to
available work and `platform.GetCoreNumAiv()`.
