# 量化稀疏 Shared-KV 注意力

```python
from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv,
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

metadata = kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="BSND", layout_kv="PA_ND",
    seqused_kv=cpu_int32_lengths, batch_size=1,
    max_seqlen_q=1, max_seqlen_kv=256,
    has_ori_kv=True, has_cmp_kv=False,
)
output, empty_lse = kv_quant_sparse_attn_sharedkv(
    q, ori_kv=packed_fp8_kv, ori_block_table=block_table,
    seqused_kv=npu_int32_lengths, sinks=sinks, metadata=metadata,
)
```

公开接口显式列出原生算子的参数，另外提供仅限关键字的 `impl`。
`auto` 和 `ascendc` 在昇腾 950 上调用现有的
`torch.ops.npu.kv_quant_sparse_attn_sharedkv`。独立的 `torch_native`
参考实现只覆盖 BF16、BSND/PA_ND、未压缩 KV 的普通分页滑动窗口路径，
会把数据复制到 CPU，仅用于精度对照。压缩 KV、稀疏索引和 TND 路径仍由
已有原生测试覆盖。

metadata host 接口要求 CPU 上连续的 int32 序列长度，返回 NPU 上的
int32[1024] 调度数据。`auto` 和 `host` 都调用原有
`torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host`。SGLang
的 A5 DSpArk 路径改用这两个公开函数；原有的低层 `torch.ops.npu`
绑定仍供其他调用方使用。

## CANN 重构下的构建状态

#880 的 `--ops` 新构建要求 CANN AICORE op host、生成的 ACLNN 接口和
Torch adapter。本算子目前仍通过旧 Torch host 启动，metadata 调度器是
纯 CPU 实现，因此还不能用 `--ops kv_quant_sparse_attn_sharedkv` 选择。
运行时继续构建 Ascend 950 的旧版 wheel，直到原生构建路径完成迁移。
