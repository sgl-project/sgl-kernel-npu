# lightning_indexer_v2

面向 CANN **9.2.0** 的 A3 接口，构建参数为 `--arch-version 2201`。

上游原始文档：[lightning_indexer_v2](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/lightning_indexer_v2/README.md)。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 在 CANN 9.2.0、arch22 上选择 `cann_ops_transformers_cann920`，不自动回退。

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2

def lightning_indexer_v2(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q='BSND',
    layout_k='BSND',
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks=None,
    candidate_block_size=None,
    impl='auto'
):
    ...
```

## 功能说明

对每个 query，先计算

```text
score[j] = sum_h w[h] * relu(dot(q[h], k[j, 0]))
```

再按分数选择 TopK。`mask_mode=3` 为右下对齐因果模式；启用压缩时使用上游定义的压缩后位置域。

普通路径调用 `aclnnLightningIndexerV2`；候选路径调用 `aclnnLightningIndexerV3`。候选块的得分为块内最大位置分数，最后一个可达块以 +inf 固定参与块级 TopK。候选块可传给 sparse consumer，两侧必须使用相同块大小及兼容的 KV 位置域。候选只在同一次 forward 内复用。

## 参数与返回值

所有输入位于同一设备。Q 头数 H 为 1..64，K 头数为 1，head_dim 为 128。

| 参数 | 类型 / shape | 默认值 | 约束 |
| --- | --- | --- | --- |
| q | fp16/bf16；BSND `[B,Q,H,128]` 或 TND `[Tq,H,128]` | 必传 | 非空、连续 |
| k | 与 q 同 dtype；BSND `[B,K,1,128]`、TND `[Tk,1,128]` 或 PA_BBND `[P,S,1,128]` | 必传 | 仅 PA_BBND 第 0 轴允许非连续 |
| w | fp32，`q.shape[:-1]` | 必传 | 连续 |
| topk | int | 必传 | 见下方后端约束 |
| cu_seqlens_q | int32 `[B+1]` | None | TND 必传累计边界；BSND 可传 `[0, 各样本有效 Q 长度…]` |
| cu_seqlens_k | int32 `[B+1]` | None | TND 必传累计边界；BSND 不传 |
| seqused_k | int32 `[B]` | None | 有效 K 长度；PA_BBND 必传 |
| cmp_residual_k | int32 `[B]` | None | 压缩率不为 1 且 mask=3 时必传，值小于压缩率 |
| block_table | int32 `[B,max_pages]` | None | PA_BBND 必传 |
| metadata | int32 `[2048]` | None | A3 kernel 不消费，可省略 |
| seqused_q、output_idx_offset | tensor | None | A3 不支持，接口提前拒绝非 None |
| max_seqlen_q | int | -1 | A3 仅接受 -1 |
| layout_q / layout_k | str | BSND / BSND | 与输入物理布局匹配 |
| mask_mode | int | 0 | 0 / 3 |
| cmp_ratio | int | 1 | 1..128 内的 2 的幂 |
| return_value | int | 0 | 0 返回空 FP32 values，1 返回对应分数 |
| impl | str，仅关键字 | auto | auto / cann_ops_transformers_cann920 / torch_native |

PA block size 为 16..1024 内的 16 倍数。可选索引 tensor 使用 int32；位置、长度、块号等内容必须符合上游契约，launch 路径不读回 tensor 验证数值。

| 候选参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| candidate_topk_blocks | int | None | None：普通路径；-1：四输出但关闭候选；正值为 64..2048 内的 64 倍数 |
| candidate_block_size | int | None | 候选路径将 None 视为 8，支持 2/4/8/16/32/64 |

普通路径 `topk` 支持 1..2048，以及 3072/4096/5120/6144/7168/8192；启用候选模式要求 topk<=2048。

`candidate_topk_blocks=None` 返回 `(indices, values)`；显式传入（含 -1）时返回 `(indices, values, candidate_block_indices, candidate_block_length)`。

- indices：int32，shape 为 `[B,Q,1,topk]` 或 `[Tq,1,topk]`，无效槽为 -1。
- values：同设备 FP32；return_value=0 时为空，否则与 indices 同形，无效槽为 -inf。
- candidate_block_indices：int32，shape 的前几维与 indices 相同，末维为 candidate_topk_blocks；无效槽为 -1，块号不含 output offset，槽位无序；候选关闭时为空。
- candidate_block_length：预留的空 int32 tensor。

## cann_ops_transformers_cann920 后端

要求 CANN 9.2.0，wheel 中包含匹配的 vendor、绑定、LICENSE 和 manifest。首次调用可能 JIT 编译 C++ 适配代码，需要编译器及匹配的 Torch / torch_npu。通过 `ops_transformer` profile 构建；metadata 的 AICPU 资源随依赖包分发。

## torch_native 后端

独立使用 FP32 计算，不调用被测绑定。支持 BSND/BSND、cmp_ratio=1、mask=0/3 和可选的 BSND 有效 Q/K 长度；不支持 packed/paged/compressed KV。忽略不透明 metadata。同分选择和输出槽位顺序可能与 CANN 实现不同。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import (
    lightning_indexer_v2, lightning_indexer_v2_metadata, sparse_lightning_indexer,
)

q = torch.randn(1, 1, 32, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(1, 512, 1, 128, device="npu", dtype=torch.bfloat16)
w = torch.randn(1, 1, 32, device="npu", dtype=torch.float32)
indices, values, blocks, length = lightning_indexer_v2(
    q, k, w, 16, candidate_topk_blocks=64, candidate_block_size=8,
)
```

上游版权：Huawei Technologies Co., Ltd.；CANN Open Software License Agreement Version 2.0。
