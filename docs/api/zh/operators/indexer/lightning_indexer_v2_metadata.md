# lightning_indexer_v2_metadata

面向 CANN **9.2.0** 的 A3 接口，构建参数为 `--arch-version 2201`。

上游原始文档：[lightning_indexer_v2_metadata](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/lightning_indexer_v2_metadata/README.md)。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | 支持 | 不支持 |

## 函数原型

默认 `auto` 在 CANN 9.2.0、arch22 上选择 `cann_ops_transformers_cann920`，不自动回退。

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2_metadata

def lightning_indexer_v2_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    *,
    impl='auto'
):
    ...
```

## 功能说明

生成供 indexer 使用的不透明调度信息。上游 Python 名为 `lightning_indexer_metadata`，构建项为 `lightning_indexer_v2_metadata`，ACLNN 为 `aclnnLightningIndexerV2Metadata`，通过 AICPU 执行。

A3 kernel 不消费该调度信息，主算子可使用 `metadata=None`；绑定与依赖一起打包，便于提供上游绑定引用的完整符号。调度数据不适合独立数值比较，通过与主算子的组合验证。

## 参数与返回值

| 参数 | 类型 / shape | 默认值 | 约束 |
| --- | --- | --- | --- |
| num_heads_q | int | 必传 | 1..64 |
| num_heads_k | int | 必传 | 1 |
| head_dim | int | 必传 | 128 |
| topk | int | 必传 | 与主算子一致：1..2048，或 3072/4096/5120/6144/7168/8192 |
| cu_seqlens_q、cu_seqlens_k | int32 tensor `[B+1]` | None | TND 使用累计边界；BSND K 不传 |
| seqused_q、seqused_k、cmp_residual_k | int32 tensor `[B]` | None | 按上游 metadata 契约使用；A3 主算子不支持 seqused_q |
| batch_size | int | None | 上游将 None 转为 0 |
| max_seqlen_q、max_seqlen_k | int | None | 上游将 None 转为 -1 |
| layout_q | str | None | 上游默认 BSND，可使用 TND |
| layout_k | str | None | 上游默认 BSND，可使用 TND / PA_BBND |
| mask_mode | int | None | 上游默认 0，另支持 3 |
| cmp_ratio | int | None | 上游默认 1，取 1..128 内的 2 的幂 |
| impl | str，仅关键字 | auto | auto / cann_ops_transformers_cann920 |

返回同一 NPU 上的 `int32[2048]` tensor。长度、布局、head 数、压缩率和 mask 应与主调用匹配。该接口不修改输入。

## cann_ops_transformers_cann920 后端

要求 CANN 9.2.0，wheel 中包含匹配的 vendor、绑定、LICENSE 和 manifest。首次调用可能 JIT 编译 C++ 适配代码，需要编译器及匹配的 Torch / torch_npu。通过 `ops_transformer` profile 构建；metadata 的 AICPU 资源随依赖包分发。

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
metadata = lightning_indexer_v2_metadata(
    32, 1, 128, 16, batch_size=1, max_seqlen_q=1, max_seqlen_k=512,
    layout_q="BSND", layout_k="BSND",
)
indices, values = lightning_indexer_v2(q, k, w, 16, metadata=metadata)
```

上游版权：Huawei Technologies Co., Ltd.；CANN Open Software License Agreement Version 2.0。
