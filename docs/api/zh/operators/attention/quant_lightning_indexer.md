# 量化 Lightning 索引器

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `cann_9_1` | 支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.attention import quant_lightning_indexer

def quant_lightning_indexer(q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

因果 TND 查询和 PA_BBND key；A3 用例使用 INT8、FP16 scale/weight，A5 用例使用 FP8、FP32 scale/weight。

`score[j] = Σ_h weights[h] × ReLU(dot(dequant(Q[h]), dequant(K[j])))`，在因果前缀中选择压缩 key 的逻辑 Top-K；并列分数可能产生不同合法顺序。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| q | Tensor | 必填 | [T,H,D] TND or [B,S,H,D] BSND; NPU in production |
| k | Tensor / int | 必填 | Indexer: paged [pages,page_size,1,D]; MoE: requested expert count |
| weights | Tensor | 必填 | [T,H], indexer head weights |
| q_scale | Tensor | 必填 | [T,H], query dequantization scales |
| k_scale | Tensor | 必填 | [pages,page_size,1,1], key scales |
| topk | int | 必填 | Requested number of logical compressed keys |
| quant_mode | int / str | 必填 | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 关键字参数

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| cu_seqlens_q | int32 Tensor | [B+1] cumulative query offsets, required by the reference. |
| seqused_k / cmp_residual_k | int32 Tensor | [B] compressed key lengths / uncompressed residual lengths. |
| block_table | int32 Tensor | [B,max_pages] paged key mapping. |
| metadata | Tensor / ReferenceMetadata / None | Matching metadata implementation only. |
| layout_q / layout_k | str | Reference defaults TND / PA_BBND. |
| mask_mode / cmp_ratio | int | Reference defaults causal 3 / compression ratio 4. |
| return_value | int | Default 0 produces empty values; nonzero score output is 不支持 on cann_9_1. |
| seqused_q / cu_seqlens_k / output_idx_offset / max_seqlen_q | optional vendor arguments | cann_9_1 accepts query lengths as an alternative to cumulative offsets; non-null key offsets/output offsets are rejected. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| indices | int32 Tensor | [T,1,topk] logical compressed-key positions, -1 padding. |
| values | FP32 Tensor | Optional selected scores; empty when return_value=0. |

## cann_ops_transformer

仅调用正式 CANN 9.2 绑定；旧 ABI 独立注册为 `cann_9_1`。显式指定与当前 SDK 不符的实现会在执行前报错。

## cann_9_1

CANN 9.1 原始 custom 算子；将累计/压缩长度、布局名和参数顺序转换为旧 ABI，先检查 metadata 与计算绑定均存在。

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import quant_lightning_indexer

q = torch.ones(1, 2, 128, device="npu", dtype=torch.int8)
k = torch.ones(1, 128, 1, 128, device="npu", dtype=torch.int8)
w = torch.ones(1, 2, device="npu", dtype=torch.float16)
qs = torch.ones(1, 2, device="npu", dtype=torch.float16)
ks = torch.ones(1, 128, 1, 1, device="npu", dtype=torch.float16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_k=i32([1]),
          cmp_residual_k=i32([0]), block_table=i32([[0]]),
          layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=4)
indices, values = quant_lightning_indexer(q, k, w, qs, ks, 1, 2, impl="torch_native", **kw)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/attention/test_quant_lightning_indexer.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
