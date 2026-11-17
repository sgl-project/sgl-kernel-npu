# 量化索引器元数据

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `ops_transformers_cann910` | 支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

def quant_lightning_indexer_metadata(num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

与 quant_lightning_indexer 使用相同实现和长度；不同 ABI 的元数据不可混用。

根据头维度、Top-K、量化模式和请求长度生成调度信息。参考元数据是参数描述对象，不作为 tiling 字节精度基准。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| num_heads_q | int | 必填 | Number of query heads |
| num_heads_k | int | 必填 | Number of key heads; reference supports 1 |
| head_dim | int | 必填 | Head width D |
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
| return_value | int | Default 0 produces empty values; nonzero score output is 不支持 on CANN 9.1. |
| seqused_q / cu_seqlens_k / output_idx_offset / max_seqlen_q | optional vendor arguments | CANN 9.1 accepts query lengths as an alternative to cumulative offsets; non-null key offsets/output offsets are rejected. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| metadata | Native Tensor or ReferenceMetadata | Opaque native scheduling storage, or reference-only descriptor. Never compare bytes across implementations. |

## cann_ops_transformers_cann920

仅调用正式 CANN 9.2 绑定；旧 ABI 独立注册为 `CANN 9.1`。显式指定与当前 SDK 不符的实现会在执行前报错。

## ops_transformers_cann910

Source: [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer_metadata). The kernel source is experimental ops-transformer. Its custom_ops Python/C++ bridge may be distributed from cann-recipes-infer; the implementation name records kernel provenance, not the bridge package name. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

q = torch.ones(1, 2, 128, device="npu", dtype=torch.int8)
k = torch.ones(1, 128, 1, 128, device="npu", dtype=torch.int8)
w = torch.ones(1, 2, device="npu", dtype=torch.float16)
qs = torch.ones(1, 2, device="npu", dtype=torch.float16)
ks = torch.ones(1, 128, 1, 1, device="npu", dtype=torch.float16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_k=i32([1]),
          cmp_residual_k=i32([0]), block_table=i32([[0]]),
          layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=4)
metadata = quant_lightning_indexer_metadata(2, 1, 128, 1, 2, impl="torch_native", **kw)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

精度测试在 CPU 上执行独立 eager 参考，并在 CPU 上排序、精确比较索引集合，隔离参考 Top-K 与比较 Sort 对 NPU 执行路径的影响。性能计时仍使用原始 NPU executor。

```bash
python3 -m pytest tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
