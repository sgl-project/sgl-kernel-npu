# 混合量化注意力元数据

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | 不支持 | 已声明，需验证 SDK 内核 |
| `torch_native` | 不支持 | 已声明，需验证 SDK 内核 |

## 接口

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

def mixed_quant_sparse_flash_mla_metadata(*, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

仅 A5；打包格式和实现须与 mixed_quant_sparse_flash_mla 一致。

生成混合量化注意力调度信息；显式稀疏可见性对应的 vendor 窗口约定由适配器转换。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 关键字参数

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| ori_kv / cmp_kv | Tensor / None | Original/optional compressed paged KV [pages,page_size,1,D]; BF16 or SDK-selected packed bytes. |
| ori_block_table / cmp_block_table | int32 Tensor / None | [B,max_pages] logical-to-physical page tables. |
| cu_seqlens_q / seqused_q | int32 Tensor / None | [B+1] cumulative query offsets / [B] query lengths. |
| seqused_ori_kv / seqused_cmp_kv / cmp_residual_kv | int32 Tensor / None | [B] original/compressed/residual KV lengths; configure consistently with cmp_ratio. |
| ori_sparse_indices / cmp_sparse_indices | int32 Tensor / None | Logical Top-K positions; -1 padding. |
| ori_topk_length / cmp_topk_length | int32 Tensor / None | Per-row valid Top-K lengths; cann_9_1 support is restricted. |
| sinks | Tensor / None | Optional per-head softmax sink logits. |
| metadata | Tensor / ReferenceMetadata / None | Build with matching metadata implementation; native bytes are not reference descriptors. |
| softmax_scale / cmp_ratio | float / int | Attention scale / compression ratio; reference defaults 1.0 / 1. |
| ori_mask_mode / cmp_mask_mode | int | Reference supports original explicit 0 or SWA 4; compressed 0 or causal 3. |
| ori_win_left / ori_win_right | int | Original visibility window; reference defaults 127 / 0. |
| layout_q / layout_kv | str | Reference defaults TND / PA_BBND; BSND queries also supported. |
| return_softmax_lse | bool | Default False; unused LSE is an empty FP32 tensor. |
| quant_mode / rope_head_dim | int / int | Mixed-quant mode 1 and 64 RoPE features in the migrated layout. |
| num_heads_q / num_heads_k / head_dim / batch_size / max_seqlen_q | int | Native scheduling geometry; use the same values as paired compute. |
| has_ori_kv / has_cmp_kv | bool | Whether original/compressed streams participate. |
| cu_seqlens_q_cpu / seqused_ori_kv_cpu | CPU int32 Tensor / None | Existing host mirrors required by cann_9_1 explicit sparse metadata; no device-to-host copy is introduced. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| metadata | Native Tensor or ReferenceMetadata | Opaque native scheduling storage, or reference-only descriptor. Never compare bytes across implementations. |

## cann_ops_transformer

保留现有注册名；包装器内部按 SDK 选择原始或正式绑定。metadata/计算配套检查和 A5 KV reader/writer 格式检查在分配或执行前完成。

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

q = torch.randn(1, 1, 512, device="npu", dtype=torch.bfloat16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_ori_kv=i32([1]),
          ori_block_table=i32([[0]]), layout_q="TND", layout_kv="PA_BBND",
          ori_mask_mode=4, cmp_mask_mode=3, cmp_ratio=1)
from sgl_kernel_npu.operators.kvcacheio import get_npu_kv_quant_path, kv_compress_epilog
path = get_npu_kv_quant_path()  # A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
kv_compress_epilog(cache, q[:, 0], i32([0]), impl="auto")
kw.update(quant_mode=1, rope_head_dim=64)
metadata = mixed_quant_sparse_flash_mla_metadata(impl="torch_native", **kw)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
