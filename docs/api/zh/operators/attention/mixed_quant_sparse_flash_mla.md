# 混合量化共享 KV 注意力

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 不支持 | 支持 |
| `ops_transformers_cann910` | 不支持 | 支持 |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

def mixed_quant_sparse_flash_mla(q, *, quant_mode=1, rope_head_dim=64, cmp_ratio=1, softmax_scale=1.0, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

仅 A5；迁移布局为 448 个 NoPE 和 64 个 RoPE 特征，reader 与 writer 必须使用同一 SDK 选择的打包格式。

每 64 个 NoPE 值按 `K = payload × scale` 反量化，拼接未量化的 RoPE 尾部，再计算稀疏注意力。旧格式每行 640 字节、E8M0 scale；正式格式每行 608 字节、BF16 scale。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| q | Tensor | 必填 | [T,H,D] TND or [B,S,H,D] BSND; NPU in production |
| quant_mode | int / str | 1 | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| rope_head_dim | int | 64 | 64 RoPE features in the migrated mixed-quant layout |
| cmp_ratio | int | 1 | Compression ratio for compressed-KV visibility |
| softmax_scale | float | 1.0 | Multiplier on QK logits before softmax |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 关键字参数

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| ori_kv / cmp_kv | Tensor / None | Original/optional compressed paged KV [pages,page_size,1,D]; BF16 or SDK-selected packed bytes. |
| ori_block_table / cmp_block_table | int32 Tensor / None | [B,max_pages] logical-to-physical page tables. |
| cu_seqlens_q / seqused_q | int32 Tensor / None | [B+1] cumulative query offsets / [B] query lengths. |
| seqused_ori_kv / seqused_cmp_kv / cmp_residual_kv | int32 Tensor / None | [B] original/compressed/residual KV lengths; configure consistently with cmp_ratio. |
| ori_sparse_indices / cmp_sparse_indices | int32 Tensor / None | Logical Top-K positions; -1 padding. |
| ori_topk_length / cmp_topk_length | int32 Tensor / None | Per-row valid Top-K lengths; CANN 9.1 support is restricted. |
| sinks | Tensor / None | Optional per-head softmax sink logits. |
| metadata | Tensor / ReferenceMetadata / None | Build with matching metadata implementation; native bytes are not reference descriptors. |
| softmax_scale / cmp_ratio | float / int | Attention scale / compression ratio; reference defaults 1.0 / 1. |
| ori_mask_mode / cmp_mask_mode | int | Reference supports original explicit 0 or SWA 4; compressed 0 or causal 3. |
| ori_win_left / ori_win_right | int | Original visibility window; reference defaults 127 / 0. |
| layout_q / layout_kv | str | Reference defaults TND / PA_BBND; BSND queries also supported. |
| return_softmax_lse | bool | Default False; unused LSE is an empty FP32 tensor. |
| quant_mode / rope_head_dim | int / int | Mixed-quant mode 1 and 64 RoPE features in the migrated layout. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| out | Tensor | Query token/head geometry with KV value width, query activation dtype. |
| lse | FP32 Tensor | Per-query/head LSE if requested, otherwise empty. |

## cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

## ops_transformers_cann910

Source: [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/kv_quant_sparse_attn_sharedkv). The kernel source is experimental ops-transformer. Its custom_ops Python/C++ bridge may be distributed from cann-recipes-infer; the implementation name records kernel provenance, not the bridge package name. Requires window-only original-KV inputs. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## ascendc

Source: [sgl-kernel-npu DSpark device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp). The C++ wrapper directly launches the checked-in Ascend C kernel; this is not inferred from the Torch namespace. Requires explicit sparse original-KV inputs. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

q = torch.randn(1, 1, 512, device="npu", dtype=torch.bfloat16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_ori_kv=i32([1]),
          ori_block_table=i32([[0]]), layout_q="TND", layout_kv="PA_BBND",
          ori_mask_mode=4, cmp_mask_mode=3, cmp_ratio=1)
from sgl_kernel_npu.operators.mem_cache import get_npu_kv_quant_path, kv_compress_epilog
path = get_npu_kv_quant_path()  # A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
kv_compress_epilog(cache, q[:, 0], i32([0]), impl="auto")
kw.update(quant_mode=1, rope_head_dim=64)
out, lse = mixed_quant_sparse_flash_mla(q, ori_kv=cache, impl="torch_native", **kw)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
