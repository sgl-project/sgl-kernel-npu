# kv_quant_sparse_flash_attention_torch_npu

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-transformer`。

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention_torch_npu


def kv_quant_sparse_flash_attention_torch_npu(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

对 sparse_indices 选出的位置执行带尺度的注意力，K/V 使用 SDK 量化编码。计算仅覆盖反量化后的选中位置。MLA 模式中 query 含旋转尾部，而结果宽度是潜变量维度。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Sq, Hq, D + Dr], BF16 在示例中 | 必传 | MLA 的潜变量与旋转 query 拼接。 |
| `key`, `value` | Tensor | paged [P, L, Hkv, ...], INT8 在示例中 | 必传 | 分开的 K 和 V 参数。示例将 INT8 键、BF16 旋转字节和 FP32 尺度打包在 key 中。 |
| `sparse_indices` | Tensor | [B, Sq, Hkv, K], INT32 | 必传 | 各 query 与 KV 头选中的逻辑 KV 位置。 |
| `scale_value` | float | - | 必传 | 作用于注意力 logits 的尺度。 |
| `key_quant_mode`, `value_quant_mode` | int | - | 必传 | 厂商编码模式，INT8 示例均为 2。 |
| `block_table`, `actual_seq_lengths_query`, `actual_seq_lengths_kv` | Tensor or None | [B, max_pages], [B], [B], INT32 | `None` | 可选分页元数据，示例中必传。 |
| `layout_query`, `layout_kv`, `attention_mode` | str, str, int | - | `"BSND"` / `0` | 可选 SDK 设置。示例采用 BSND、PA_BSND 和 MLA 模式 2。 |
| `rope_head_dim`, `tile_size`, `quant_scale_repo_mode` | int | - | `64` / `128` / `1` | 可选旋转宽度、量化块宽度和尺度存储模式，须与 key 编码方式一致。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | [B, Sq, Hq, D] with BF16 在示例中 | 输入 NPU 上的注意力结果。MLA 的 D 不含旋转尾部。 |

## ops-transformer 后端

调用 `torch_npu.npu_kv_quant_sparse_flash_attention`。形状、dtype 和布局组合以已安装的 SDK 为准。 调用带独立 key、value 和量化模式参数的厂商接口。仓内的 packed FP8 API 使用自身的缓存布局和签名。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_npu_kv_quant_sparse_flash_attention.py) [CANN ops-transformer 源码](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/kv_quant_sparse_flash_attention)

## torch_native 后端

解包每 tile 使用 INT8 或 E4M3 量化并在记录中保存 scale 的 KV 缓存，计算 TND 或 BSND query 与 PA_BSND KV 的稀疏 MLA。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention_torch_npu

b, pages, page, h, d, dr, tile = 4, 128, 256, 128, 512, 64, 128
q = torch.randn((b, 1, h, d + dr), dtype=torch.bfloat16, device="npu")
value = torch.zeros((pages, page, 1, d), dtype=torch.int8, device=q.device)
key_rope = torch.zeros((pages, page, 1, dr), dtype=q.dtype, device=q.device)
scales = torch.ones((pages, page, 1, d // tile), dtype=torch.float32, device=q.device)
key = torch.cat((value, key_rope.view(torch.int8), scales.view(torch.int8)), dim=-1)
positions = torch.arange(2048, dtype=torch.int32, device=q.device).view(1, 1, 1, -1)
positions = positions.expand(b, 1, 1, -1).contiguous()
table = torch.arange(pages, dtype=torch.int32, device=q.device).reshape(b, -1)
q_lens = torch.ones((b,), dtype=torch.int32, device=q.device)
kv_lens = torch.full((b,), 4096, dtype=torch.int32, device=q.device)
out = kv_quant_sparse_flash_attention_torch_npu(
    q,
    key,
    value,
    positions,
    (d + dr) ** -0.5,
    2,
    2,
    block_table=table,
    actual_seq_lengths_query=q_lens,
    actual_seq_lengths_kv=kv_lens,
    layout_query="BSND",
    layout_kv="PA_BSND",
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile,
    rope_head_dim=dr,
    impl="ops-transformer",
)
```
