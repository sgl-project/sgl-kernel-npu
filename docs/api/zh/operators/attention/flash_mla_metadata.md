# flash_mla_metadata

该算子是 CANN 9.2.0 版本引入的非正式算子。

上游原始文档：[Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/37c98efe33ee7faf7d17882532dc1305221c727e/attention/flash_mla/docs/torchapi_flash_mla.md)。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | 不支持 | 支持 |

## 函数原型

默认 `auto` 在 CANN 9.2.0、arch35 上选择 `cann_ops_transformers_cann920`。

```python
from sgl_kernel_npu.operators.attention import flash_mla_metadata

def flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    *,
    impl="auto",
):
    ...
```

## 功能说明

为 [flash_mla](flash_mla.md) 生成调度信息。将返回的 tensor 直接传给主算子的 `metadata` 参数。

序列长度、头数、维度、掩码、窗口及 Query 布局必须与主算子一致。这些参数或使用的设备变化后，需要重新生成 metadata。

## 参数与返回值

所有 tensor 输入必须位于当前 NPU。`B` 为请求数，`T` 为打包后的物理 Query token 总数。

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `cache_seqlens` | Tensor `[B]`；int32 | 必填 | 每个请求的有效 KV 长度，与主算子及其缓存容量一致。 |
| `num_heads_q` | `int` | 必填 | 本地 Query 头数，范围 `[1,128]`，与 `q` 一致。 |
| `num_heads_kv` | `int` | 必填 | 当前 MLA 内核要求为 1。 |
| `cu_seqlens_q` | 可选 Tensor `[B+1]`；int32 | `None` | TND 必填，其他布局不传；物理 Query 累计长度，首项为零、末项为 `T`。 |
| `seqused_q` | 可选 Tensor `[B]`；int32 | `None` | 实际 Query 长度，不超过对应物理长度；省略时使用完整分段。 |
| `max_seqlen_q`、`max_seqlen_kv` | `int` | `-1` | 上游最大长度属性；`-1` 使用上游默认值，否则为非负值。 |
| `head_dim_qk` | `int` | `576` | 必须为 576，包含 Key 的全部通道。 |
| `head_dim_v` | `int` | `512` | 必须为 512。 |
| `mask_mode` | `int` | `0` | `0`：无掩码；`3`：右对齐因果；`4`：右对齐滑动窗口。 |
| `win_left`、`win_right` | `int` | `-1` | 模式 4 的窗口范围，`-1` 表示无界；其他模式必须为 `-1`。 |
| `layout_q` | `str` | `"BSND"` | `BSND`、`BNSD` 或 `TND`，与主算子一致。 |
| `impl` | `str`，仅关键字传入 | `"auto"` | `auto` 或 `cann_ops_transformers_cann920`。 |

| 返回值 | 类型 / Shape / Dtype | 描述 |
| --- | --- | --- |
| `metadata` | Tensor `[M]`；当前 NPU 上的 int32 | 调度信息，长度由接口根据输入和设备自动确定；不修改输入。 |

将返回值原样传入 `flash_mla(..., metadata=metadata)`。

## cann_ops_transformers_cann920 后端

需要 CANN 9.2.0，并安装包含 Flash MLA 和 metadata 算子的 sgl-kernel-npu wheel。首次调用可能编译适配代码，请保留 C++ 编译环境。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_mla_metadata

lengths = torch.tensor([128], device="npu", dtype=torch.int32)
boundaries = torch.tensor([0, 1], device="npu", dtype=torch.int32)
metadata = flash_mla_metadata(
    lengths,
    num_heads_q=64,
    num_heads_kv=1,
    cu_seqlens_q=boundaries,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    layout_q="TND",
    impl="cann_ops_transformers_cann920",
)
print(metadata.shape, metadata.dtype)
# 保持原样传给长度和属性一致的 flash_mla。
```
