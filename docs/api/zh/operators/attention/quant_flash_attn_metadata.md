# quant_flash_attn_metadata

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformer`（`auto`） | 不支持 | 支持 |

要求 CANN 9.2.0、匹配的 PyTorch/torch_npu，以及导出 QFA 和 metadata 接口的 `cann_ops_transformer` Python 扩展。这些依赖与 `sgl_kernel_npu` wheel 分别安装。

## 函数原型

```python
from sgl_kernel_npu.operators.attention import quant_flash_attn_metadata

def quant_flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=None,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    is_grad_enabled=False,
    impl="auto",
):
    ...
```

`auto` 选择 `cann_ops_transformer`，两者调用同一个实现。该接口没有其他调度实现或自动回退路径。

## 功能说明

生成供 [quant_flash_attn](quant_flash_attn.md) 使用的设备端任务划分计划。返回值是 CANN 调度 metadata，并非序列长度副本或页表。wrapper 透传调用方已有的设备长度和属性，不推导长度、不叠加 decode／投机步数偏移，也不构造页表。

调用 QFA 前生成调度计划，相关维度和属性相同的多层可以共享。长度改变后由调用方重新生成。metadata 与主算子调用必须保持 head 数、head dim、量化模式、mask、布局和序列长度配置一致。所有输入均不修改。

## 参数与返回值

`B` 表示请求数。Tensor 参数为与主算子同设备、连续的 int32 NPU Tensor。TND Q 使用累积 Q 边界；PA_BNBD KV 使用有效 KV 长度。

| 参数 | 类型／默认值 | 描述及约束 |
| --- | --- | --- |
| `num_heads_q` | int；必选 | 本卡 Q head 数。 |
| `num_heads_kv` | int；必选 | 本卡 KV head 数，必须可整除 Q head 数。 |
| `head_dim` | int；必选 | Q/K head dim；本次集成覆盖 64/128/256。 |
| `quant_mode` | int；必选 | 本次覆盖的 MXFP8 推理路径为 `1`；其他值原样透传。 |
| `cu_seqlens_q`、`cu_seqlens_kv` | Tensor 或 None；`None` | `[B+1]` 累积长度，首元素为零。 |
| `seqused_q`、`seqused_kv` | Tensor 或 None；`None` | `[B]` 有效长度；分页 KV 使用 `seqused_kv`。 |
| `batch_size` | int 或 None；`None` | CANN 可由 Q 边界／长度 Tensor 的 shape 推导 B；TND 应保持 None。 |
| `max_seqlen_q`、`max_seqlen_kv` | int；`-1`、`-1` | 序列长度上界；`-1` 表示未指定。 |
| `head_dim_v` | int 或 None；`None` | V head dim；None 使用 `head_dim`。CANN 9.2 该路径要求 Q/K 与 V head dim 相等。 |
| `mask_mode` | int；`0` | `0`：无 mask 的 decode；`3`：右下对齐 causal。 |
| `win_left`、`win_right` | int；`-1`、`-1` | 上游窗口属性；本路径保留默认值。 |
| `layout_q`、`layout_out` | str；`"BSND"` | 本次 MXFP8 路径需显式设置为 `"TND"`。 |
| `layout_q_descale` | str；`"BSND"` | MXFP8 为 `"TND"` 或 `"N2TGD"`，必须与 QFA 的实际 scale 布局一致。 |
| `layout_kv` | str；`"BSND"` | 本次分页路径需设置为 `"PA_BNBD"`。 |
| `is_grad_enabled` | bool；`False` | CANN 的反向调度选项；推理使用 False。 |
| `impl` | str；`"auto"` | 仅限关键字参数，可选 `"auto"` 或 `"cann_ops_transformer"`。 |

| 返回值 | 类型／dtype | shape 与含义 |
| --- | --- | --- |
| `metadata` | Tensor／int32 | NPU 上的 `[2, max_schedule_size]`。容量由 CANN 根据 batch/head 配置决定；将结果原样交给 QFA。 |

## cann_ops_transformer 后端

首次调用时从 `cann_ops_transformer.ops` 导入 `quant_flash_attn_metadata` 并缓存 callable。输出分配、调度和校验由 CANN 负责。wrapper 不添加 host 读取、Tensor 转换或按 shape 回退。

本次没有实现独立的 `torch_native` 调度器：复制 CANN 调度格式不属于数值参考的范围。正确性通过 attention 消费 metadata 后与独立 attention 参考比较，并配合接口透传和返回属性检查进行验证；不将 `auto` 与 `cann_ops_transformer` 的对比算作独立精度对比。本次测试不包含动态图重放。

完整上游约束见 [CANN 9.2 API](https://gitcode.com/cann/ops-transformer/blob/v9.2.0/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md)。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import quant_flash_attn_metadata

cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
used_kv = torch.tensor([128, 257], dtype=torch.int32, device="npu")
metadata = quant_flash_attn_metadata(
    num_heads_q=4,
    num_heads_kv=2,
    head_dim=128,
    quant_mode=1,
    cu_seqlens_q=cu_q,
    seqused_kv=used_kv,
    max_seqlen_q=1,
    mask_mode=0,
    layout_q="TND",
    layout_q_descale="N2TGD",
    layout_kv="PA_BNBD",
    layout_out="TND",
    impl="cann_ops_transformer",
)
print(metadata.shape, metadata.dtype)
```

完整的 metadata 与主算子联合调用见 [quant_flash_attn 示例](quant_flash_attn.md#调用示例)。
