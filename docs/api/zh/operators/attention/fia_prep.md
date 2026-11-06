# fia_prep

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention import fia_prep

def fia_prep(ti, seq_lens, per_query_req, req_to_token, block_size, block_table_out=None, actual_kvlen_out=None, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

保留其他选中块的顺序，将当前块置于末尾，通过 token 槽整除块长得到物理页，并计算因果长度。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `ti` | torch.Tensor | `[T,S]; int32` | 必填 | 逻辑页号，-1 填充；须为当前页保留容量。 |
| `seq_lens` | torch.Tensor | `[T] or [B]; int32` | 必填 | 每个查询或请求的因果 KV 长度。 |
| `per_query_req` | torch.Tensor | `[T]; int32/int64` | 必填 | 每个查询对应的请求索引。 |
| `req_to_token` | torch.Tensor | `[requests,max_ctx]; int32/int64` | 必填 | token 槽映射；与 ti 位于同一 NPU。 |
| `block_size` | int | `—` | 必填 | 每个 KV 页的 token 数；须为正。 |
| `block_table_out` | torch.Tensor 或 None | `[T,S]; int32` | `None` | 可选输出缓冲区，会被覆盖并返回。 |
| `actual_kvlen_out` | torch.Tensor 或 None | `[T]; int32` | `None` | 可选因果长度输出缓冲区。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `block_table` | torch.Tensor 或 None | `[T,S]; int32` | 重排后的物理页；可复用 block_table_out。 |
| `actual_kvlen` | torch.Tensor 或 None | `[T]; int32` | 其他完整页长度加选中当前页的因果前缀长度。 |

## triton 后端

当前启动要求 S 为二次幂。未选择当前页时须留出一个输出槽；请求和页索引必须有效。

## torch_native 后端

通过独立 PyTorch 运算生成参考结果；主机元数据传输可能触发同步，仅用于精度对比。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention import fia_prep

ti = torch.tensor([[0, 1, -1, -1]], device="npu", dtype=torch.int32)
lengths = torch.tensor([17], device="npu", dtype=torch.int32)
req = torch.tensor([0], device="npu", dtype=torch.int32)
slots = torch.arange(64, device="npu", dtype=torch.int32).reshape(1, 64)
table, kvlen = fia_prep(ti, lengths, req, slots, 16, impl="triton")
```
