# clear_mamba_slots

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 支持 |
| `torch_native` | 待验证 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

def clear_mamba_slots(
    buf,
    indices,
    *,
    impl="auto",
):
    ...
```

## 功能说明

原地将 Mamba/GDN 状态池张量 `buf`（形状 `[num_layers, pool_size, ...]`）中所有层在第 1 维指定的槽位清零：

$$
\text{buf}[\ell, \text{indices}[i], :] \leftarrow 0, \quad \forall \ell \in [0, L), \; \forall i \in [0, N)
$$

Triton 实现采用持久化 1D AIV 分块调度（`BLOCK = 8192`），并使用 64 位整数偏移（`tl.int64`）计算元素地址，避免大型 SSM 状态池总元素数超过 $2^{31} - 1$ 时发生 `int32` 索引回绕。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[L, P, ...]` | `torch.bfloat16`、`torch.float16`、`torch.float32` | 必填 | NPU 上的可写状态池张量（`ndim >= 2`）；Triton 要求单个槽位切片连续（`buf[0, 0].is_contiguous()`）。原地修改。 |
| `indices` | `torch.Tensor` | `[N]` | `torch.int32`、`torch.int64` | 必填 | 待清零的一维槽位索引，取值范围 `[0, P)`，需与 `buf` 位于同一设备；Triton 要求连续存储。空张量直接返回。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地修改 `buf`，返回 `None`。 |

## triton 后端

- 单次 launch 最多启动 56 个持久化 AIV program（`BLOCK = 8192`），打满 HBM 写带宽。
- 对 `layer_stride`、`slot_stride` 与块内偏移统一使用 `tl.int64` 计算，支持总元素数超过 $2^{31}$ 的超大状态池。
- 要求 `indices` 连续且每个槽位特征切片 `buf[0, 0]` 连续。

## torch_native 后端

- 直接在输入设备上调用 `buf.index_fill_(1, indices.to(torch.int64), 0)`。
- 作为逐位精确的参考实现。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

buf = torch.randn(48, 32, 48, 128, 128, device="npu", dtype=torch.bfloat16)
indices = torch.tensor([1, 5, 12], device="npu", dtype=torch.int64)

print(list_impls(clear_mamba_slots))  # ['auto', 'torch_native', 'triton']
buf_ref = buf.clone()
clear_mamba_slots(buf, indices)  # 默认 auto，使用 Triton
clear_mamba_slots(buf_ref, indices, impl="torch_native")
torch.testing.assert_close(buf, buf_ref, rtol=0, atol=0)
```
