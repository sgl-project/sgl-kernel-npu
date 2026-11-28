# move_kv_cache_rows

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

`auto` 选择 `triton`；两个后端均未限制架构注册。所有 Tensor 使用同一 NPU。

```python
from sgl_kernel_npu.operators.mem_cache import move_kv_cache_rows

def move_kv_cache_rows(keys, values, tgt_loc, src_loc, *, impl="auto"): ...
```

## 功能说明

从源快照原地移动 MHA 缓存行；即使源目标重叠也逐位保留 16 位数据。

## 参数与返回值

| 参数 | 类型、形状及约束 |
| --- | --- |
| `keys` | FP16/BF16 `[layers,pages,slots,heads,dim]`；步长均一、行内连续的 K 缓存。 |
| `values` | 布局同 `keys`；V 行宽可以不同。 |
| `tgt_loc` | 一维 int32/int64；平铺目标槽，非原地目标不可重复。 |
| `src_loc` | 一维 int32/int64；对应的平铺源槽，可重复或与目标交叠。 |
| `impl` | str：`auto`、`triton` 或 `torch_native`，默认 `auto`。 |

源和目标索引的长度必须一致。可重复原地复制作为补齐。

| 返回值 | 类型和含义 |
| --- | --- |
| `None` | 原地更新输出缓冲区。 |

## triton 后端

Triton 内核实现。

## torch_native 后端

独立 PyTorch 参考实现；已在 Ascend A3（arch22）验证。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import move_kv_cache_rows

device = "npu"
keys = torch.zeros((1, 1, 4, 1, 16), dtype=torch.float16, device=device)
values = keys.clone()
src = torch.tensor([0], dtype=torch.int32, device=device)
dst = torch.tensor([1], dtype=torch.int32, device=device)
move_kv_cache_rows(keys, values, dst, src, impl="triton")
```
