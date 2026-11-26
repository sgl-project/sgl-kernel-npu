# move_mla_kv_cache

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

`auto` 选择 `triton`；两个后端均未限制架构注册。所有 Tensor 使用同一 NPU。

```python
from sgl_kernel_npu.operators.mem_cache import move_mla_kv_cache

def move_mla_kv_cache(k_cache, rope_cache, tgt_loc, src_loc, page_size, use_fia_nz=False, *, impl="auto"): ...
```

## 功能说明

从快照原地移动 MLA latent 和 RoPE 行；NZ 使用物理布局 [layer,page,dim/16,slot,16]。

## 参数与返回值

| 参数 | 类型、形状及约束 |
| --- | --- |
| `k_cache` | FP16/BF16 `[layers,pages,slots,1,dim]`；latent 缓存。 |
| `rope_cache` | 布局同 `k_cache`；RoPE 行宽可以不同。 |
| `tgt_loc` | int32/int64；平铺目标槽，非原地目标不可重复。 |
| `src_loc` | int32/int64；对应的平铺源槽，可重复或与目标交叠。 |
| `page_size` | int；每页逻辑槽数。 |
| `use_fia_nz` | bool（默认 `False`）；启用时两个缓存采用连续 NZ 存储，行宽须为 16 的倍数。 |
| `impl` | str：`auto`、`triton` 或 `torch_native`，默认 `auto`。 |

源和目标索引的元素数必须一致。可重复原地复制作为补齐；NZ 布局的补齐槽不能同时是实际搬运目标。

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
from sgl_kernel_npu.operators.mem_cache import move_mla_kv_cache

device = "npu"
k = torch.zeros((1, 1, 4, 1, 16), dtype=torch.float16, device=device)
r = k.clone()
src = torch.tensor([0], dtype=torch.int32, device=device)
dst = torch.tensor([1], dtype=torch.int32, device=device)
move_mla_kv_cache(k, r, dst, src, 4, impl="triton")
```
