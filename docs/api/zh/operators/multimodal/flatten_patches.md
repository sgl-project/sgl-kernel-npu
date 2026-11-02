# flatten_patches

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 未验证 | 已验证 |

## 函数签名

```python
from sgl_kernel_npu.operators.multimodal import flatten_patches

def flatten_patches(patches, patch_size, merge_size, temporal_patch_size, *, impl="auto"):
    ...
```

`auto` 选择 `torch_native`。

## 功能说明

将图像或视频像素整理为模型输入所需的展平 patch。帧数不足以组成完整时间 patch 时重复最后一帧，再按指定的 merge size 对空间 patch 分组。

返回展平特征及时间、高度、宽度三个方向的网格大小。布局转换保留所需的 patch 和特征顺序，过程中张量最多为八维。输入不被修改；恒等布局下输出可能与输入共享存储。

## 参数

| 参数 | 类型 | 说明与约束 |
| --- | --- | --- |
| `patches` | Tensor `[B,T,C,H,W]` | NPU 像素张量；测试用例包含 FP32、FP16、BF16。各维必须为正数，支持非连续输入。 |
| `patch_size` | int | 正数空间 patch 大小 `P`；必填。 |
| `merge_size` | int | 正数合并倍率 `M`；`H`、`W` 必须能被 `P*M` 整除；必填。 |
| `temporal_patch_size` | int | 正数时间 patch 帧数 `TP`；必填。 |
| `impl` | str | 默认 `auto`，也可指定 `torch_native`。 |

## 返回值

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `flatten_patches` | Tensor `[B,grid_t*grid_h*grid_w,C*TP*P*P]` | dtype、device 与输入一致。 |
| `grid_t` | int | 补齐后的帧数除以 `TP`。 |
| `grid_h` | int | `H/P`。 |
| `grid_w` | int | `W/P`。 |

## torch_native

通过帧扩展、拼接及两步最多八维的布局转换实现。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.multimodal import flatten_patches

pixels = torch.randn((1, 3, 3, 28, 28), device="npu", dtype=torch.bfloat16)
patches, grid_t, grid_h, grid_w = flatten_patches(
    pixels, patch_size=14, merge_size=2, temporal_patch_size=2, impl="torch_native"
)
assert patches.shape == (1, 8, 1176)
assert (grid_t, grid_h, grid_w) == (2, 2, 2)
```
