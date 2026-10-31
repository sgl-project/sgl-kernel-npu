# flatten_patches

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 未验证 | 未验证 |

## 函数签名

```python
from sgl_kernel_npu.operators.mm import flatten_patches

def flatten_patches(patches, patch_size, merge_size, temporal_patch_size, *, impl="auto"):
    ...
```

`auto` 选择 `torch_native`。

## 功能说明

将 GLM 图像或视频像素转换为展平 patch。重复最后一帧，直到帧数能被
`temporal_patch_size` 整除。token 依次按时间网格、合并块行、合并块列、
块内行、块内列排序；特征依次按通道、时间 patch 内帧、像素行、像素列排序。

记 `P=patch_size`、`M=merge_size`、`TP=temporal_patch_size`，网格为
`(ceil(T/TP), H/P, W/P)`。每次 reshape/permute 最多使用八维布局，满足 NPU
置换操作的维度约束。输入只读；恒等布局下输出可能与输入共享存储。

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
from sgl_kernel_npu.operators.mm import flatten_patches

pixels = torch.randn((1, 3, 3, 28, 28), device="npu", dtype=torch.bfloat16)
patches, grid_t, grid_h, grid_w = flatten_patches(
    pixels, patch_size=14, merge_size=2, temporal_patch_size=2, impl="torch_native"
)
assert patches.shape == (1, 8, 1176)
assert (grid_t, grid_h, grid_w) == (2, 2, 2)
```
