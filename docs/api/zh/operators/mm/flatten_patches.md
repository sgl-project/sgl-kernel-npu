# flatten_patches

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 未验证 | 未验证 |

实现使用 PyTorch 张量操作。抽取后公共 API 在两种架构上的 NPU 执行与性能均待验证。

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
`(ceil(T/TP), H/P, W/P)`。每次 reshape/permute 最多使用八维布局，避免在 NPU
执行上游十维置换。输入只读；恒等布局下输出可能与输入共享存储。

来源为 SGLang
[`932adcd205` 的 glm_vl_utils.py](https://github.com/IceShip423/sglang/blob/932adcd20532c57e59e89076e49bf71789650293/python/sglang/srt/hardware_backend/npu/modules/glm_vl_utils.py)
中的 `_npu_safe_flatten_patches`。

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

通过帧扩展、拼接及两步低维布局转换实现。本算子只做布局变换，仅有一个实现；
`auto` 是别名，不作为独立 reference。测试在 CPU 执行上游十维变换，
再将预期转至输入设备逐元素精确比较，不在 NPU 执行十维 reference。
测试提供代表性图像与视频形状的可选 profiling，本文不提供未测量的性能结果。

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
