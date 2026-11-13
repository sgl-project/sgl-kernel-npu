# store_kv_cache_prefix_valid

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mem_cache import store_kv_cache_prefix_valid


def store_kv_cache_prefix_valid(
    dst_k, dst_v, src_k, src_v, loc_2d, commit_lens, *, impl="auto"
): ...
```

## 功能描述

只把每个请求接受的前缀从打包 K/V 源行写入展平缓存槽。K 和 V 的头维度可不同；不访问未接受位置对应的槽号，其余缓存行保持不变。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dst_k` | Tensor | `Required` | 可写缓存 [slots,heads,Kdim]，head/dim 轴连续。 |
| `dst_v` | Tensor | `Required` | 可写缓存 [slots,heads,Vdim]，与 dst_k 同类型。 |
| `src_k` | Tensor | `Required` | 打包 key [batch*width,heads,Kdim]，与缓存同类型。 |
| `src_v` | Tensor | `Required` | 打包 value [batch*width,heads,Vdim]，与缓存同类型。 |
| `loc_2d` | Tensor | `Required` | 整数目标槽 [batch,width]，已接受位置必须有效，未接受位置可用哨兵。 |
| `commit_lens` | Tensor | `Required` | 整数接受前缀长度 [batch]，范围 [0,width]。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | None | 原地更新选定 dst_k/dst_v 行。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import store_kv_cache_prefix_valid

dst_k = torch.zeros(16, 2, 64, device="npu", dtype=torch.bfloat16)
dst_v = torch.zeros_like(dst_k)
src_k = torch.randn(4, 2, 64, device="npu", dtype=dst_k.dtype)
src_v = torch.randn_like(src_k)
locations = torch.tensor([[3, 4, -1, -1]], device="npu", dtype=torch.int64)
lengths = torch.tensor([2], device="npu", dtype=torch.int32)
store_kv_cache_prefix_valid(
    dst_k, dst_v, src_k, src_v, locations, lengths, impl="triton"
)
```
