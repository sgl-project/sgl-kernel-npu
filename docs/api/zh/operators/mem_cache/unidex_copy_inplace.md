# unidex_copy_inplace

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 不支持 |

## 函数签名

`auto` 选择 `ascendc`。

```python
from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace


def unidex_copy_inplace(
    src, dst, src_index, dst_index, valid_mask, src_address_ndims, dst_address_ndims,
    block_dim=8, src_ptr=None, dst_ptr=None, *, impl="auto"
): ...
```

## 功能描述

按索引在 KV 缓冲区之间复制逻辑行。掩码非零且源、目标索引均在范围内时，将源行复制到对应目标行。前若干维度展平为行索引，其余维度共同决定每行复制的字节数。源、目标每行字节数必须相等，且最多为 32 KiB。并发执行的有效映射应使用不同目标行。

## 参数与返回值

所有张量必须连续。索引和掩码张量位于同一 NPU。没有地址覆盖参数的端点也须在该 NPU 上。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `src` | Tensor | 必填 | 源缓冲区，dtype 和每行字节数与 `dst` 相同。 |
| `dst` | Tensor | 必填 | 目标缓冲区，原地写入。 |
| `src_index` | Tensor | 必填 | INT64 `[mappings]` 源行索引。 |
| `dst_index` | Tensor | 必填 | INT64 `[mappings]` 目标行索引。 |
| `valid_mask` | Tensor | 必填 | Bool 或 UINT8 `[mappings]` 复制掩码。 |
| `src_address_ndims` | int | 必填 | 参与源行寻址的前导维数，大于 0 且小于源张量秩。 |
| `dst_address_ndims` | int | 必填 | 参与目标行寻址的前导维数，大于 0 且小于目标张量秩。 |
| `block_dim` | int | `8` | AscendC 使用的向量核数，须为正数。 |
| `src_ptr` | int 或 None | `None` | 可选的 NPU 可见源地址，由 AscendC 使用。 |
| `dst_ptr` | int 或 None | `None` | 可选的 NPU 可见目标地址，由 AscendC 使用。 |
| `impl` | str | `"auto"` | 使用的实现。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 原 `dst` 张量。映射为空时保留原数据。 |

## ascendc 后端

- 直接在 NPU 缓冲区之间或 NPU 与已注册的主机内存之间按字节复制。
- 调用者须保证覆盖地址在当前流执行完成前有效。可通过 `sgl_kernel_npu.operators.mem_cache` 中的 `create_shm_tensor` 和 `free_shm` 管理注册的共享内存。

## torch_native 后端

- 用 PyTorch 索引复制有效行，读写地址来自源和目标张量。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace

src = torch.randn(4, 32, dtype=torch.float16, device="npu")
dst = torch.zeros_like(src)
src_index = torch.tensor([3, 0], dtype=torch.int64, device="npu")
dst_index = torch.tensor([0, 2], dtype=torch.int64, device="npu")
mask = torch.tensor([True, False], device="npu")
result = unidex_copy_inplace(src, dst, src_index, dst_index, mask, 1, 1, impl="ascendc")
assert result is dst
```
