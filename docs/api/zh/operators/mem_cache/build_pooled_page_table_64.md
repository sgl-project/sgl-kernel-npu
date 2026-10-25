# build_pooled_page_table_64

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Unverified | Unverified |

## 函数原型

`auto` 是 `torch_native` 的别名；省略 `impl` 与传入 `impl="auto"` 效果相同。

```python
from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

def build_pooled_page_table_64(
    page_table_64,
    pool_size,
    *,
    impl="auto",
):
    ...
```

## 功能说明

选取每个逻辑 pool 分组中的第一个 token 页表项。若输入最后一维有 `NUM_BLOCKS` 项，输出最后一维对应索引 `0, pool_size, 2 * pool_size, ...`。计算等价于在输入设备上构造 `torch.arange(0, NUM_BLOCKS, pool_size)` 并做高级索引。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `page_table_64` | `torch.Tensor` | `[..., NUM_BLOCKS]` | 后端不限制；应按整数页 ID 类型使用 | 必填 | token 页表，对最后一维抽样。 |
| `pool_size` | `int` | — | — | 必填 | 一个 pool 页表示的 token 页数；必须为正数，且整除后端常量 `BLOCK_SIZE_K = 128`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., ceil(NUM_BLOCKS / pool_size)]` | 与 `page_table_64` 相同 | 新建结果，不修改输入；调用方若有额外连续性要求，需自行处理。 |

## torch_native 后端

- 使用 `torch.arange` 和张量索引，整除要求通过断言检查。
- `auto` 只是别名，不是动态选择或 fallback。导入公共 registry 时会执行 ACL 架构检测，并要求 NPU 运行时可用。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

page_table = torch.arange(32, device="npu", dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table_64(page_table, pool_size=4)

print(pooled.shape)  # torch.Size([2, 4])
print(list_impls(build_pooled_page_table_64))  # ['auto', 'torch_native']
```
