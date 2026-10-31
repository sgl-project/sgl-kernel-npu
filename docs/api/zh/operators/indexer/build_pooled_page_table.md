# build_pooled_page_table

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.indexer import build_pooled_page_table

def build_pooled_page_table(
    page_table_64: torch.Tensor,
    pool_size: int,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` 固定选择 `torch_native`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

按 `pool_size` 指定的间隔，从 token 页表中选取每组的第一个页 ID，生成 pooled 页表。

例如，`pool_size=4` 时，一行 `[10, 11, 12, 13, 20, 21]` 得到 `[10, 20]`。保留输入的前导维度和数据类型，不修改输入。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `page_table_64` | `torch.Tensor` | `[..., NUM_BLOCKS]` | 后端不限制；应按整数页 ID 类型使用 | 必填 | token 页表，对最后一维抽样。 |
| `pool_size` | `int` | — | — | 必填 | 一个 pool 页表示的 token 页数；必须为正数，且整除后端常量 `BLOCK_SIZE_K = 128`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., ceil(NUM_BLOCKS / pool_size)]` | 与 `page_table_64` 相同 | 新建结果，不修改输入；调用方若有额外连续性要求，需自行处理。 |

输入输出位于同一 NPU，输入不被修改。

## torch_native 后端

使用输入设备上的 torch.arange 和高级索引，不要求输入连续，输入须有最后一维。仅注册 auto、torch_native，未实现 triton。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import build_pooled_page_table
page_table = torch.arange(32, device='npu', dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table(page_table, pool_size=4, impl='torch_native')
print(pooled.shape)
```
