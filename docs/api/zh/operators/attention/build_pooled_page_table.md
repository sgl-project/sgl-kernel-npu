# build_pooled_page_table

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 支持 | 未验证 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention import build_pooled_page_table

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

令 `K=pool_size`、输入为 `T[...,M]`：

$$O_{\ldots,p}=T_{\ldots,pK},\qquad 0\le p<\lceil M/K\rceil.$$

前导维度保持不变，`M=0` 时最后一维仍为空；不会像 `kpool_indexer_prepare_decode_input` 那样补到至少一列。输入只读，输出 dtype 与输入相同。`pool_size` 是 128 的正因子；输入列数无需整除 128。

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

公共 registry 导入时检测 NPU 架构，`torch_native` 入口也需要该环境。支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import build_pooled_page_table
page_table = torch.arange(32, device='npu', dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table(page_table, pool_size=4, impl='torch_native')
print(pooled.shape)
```
