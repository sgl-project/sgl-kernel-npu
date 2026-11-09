# adler32_strided_checksum

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

两种实现均已在 Ascend A3（arch22）和 Ascend A5（arch35）上验证。

## 函数签名

```python
import torch
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

def adler32_strided_checksum(
    data_ptrs: list[int] | list[torch.Tensor],
    strides: list[int],
    indices: list[torch.Tensor],
    *,
    impl="auto",
) -> int:
    ...
```

`auto` 选择 `triton`。不提供自动回退。

## 功能说明

为 PD KV cache 传输计算 Adler-32。按输入顺序拼接描述符；对每个描述符，
按照索引顺序读取 `data_ptr + indices[j] * stride` 起始的 `stride` 个字节。
重复索引会重复计入对应字节。存储 dtype 不改变按原始字节计算的语义。

对拼接后的字节流 `x[0:N]`，令 `M = 65521`：

- `a = (1 + sum(x[j])) % M`
- `b = (N + sum((N - j) * x[j])) % M`
- `checksum = (b << 16) | a`

空字节流返回 1，不修改输入。只将归约后的部分和复制到主机；返回 Python 整数
时会同步该结果。checksum 相同并不能证明模型计算本身正确。

## 参数与返回值

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `data_ptrs` | `list[torch.Tensor]` 或 `list[int]` | 必填 | 每个描述符对应连续的 NPU Tensor。`auto`/`triton` 也支持整数地址；非空且 stride 非零的描述符要求有效正地址。`torch_native` 要求传入 Tensor。 |
| `strides` | `list[int]` | 必填 | 非负的单项字节长度，即每个索引选中的连续字节范围。 |
| `indices` | `list[torch.Tensor]` | 必填 | 每个描述符对应连续的一维 `[num_items]` 索引，int32 或 int64，全部位于同一 NPU。 |
| `impl` | `str` | `"auto"` | 仅关键字参数；支持 `auto`、`triton`、`torch_native`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| 校验值 | `int` | 以 Python 整数表示的无符号 32 位 Adler-32。 |

三个列表的长度必须相同。调用方负责保持底层内存有效，保证索引非负且不越界、
连续存储容量充足，并在算子读取之前完成写入或建立正确的执行顺序。
原始地址不包含内存边界或同步信息。

## triton 后端

按字节读取选中的行，计算模 65521 的加权部分和，再归约得到 checksum。
描述符和索引的顺序语义与 SGLang CUDA 实现一致。本次迁移保持原有计算公式
和启动配置。

## torch_native 后端

使用普通 PyTorch 从 Tensor 缓冲区读取所选原始字节，再用 CPU
`zlib.adler32` 计算独立参考值，不调用 Triton 算子。两个后端可以接收同一组
Tensor 输入，供统一精度测试比较。

`auto`/`triton` 继续支持整数地址，以兼容 SGLang 现有调用；参考实现要求
传入所属 Tensor，拒绝整数地址。CPU 复制使该后端适用于正确性检查，不适合
生产 KV 传输；不会尝试从无所有权的原始地址重建设备存储。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

data = torch.arange(32, dtype=torch.uint8, device="npu").reshape(4, 8)
indices = torch.tensor([2, 0], dtype=torch.int64, device="npu")
value = adler32_strided_checksum(
    [data], [data[0].nbytes], [indices], impl="triton"
)
reference = adler32_strided_checksum(
    [data], [data[0].nbytes], [indices], impl="torch_native"
)
assert value == reference
print(value)
```
