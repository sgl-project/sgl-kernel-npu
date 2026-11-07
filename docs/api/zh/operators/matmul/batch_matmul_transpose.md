# batch_matmul_transpose

## 芯片支持

| 实现 | arch22（A2/A3，2201） | arch35（A5，3510） |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 待验证 |

AscendC 内核支持 Ascend A2/A3（arch22，2201）。

## 函数原型

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None, *, impl="auto"
):
    ...
```

`auto` 选择 `ascendc`，不自动回退。函数将结果写入调用方分配的 `tensor_c`，并返回 `None`。

## 功能说明

计算 token 轴和 batch 轴交错布局的批量矩阵乘：

$$
C_{m,b,n} = \sum_{k=0}^{K-1} A_{m,b,k} W_{b,k,n}.
$$

其中 M 为 token 数，B 为独立矩阵或头数，K 为归约维度，N 为输出特征维度。
两个实现均以 FP32 累加，将结果转换为输入 dtype。

NZ 权重使用显式四维打包：$W^{NZ}_{b,j,k,r}=W_{b,k,16j+r}$。
传入的是具有该物理排列的 ND 张量，不是要求转换 `torch_npu` 的不透明存储格式。
可通过 `w.reshape(B, K, N // 16, 16).permute(0, 2, 1, 3).contiguous()` 打包。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `tensor_a` | Tensor | `[M, B, K]` | FP16 / BF16 | 必填 | 连续；`1 <= M <= 1024`。 |
| `tensor_b` | Tensor | ND：`[B, K, N]`；NZ：`[B, N/16, K, 16]` | 与 A 相同 | 必填 | 连续；NZ 要求 K、N 为 16 的倍数。 |
| `tensor_c` | Tensor | `[M, B, N]` | 与 A 相同 | 必填 | 调用方分配的连续输出，原地覆盖；不能与 A、B 重叠。 |
| `format_mode` | str / None | — | — | `None` | None 等价于 `"ND"`；支持 `"ND"`、`"NZ"`。 |
| `quant_mode` | str / None | — | — | `None` | 支持 `per_channel_symm`、`per_channel_asymm`、`per_token_symm`。`None` 等价于 `per_channel_symm`。所有取值均执行浮点计算，不启用量化。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字；支持 `auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | Shape | Dtype | 副作用 |
| --- | --- | --- | --- | --- |
| None | NoneType | — | — | 写入 C，不修改 A、B。 |

所有维度必须为正，每个张量元素数不超过 `UINT32_MAX`，各张量 dtype、设备必须相同。
`ascendc` 要求 NPU；直接加载 torch-native 实现也可用于 CPU 数学验证，
公开包仍需要正常的 NPU 和原生运行时环境。不支持的布局、dtype、shape 或参数名会报错。

## ascendc 后端

Torch adapter 在当前 NPU stream 上启动包内的 ACLNN 算子。
CANN Tiling 回调根据输入维度和设备属性选择分块大小与核调度方式。
CANN 管理 tiling 数据，Kernel 根据这些数据执行批量矩阵乘。

## torch_native 后端

该实现会解包 NZ 权重，使用 PyTorch `bmm` 对 FP32 输入进行计算，再将结果转换为输出 dtype 并复制到 `tensor_c`。
累加顺序不同可能造成数值差异。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

a = torch.randn(3, 2, 32, device="npu", dtype=torch.float16)
b = torch.randn(2, 32, 48, device="npu", dtype=a.dtype)
c = torch.empty(3, 2, 48, device="npu", dtype=a.dtype)
assert batch_matmul_transpose(a, b, c, impl="ascendc") is None
reference = torch.empty_like(c)
batch_matmul_transpose(a, b, reference, impl="torch_native")
torch.testing.assert_close(c, reference, atol=1e-3, rtol=1e-3)
```

## 构建与测试

在仓库根目录构建并安装单算子 wheel，然后在 Ascend A2/A3 设备上运行算子测试。

```bash
python3 setup.py --make-clean --arch-version 2201 --ops batch_matmul_transpose \
  --artifact wheel --without-third-party -j 8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl

python3 -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -rs
```
