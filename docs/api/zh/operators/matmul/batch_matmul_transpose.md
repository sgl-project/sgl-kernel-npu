# batch_matmul_transpose

## 芯片支持

| 实现 | arch22（A2/A3，2201） | arch35（A5，3510） |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_npu` | 受限支持 | 支持 |
| `torch_native` | 支持 | 待验证 |

AscendC 内核支持 Ascend A2/A3（arch22，2201）。

## 函数原型

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(tensor_a, tensor_b, *, impl="auto"):
    ...
```

输入采用 ND 布局，函数返回形状为 `[M, B, N]` 的新张量。

MLA 使用 `auto`。在 arch22 上，当 `M`、`B * K` 或 `N` 中任一值大于等于
65536 时选择 `ascendc`，其余形状选择 `torch_npu`。在 arch35 上，`auto` 选择
`torch_npu`。在所有前向模式中，DSA 的 Q 和 V 投影均显式传入
`impl="torch_npu"`。

## 功能说明

计算 token 轴和 batch 轴交错布局的批量矩阵乘：

$$
C_{m,b,n} = \sum_{k=0}^{K-1} A_{m,b,k} W_{b,k,n}.
$$

其中 M 为 token 数，B 为独立矩阵或头数，K 为归约维度，N 为输出特征维度。
`ascendc` 和 `torch_native` 以 FP32 累加，将结果转换为输入 dtype。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `tensor_a` | Tensor | `[M, B, K]` | FP16 / BF16 | 必填 | `ascendc` 和 `torch_native` 要求 `1 <= M <= 1024`。 |
| `tensor_b` | Tensor | `[B, K, N]` | 与 A 相同 | 必填 | `ascendc` 和 `torch_native` 要求权重连续。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字。支持 `auto`、`ascendc`、`torch_npu`、`torch_native`。 |

| 返回值 | 类型 | Shape | Dtype | 副作用 |
| --- | --- | --- | --- | --- |
| 结果 | Tensor | `[M, B, N]` | 与 A 相同 | 新分配的结果张量。 |

`ascendc` 和 `torch_native` 要求各维度为正，每个张量的元素数不超过 `UINT32_MAX`。
`ascendc` 会按需将 A 转为连续张量。两个输入的 dtype 和设备必须相同。`ascendc` 要求
NPU，公开包需要正常的 NPU 和原生运行时环境。

## ascendc 后端

Torch adapter 在当前 NPU stream 上启动包内的 ACLNN 算子。
CANN Tiling 回调根据输入维度和设备属性选择分块大小与核调度方式。
CANN 管理 tiling 数据，Kernel 根据这些数据执行批量矩阵乘。

## torch_npu 后端

该实现调用 `torch_npu.npu_transpose_batchmatmul` 并返回其输出。两个输入均采用 ND 布局，
形状分别为 `[M, B, K]` 和 `[B, K, N]`。默认转置参数为 `perm_x1=(1, 0, 2)`、
`perm_x2=(0, 1, 2)` 和 `perm_y=(1, 0, 2)`。

在 arch22（A2/A3）上，CANN 9.1 规定了 `1 <= B <= 65535` 和 `1 <= N <= 65535`。
对 `[M, B, K]` 输入执行融合转置时还要求 `B * K <= 65535`。第一个输入采用
`[B, M, K]` 布局时，对应的输入步长约束为 `K <= 65535`。
[CANN 9.1 TransposeBatchMatMul 参考文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnTransposeBatchMatMul.md)

在 arch22 上，当 `B * K >= 65536` 时，该后端会先将第一个输入转置为连续的
`[B, M, K]`，再以 `perm_x1=(0, 1, 2)` 调用 TorchNPU。`perm_y=(1, 0, 2)`
使输出形状保持为 `[M, B, N]`。上述 `[B, M, K]` 布局对 `B`、`K` 和 `N`
的限制仍然适用。

## torch_native 后端

该实现使用 PyTorch `bmm` 对 FP32 输入进行计算，再将结果转换为输入 dtype。
累加顺序不同可能造成数值差异。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

a = torch.randn(3, 2, 32, device="npu", dtype=torch.float16)
b = torch.randn(2, 32, 48, device="npu", dtype=a.dtype)
output = batch_matmul_transpose(a, b, impl="ascendc")
reference = batch_matmul_transpose(a, b, impl="torch_native")
torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)
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
