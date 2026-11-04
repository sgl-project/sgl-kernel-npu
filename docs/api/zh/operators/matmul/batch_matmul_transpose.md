# batch_matmul_transpose

## 芯片支持

| 实现 | arch22（A2/A3，2201） | arch35（A5，3510） |
| --- | --- | --- |
| `ascendc` | 仅支持 910C | 不支持 |
| `torch_native` | 支持 | 支持 |

AscendC 算子仅支持 910C（`ascend910_93`），不支持其他 arch22 硬件平台。

## 函数原型

```python
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose

def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None, *, impl="auto"
):
    ...
```

`auto` 选择 `ascendc`，不自动回退。函数写入 `tensor_c`，**返回 None**，
保留旧 `torch.ops.npu.batch_matmul_transpose` 的接口语义。旧构建和调用路径继续保留。

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
| `quant_mode` | str / None | — | — | `None` | 支持 `per_channel_symm`、`per_channel_asymm`、`per_token_symm`；None 等价于第一项。保留旧选择参数，**不启用量化**。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字；支持 `auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | Shape | Dtype | 副作用 |
| --- | --- | --- | --- | --- |
| None | NoneType | — | — | 写入 C，不修改 A、B。 |

所有维度必须为正，每个张量元素数不超过 `UINT32_MAX`，各张量 dtype、设备必须相同。
`ascendc` 要求 NPU；直接加载 torch-native 实现也可用于 CPU 数学验证，
公开包仍需要正常的 NPU 和原生运行时环境。不支持的布局、dtype、shape 或参数名会报错。

## ascendc 后端

沿用原 arch22 Cube Kernel、Tiling 成本模型、swizzle 选择和打包结构。
Host 调度接入 CANN Tiling 回调，沿用原 `HardwareInfo()` 构造函数，
通过 `PlatformInfo::Instance()` 获取硬件信息。
Tiling 存储由 CANN 管理，不再使用旧的进程级 1024 槽位张量及同步 H2D 拷贝；
接口仍保留 `M <= 1024` 限制。Torch adapter 使用包内 ACLNN 和当前 NPU stream。
Kernel 的共享辅助代码仍来自 `csrc/mla_preprocess` 和 `csrc/utils`。

当前 SGLang MLA 调用点在调用 Torch 算子前，已经按指定 dtype 分配 `[M, B, N]` 输出。
本次接入采用 ACLNN 单算子路径，不注册手写 shape/dtype 推导。
保留原有私有文件布局：`op_host/batch_matmul_transpose.cpp` 放置 OpDef 和 CANN Tiling 回调；
`op_host/tiling/tiling_data.h` 保存声明和打包数据结构，`tiling_data.cpp` 实现调度。
调度直接复用 `csrc/utils/common_tiling.h`，不向新算子复制公共辅助函数。
共享调度接口沿用原写法，仅在 Torch 构建中启用 Torch 校验。
两个 Host 源文件均参与 codegen 和 Tiling 编译。
Kernel 保留原文件名 `op_kernel/batch_matmul_transpose_kernel.cpp`。
CANN 入口与注册和迁移后的包含路径需要适配，计算与调度公式保持一致。
ACLNN 调用与图模式推导注册的区别见[昇腾算子原型定义说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/operator_prototype_definition.html)。

## torch_native 后端

使用独立的 PyTorch FP32 `bmm`，显式解包 NZ，转换 dtype 后复制到 C。
不调用自定义算子或其他实现后端。累加顺序不同可能造成数值差异。

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

在仓库根目录运行 `python3 setup.py --arch-version 2201 --ops batch_matmul_transpose` 构建。
测试新 ACLNN 路径前，安装生成的 wheel：
`pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl`。
精度测试：
`python -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py --enable-perf=false -v -rs`。
不传 `--enable-perf` 时，仅对选定的 BF16 MLA shape 采集性能。
用例根据 SGLang MLA 调用布局设置，并标注模型维度、代表性 token 负载，未宣称为运行时采集记录。
