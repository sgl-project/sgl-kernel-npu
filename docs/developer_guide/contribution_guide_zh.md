# sgl-kernel-npu 新算子贡献指南

本文面向为 sgl-kernel-npu 贡献新算子的开发者，介绍不同后端共用的贡献流程与接口规范。一个完整贡献包含：**实现、公开接口、正确性测试、性能数据和中英文 API 文档**。所有代码路径相对于 sgl-kernel-npu 仓库根目录；示例中的 `example_op` 和 `<module>` 需替换为实际名称。

## 开发准备

先将仓库 fork 到自己的 GitHub 账号，再克隆并创建开发分支：

```bash
git clone "https://github.com/<your_user_name>/sgl-kernel-npu.git"
cd sgl-kernel-npu
git switch -c feature/new-operator
```

按[构建安装文档](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)准备匹配的 CANN、PyTorch、torch_npu 及所需后端环境。使用当前修改构建并安装包，再执行测试，确认测试实际加载的是本次构建的代码。具体构建参数以目标分支的 `python/sgl_kernel_npu/README.md` 为准。

## 1. 明确算子需求与接口

提交实现前，说明算子的用途、对应模型或 SGLang 调用场景，并检查仓库是否已有语义相同的接口；已有接口优先新增实现。

明确输入输出、shape、dtype、布局、可选参数、原地修改行为和目标硬件架构。计算类算子应明确数学语义；量化算子还需明确缩放、偏移、舍入和饱和规则。涉及外部代码时，保留来源、许可证和必要的版权声明。

## 2. 提供实现与独立对比实现

建议文件布局（以 Triton 和 torch_native 的文件位置展示目录约定，按实际后端调整）：

```text
python/sgl_kernel_npu/sgl_kernel_npu/
├── implementation/
│   ├── triton/<module>/example_op.py
│   ├── torch_native/<module>/example_op.py
│   └── cpu_reference/<module>/example_op.py  # 可选的存量 CPU 参考实现
└── operators/<module>/
    ├── __init__.py
    └── example_op.py

tests/python/operators/<module>/test_example_op.py
docs/api/zh/operators/<module>/example_op.md
docs/api/en/operators/<module>/example_op.md
```

- kernel 和第一层 wrapper 放在 `implementation/<backend>` 中，负责输出分配、launch 配置和 kernel 调用。
- 实现特有的限制在 wrapper 中、启动 kernel 前检查；优先检查 shape、dtype、stride 等元信息，错误信息需明确限制。
- 提供语义一致、独立计算的 `torch_native` 实现作为对比后端，避免调用待测 kernel。
- 确实无法提供 torch_native 时，选择其他可信的独立参考实现，并在 PR 中说明。
- 已有 CPU 参考实现可归档到 `implementation/cpu_reference/<module>`，注册为 `impl="cpu_reference"`。计算应独立于 NPU 后端，并遵守公开接口的输入输出契约，包括原地修改及返回设备；迁移时如调整原参考语义，需明确说明。例如 `sampler.apply_token_bitmask` 将 NPU 输入复制到 CPU 计算，再回写原 logits。这类实现用于精度对照，不用于 NPU 性能测试。
- Ascend C 算子通过 operators 中的适配入口调用已暴露的原生绑定，见第 3 节；其他外部算子同样通过其公开接口接入。
- 需要管理 buffer、handle 或通信状态的接口，应先明确生命周期和模块层设计。

## 3. 注册实现并建立公开出口

**计算类算子必须提供两种及以上独立实现，用于精度对比，建议以 `torch_native` 作为对比实现。** 同一实现的不同别名（如 `auto` 和其对应后端）不计为两种实现。确因特殊情况无法满足时，需在 PR 中说明原因及替代的正确性验证方法。

在 `operators/<module>/example_op.py` 中注册实现和公开函数，公开接口采用仅限关键字参数 `impl="auto"`。默认后端同时注册为 `auto`。Python 后端函数采用首次导入、后续缓存的方式加载；Ascend C 入口直接调用已注册的原生绑定。

以下在同一个公开接口下注册 Triton、Ascend C 和 torch_native，示例默认选择 Triton，torch_native 用于精度对比。按实际提供的后端保留对应入口：

```python
import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.example_op", "auto")
@register_impl("norm.example_op", "triton")
def _triton_impl(x):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.example_op import example_op

        _triton_fn = example_op
    return _triton_fn(x)


@register_impl("norm.example_op", "ascendc")
def _ascendc_impl(x):
    return torch.ops.npu.example_op(x)


@register_impl("norm.example_op", "torch_native")
def _torch_native_impl(x):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.example_op import example_op

        _torch_native_fn = example_op
    return _torch_native_fn(x)


@register_operator("norm.example_op")
def example_op(x, *, impl="auto"):
    """auto selects Triton. Describe the operation and x's type and shape."""
    impl_fn = get_impl_by_name("norm.example_op", impl)
    return impl_fn(x)
```

本例的 `auto` 仅注册到 `_triton_impl`。如果默认选择 Ascend C，将 `auto` 装饰器移到 `_ascendc_impl`，避免对同一架构重复注册。Ascend C 示例假设包加载流程已注册 `torch.ops.npu.example_op`，命名空间、参数和返回值按实际接口调整；无需增加 `implementation/ascendc` 转发层。`import torch` 放在文件顶部，避免在每次调用时执行导入语句。

在 `operators/<module>/__init__.py` 导出公开函数。docstring 简述默认实现、功能、参数含义及类型，Tensor 标注 shape。

有架构限制时，使用 `Generation` 枚举替换上例中对应实现的注册装饰器，不要追加重复注册；函数体保持不变：

```python
from ..utils.generation import Generation


@register_impl(
    "norm.example_op", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "norm.example_op", "triton", supported_generations={Generation.ARCH22}
)
def _triton_impl(x):
    ...  # Use the same loading and invocation logic.
```

也可以使用 `excluded_generations={Generation.ARCH35}`；两种配置互斥。省略时表示支持全部已知架构，应根据实际支持范围声明。作为同一默认入口别名的 auto 与具体 impl 应保持声明一致。`auto` 可以只是默认实现的别名，不代表自动回退。

## 4. 编写公开接口测试

新增功能应提供对应测试；修复缺陷应补充能够复现问题的回归用例。

测试放在 `tests/python/operators/<module>/test_<operator>.py`，通过公开接口调用算子。编写时重点遵循以下四点：

1. **所有测试参数由 pytest 管理。** 使用 `pytest.mark.parametrize` 管理 shape、dtype、mode 和算子参数。Shape 同时包含**生产 shape**和**边缘测试 shape**：生产 shape 尽量与 SGLang 的真实调用场景一致，考虑 reshape、分片后的实际输入，并注明模型或场景；边缘 shape 根据支持范围选取最小输入、非对齐维度、分块尾部等情况。
2. **数据生成使用独立函数。** 将输入构造集中在 `generate_data` 中，使用参数化配置生成数据。现有 conftest 提供默认 NPU 设备上下文。
3. **统一测试流程。** 尽量采用 `generate_data → check_accuracy → check_performance`，精度通过后再采集性能。特殊场景采用自定义流程时，在 PR 中说明原因和测试方法。
4. **显式声明性能看护 shape。** 使用独立的 `perf_shape` 列表声明需要采集性能的 shape，并传入 `enable_perf`；可按需叠加 dtype、mode 等条件。

下面借用 RMSNorm 公开接口，展示如何组织测试，而非介绍算子的特殊功能。示例 shape 用于演示，贡献新算子时应替换为对应生产场景和边缘场景；示例中的模型维度不代表已采集的真实调用。

```python
import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [
    (1, 4096),
    (16, 7168),
]


def generate_data(shape, dtype, mode):
    # conftest.py 提供默认 NPU 设备上下文。
    torch.npu.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    bias = torch.randn(shape[-1], dtype=dtype)
    if mode == "zero":
        x.zero_()
    elif mode != "normal":
        raise ValueError(f"Unknown mode: {mode}")
    return x, weight, bias


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-width"),
        pytest.param((16, 7168), id="deepseek-v3-width"),
        pytest.param((1, 1), id="edge-minimum"),
        pytest.param((7, 1025), id="edge-tail"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias

    x, weight, bias = generate_data(shape, dtype, mode)
    executor = check_accuracy(
        rmsnorm_bias,
        args=(x, weight, bias, eps),
    )
    # 精度检查失败会抛出异常，不会继续采集性能。
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.float16
            and mode == "normal"
        ),
    )
```

`mode` 的取值应根据算子的测试目的定义，并通过注释或参数 ID 说明。示例中的 `normal` 表示随机数据，`zero` 表示零输入。其他算子的仅限关键字参数可通过 `check_accuracy(..., kwargs={...})` 传入。

参考样例（路径相对于 sgl-kernel-npu 仓库根目录）：

- RMSNorm 测试：`tests/python/operators/norm/test_rmsnorm_bias.py`。
- 公共测试组件使用说明：`tests/python/README.md`。

## 5. 验证精度、泛化性与性能

在所声明支持的硬件和软件环境中运行接口测试，先通过精度检查，再采集性能。通过独立的 `perf_shape` 列表选择代表性 shape，按需叠加 dtype、mode 条件。

```bash
# Replace the test path with the new operator's test.
python -m pytest "tests/python/operators/<module>/test_example_op.py" \
  --skip-acc-check=false --enable-perf=false -v -s -x -rs

python -m pytest "tests/python/operators/<module>/test_example_op.py" \
  --skip-acc-check=false \
  --prof-dir=results/profilings --csv-path=results/example_op.csv -v -s
```

未传 `--enable-perf` 时采用用例内配置；显式 true/false 覆盖全部用例。不传 `--csv-path` 则不写 CSV。当前通用性能工具用于串行／单 worker 场景，指标为平均 NPU task 耗时之和，而非端到端延迟。

提交性能对比时，注明基线实现、shape、dtype、硬件、软件版本和测量方式。扩大 shape、dtype 或布局支持后重新验证精度与性能；无法支持的输入应在启动 kernel 前明确拒绝。原始 profiling、构建产物等生成文件保留在本地。

## 6. 补齐中英文 API 文档

同时更新 `docs/api/{zh,en}/API_list.md`，索引仅列接口名称链接和支持的后端。每个算子详情按以下顺序组织：

| 章节 | 必填内容 |
| --- | --- |
| 芯片支持 | 一张表：行是 impl，列是 arch22、arch35 等架构，标明支持、不支持或待验证。 |
| 函数原型 | 公开导入路径、完整签名、默认参数，以及 auto 实际选择的实现。 |
| 功能说明 | 行为描述；计算类算子给出数学公式和必要的数值转换规则。 |
| 参数与返回值 | 分别用表格列出类型、描述和约束；Tensor 包含 shape、dtype、设备与布局要求，参数列出默认值。 |
| 各后端说明 | 每个后端单独使用二级标题，说明支持范围和关键差异。 |
| 调用示例 | 最小可运行示例，展示输入构造、公开接口和 impl 选择。 |

中英文内容保持一致。模块接口如有新增，同步提供模块文档；详细测试日志与性能结果放在 PR 中。

仓库内已有完整的双语示例，目录如下：

```text
docs/api/
├── zh/operators/norm/rmsnorm_bias.md
└── en/operators/norm/rmsnorm_bias.md
```

对应索引为 `docs/api/zh/API_list.md` 和 `docs/api/en/API_list.md`。新算子可参考上述文档的章节结构、参数表、架构支持表及后端说明。

## 7. 提交 PR

首次配置开发环境时安装 pre-commit 并启用 Git hook，提交前运行完整检查：

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

自动格式化后检查 diff，修复剩余问题并重新运行检查，确认全部通过。将修改提交到开发分支，推送到自己的 fork，再向上游仓库发起 PR；不要直接在 `main` 上提交。

PR 描述需包含：

- **算子用途与接口**：说明算子完成什么计算、用于哪个模型或 SGLang 场景；给出公开导入路径、函数签名和调用示例，并注明 `auto` 默认选择哪个实现。
- **正确性**：贴出测试命令、测试结果和验证环境（芯片及软件版本）。
- **性能**：新增算子需贴出性能看护用例的性能基准；优化已有算子需贴出相同测试条件下优化前后的性能对比。

提交检查清单：

- [ ] kernel / wrapper 和独立对比实现已准备。
- [ ] op、impl、auto 与架构支持声明完成，公开导出可用。
- [ ] 生产 shape、edge case、参数分支和限制检查有对应测试。
- [ ] 精度检查通过，代表性场景已采集性能。
- [ ] 中英文 API 索引与详情已更新。
- [ ] 构建与依赖配置齐全，pre-commit 通过。

可参考仓库中的 `python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py`、`tests/python/operators/norm/test_rmsnorm_bias.py` 和 `tests/python/README.md`。
