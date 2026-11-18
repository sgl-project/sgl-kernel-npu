# Ascend C 算子教程：ELU

本教程以完整算子 ELU 为例，沿当前仓库的流水线走一遍各个环节：构建声明、Host tiling、
Ascend C Kernel、生成的 ACLNN 接口、Torch adapter、Python API、测试与 API 文档。
ELU 特意保持简单，让流程本身清晰可见，而不是被 Kernel 中的数学推导掩盖。

ELU（Exponential Linear Unit，指数线性单元）逐元素计算：

$$
\mathrm{elu}(x)=\begin{cases}x, & x > 0 \\ \alpha\left(e^{x}-1\right), & x \le 0\end{cases}
$$

## 1. 完整算子的文件构成

| 路径 | 职责 |
| --- | --- |
| `csrc/examples/elu/CMakeLists.txt` | 构建声明：执行单元、源码角色、支持的架构 |
| `csrc/examples/elu/op_host/elu.cpp` | OpDef、shape/dtype 推导、tiling |
| `csrc/examples/elu/op_kernel/elu_tiling.h` | Host 与 Kernel 共享的 tiling 布局 |
| `csrc/examples/elu/op_kernel/kernel_elu.cpp` | Ascend C Kernel |
| `csrc/examples/elu/torch_adapter.cpp` | 在生成的 ACLNN 接口之上提供 `torch.ops.npu.elu` |
| `python/sgl_kernel_npu/sgl_kernel_npu/operators/examples/elu.py` | 公开的 Python API 与实现注册表 |
| `python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native/examples/elu.py` | 独立的 PyTorch 参考实现 |
| `tests/python/operators/examples/test_elu.py` | 精度与性能测试 |
| `docs/api/{en,zh}/operators/examples/elu.md` | 中英文 API 页面 |

算子目录直接放在所属 domain 目录下，不再增加额外的分类层级。`csrc/examples` 已在
`csrc/cmake/domains.cmake` 中注册为 CANN domain，因此新增 `csrc/examples/elu` 无需
修改框架代码。

## 2. 向构建系统声明算子

```cmake
# csrc/examples/elu/CMakeLists.txt
sgl_register_operator(
    ID elu
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglElu
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/elu.cpp
    INFER_SOURCES op_host/elu.cpp
    TILING_SOURCES op_host/elu.cpp
    KERNEL_SOURCES op_kernel/kernel_elu.cpp
)
```

- `EXECUTION_UNIT` 选择执行单元：`AICORE`、`AICPU` 或 `HOST`。
- `CANN_OP_TYPE` 是 GE 与生成的 ACLNN 接口使用的算子类型（`SglElu` 会生成
  `aclnnSglElu*` 和 `aclnn_sgl_elu.h`）。
- `SUPPORTED_ARCHS` 将算子限定为真正验证过的架构。`2201` 对应
  `csrc/cmake/architectures.json` 中的 arch22（A2/A3）。由于没有记录 Ascend950 的
  运行结果，因此未声明 arch35；声明但未验证的架构会声称项目无法保证的支持能力。
- `OPDEF_SOURCES`、`INFER_SOURCES` 和 `TILING_SOURCES` 可以指向同一个文件：由一份
  合并的 Host 实现同时提供 OpDef、shape 推导和 tiling 函数。
- 对于 AICORE，`KERNEL_SOURCES` 必须且只能包含一个 Kernel 入口文件。
- `TORCH_ADAPTER` 会被编译进 `libsgl_kernel_npu.so`，使 Python 可以通过
  `torch.ops.npu` 访问该算子。

同一个文件也说明了*不需要*改动哪些源码：无需修改 `include/`，也无需修改
`csrc/pytorch_extensions.cpp`。注册信息位于声明中，torch 绑定位于 `torch_adapter.cpp` 中。

## 3. Host 侧：OpDef 与 shape 推导

```cpp
namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    auto *yShape = context->GetOutputShape(0);
    if (!xShape || !yShape) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

class SglElu : public OpDef
{
public:
    explicit SglElu(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingElu);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglElu);

}  // namespace ops
```

`DataType` 列表的每个条目都是一个 dtype *组合*：第一条是 fp16，第二条是 fp32，因此
SDK 会编译两份 Kernel。`SGL_CANN_CONFIGS` 列出构建时所选架构的全部计算单元，因此
一份声明即可同时覆盖 A2 和 A3 SoC。属性 `alpha` 声明了默认值，生成的 ACLNN 接口
会在输入之后、输出之前接收它。

## 4. Host 侧：tiling

Tiling 把“这个张量在该设备上有 N 个元素”转换为具体的 Kernel 参数。ELU 需要做三个决策。

1. **Tile 长度。** 峰值时同时存活的 UB tile 有六个：双缓冲的输入队列和输出队列
   （2 + 2），加上两个单缓冲的 scratch 张量（1 + 1）。因此 tile 长度是在可用 UB 中
   能容纳六个存活 tile 的最大 512 元素倍数，上限为 4096。
2. **核数。** 每个 tile 使用一个 core，最多不超过 AIV core 的数量。
3. **任务划分。** 元素数被平均分配：每个 core 拥有 `baseLength` 个元素，前
   `extraBlocks` 个 core 各多拥有一个。每个 core 的最后一个 tile 通常短于
   `tileLength`。

```cpp
const uint32_t tileLength = ComputeTileLength(ubSize, elementSize);
int64_t tileCount = (numel + tileLength - 1) / tileLength;
uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, tileCount));
const uint32_t baseLength = static_cast<uint32_t>(numel / blockDim);
const uint32_t extraBlocks = static_cast<uint32_t>(numel % blockDim);

tiling->totalLength = static_cast<uint32_t>(numel);
tiling->tileLength = tileLength;
tiling->blockDim = blockDim;
tiling->baseLength = baseLength;
tiling->extraBlocks = extraBlocks;
tiling->alpha = *alphaAttr;

raw->SetDataSize(sizeof(SglEluTilingData));
context->SetBlockDim(blockDim);
context->SetTilingKey(0);
workspace[0] = 0;
```

Tiling 函数还会校验 Kernel 所依赖的前提：dtype 为 fp16 或 fp32，元素数为正且能放入
`uint32_t`，并且 `alpha` 为有限值。返回 `ge::GRAPH_FAILED` 会在任何内容被启动之前
拒绝该调用。

## 5. Kernel

### 5.1 共享的 tiling 布局

Host 与 Kernel 必须对该布局逐字节保持一致。把它放在双方都会包含的头文件中，并断言其
大小，这样新增字段就不会悄无声息地改变 ABI。

```cpp
namespace optiling {
struct SglEluTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t blockDim;
    uint32_t baseLength;
    uint32_t extraBlocks;
    float alpha;
};
static_assert(sizeof(SglEluTilingData) == 6 * sizeof(uint32_t), "...");
}  // namespace optiling
```

### 5.2 流水线

Kernel 类遵循标准的 CopyIn / Compute / CopyOut 结构：

- `Init` 从 tiling 数据中解析出本 core 的处理范围，将输入和输出的 global buffer
  绑定到该范围，分配两个双缓冲队列和两个 scratch buffer，并将 `alpha` 一次性转换
  为元素类型。
- `Process` 逐 tile 遍历本 core 的范围。除最后一个 tile 外，每个 tile 都是
  `tileLength` 个元素，最后一个 tile 则是剩余的所有元素。
- `CopyIn` 使用 `DataCopyPad` 把一个 tile 从 global memory 搬到 UB。
- `Compute` 执行无分支的 ELU 指令序列。
- `CopyOut` 再次使用 `DataCopyPad` 将 tile 写回。

正是 `DataCopyPad` 让“任意元素数”无需 Host 侧补齐即可工作：普通的 `DataCopy` 要求
字节数为 32 的倍数，而 `DataCopyPad` 精确拷贝 `curLength * sizeof(T)` 个字节。因此
Host 不需要对输入做补齐，输出也能保持调用者给定的精确 shape。

### 5.3 无分支计算

```cpp
/* tmpA = min(x, 0) */
AscendC::Mins(tmpA, xLocal, static_cast<T>(0), curLength);
/* tmpB = exp(min(x, 0)) */
AscendC::Exp(tmpB, tmpA, curLength);
/* out = exp(min(x, 0)) - 1 */
AscendC::Adds(outLocal, tmpB, static_cast<T>(-1), curLength);
/* tmpB = alpha * (exp(min(x, 0)) - 1) */
AscendC::Muls(tmpB, outLocal, this->alpha, curLength);
/* out = x - min(x, 0) = max(x, 0) */
AscendC::Sub(outLocal, xLocal, tmpA, curLength);
/* out = alpha * (exp(min(x, 0)) - 1) + max(x, 0) */
AscendC::Add(outLocal, outLocal, tmpB, curLength);
```

分段定义被改写为 `alpha * (exp(min(x, 0)) - 1) + (x - min(x, 0))`，该式对每个 `x`
（包括 `x == 0`）都精确成立，且从不在正值上计算 `exp`。非正参数的 `exp` 结果始终
落在 `(0, 1]` 内，因此 fp16 不会溢出。

### 5.4 Kernel 入口

```cpp
extern "C" __global__ __aicore__ void sgl_elu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglEluTilingData);
    GET_TILING_DATA(data, tiling);
    KernelElu<DTYPE_X> op;
    op.Init(x, y, data);
    op.Process();
}
```

入口名称等于 OpType 的小写下划线形式，参数顺序为输入、输出、workspace、tiling。
`DTYPE_X` 会按 OpDef 声明的每个 dtype 组合进行替换，因此一个入口即可同时服务 fp16
和 fp32。

## 6. Torch adapter

SDK 根据 OpDef 生成 ACLNN 接口。Torch adapter 负责分配输出、测量 workspace 大小，
并在当前 NPU stream 上排队启动：

```cpp
static auto query = reinterpret_cast<decltype(&aclnnSglEluGetWorkspaceSize)>(
    sgl_cann::op_api_symbol("aclnnSglEluGetWorkspaceSize"));
static auto execute = reinterpret_cast<decltype(&aclnnSglElu)>(sgl_cann::op_api_symbol("aclnnSglElu"));
const auto dtype = x.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_FLOAT16;
auto xDesc = sgl_cann::tensor_descriptor(x, dtype);
auto yDesc = sgl_cann::tensor_descriptor(y, dtype);
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
// Argument order is inputs, attributes, outputs.
auto status = query(xDesc.get(), static_cast<float>(alpha), yDesc.get(), &workspaceSize, &executor);
...
auto launch = [x, y, workspace, xDesc, yDesc, workspaceSize, executor, aclStream]() -> int {
    auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
    TORCH_CHECK(result == ACL_SUCCESS, "ELU ACLNN execution failed: ", aclGetRecentErrMsg());
    return result;
};
at_npu::native::OpCommand::RunOpApi("aclnnSglElu", launch);
```

有三个细节容易出错：

- **参数顺序**：先输入，再属性，最后输出。`alpha` 位于 `xDesc` 之后、`yDesc` 之前。
- **生命周期**：lambda 捕获 descriptor 和张量，使 host 队列在存活期内持有它们；
  `record_stream` 则为异步的设备使用保持存储存活。
- **注册**：`TORCH_LIBRARY_FRAGMENT(npu, m)` 声明
  `elu(Tensor x, float alpha=1.0) -> Tensor`，
  `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 将其绑定。无需修改任何共享的注册文件。

## 7. Python 层

公开函数位于 `operators/examples/elu.py` 中，并通过共享注册表注册：

```python
@register_impl(_IMPL_NAME, "auto", supported_generations=_A2_A3_ONLY)
@register_impl(_IMPL_NAME, "ascendc", supported_generations=_A2_A3_ONLY)
def _ascendc_impl(x, alpha):
    return torch.ops.npu.elu(x, float(alpha))


@register_impl(_IMPL_NAME, "torch_native")
def _torch_native_impl(x, alpha):
    ...


@register_operator(_IMPL_NAME)
def elu(x, alpha=1.0, *, impl="auto"):
    ...
    return get_impl_by_name(_IMPL_NAME, impl)(x, alpha)
```

- `auto` 是 `ascendc` 的别名：它解析到同一个函数，且绝不会静默回退。
- `supported_generations` 使注册表在 arch35 上拒绝 Ascend C 实现，而不是等到启动
  过程中才失败。
- 后端 import 会缓存在模块级变量中，因此重复调用不会反复导入参考实现模块。
- 签名正下方的 docstring 说明算子的功能、`auto` 会选择哪个实现，以及每个参数的
  含义、dtype 和 shape。
- 公开函数在分派前进行校验：张量类型、NPU 设备、dtype、rank、元素数、连续性、
  ND 存储格式和 `alpha`。

## 8. 测试

测试位于 `tests/python/operators/<domain>/test_<operator>.py` 下，并使用共享的
精度/性能辅助函数：

```python
perf_shape = [(1, 4096)]


def generate_data(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="npu")
    return (x,)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param((1, 4096), torch.float32, id="production-aligned-fp32"),
        pytest.param((7, 1025), torch.float16, id="unaligned-tail-fp16"),
    ],
)
def test_elu(shape, dtype):
    from sgl_kernel_npu.operators.examples import elu

    (x,) = generate_data(shape, dtype)
    executor = check_accuracy(elu, args=(x,), kwargs={"alpha": 1.0})
    check_performance(executor, enable_perf=shape in perf_shape)
```

演示性质的算子保持精简的测试集：一个生产用例和一个有意义的边界用例。这里的边界用例
是未对齐的元素数，用于覆盖 partial-tile 路径。为每组参数设置有描述性的 id，先校验
精度再检查性能，并把性能 shape 放在单独的列表中。

在装有 NPU 且已安装该包的机器上运行：

```bash
python -m pytest tests/python/operators/examples/test_elu.py -v -s
python -m pytest tests/python/operators/examples/test_elu.py --enable-perf=true -v
```

## 9. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Build the Python wheel with this operator and its CANN artifacts.
python3 setup.py --arch-version 2201 --ops elu --without-third-party

# Or use the default behavior: build every operator supported on the
# architecture, including the third-party operators configured for the CANN
# version.
python3 setup.py --arch-version 2201
```

安装 wheel，然后运行示例：

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.examples import elu

x = torch.randn(4, 4096, device="npu", dtype=torch.float16)
y = elu(x, alpha=1.0)
print(y.shape, y.dtype)
```

## 10. API 文档

新算子需要两种语言的页面，以及两处索引行：

- `docs/api/en/operators/examples/elu.md`
- `docs/api/zh/operators/examples/elu.md`
- `docs/api/en/API_list.md` 和 `docs/api/zh/API_list.md` 中各增加一行

遵循其他算子使用的页面顺序：芯片支持表、函数原型、功能说明、参数与返回值、每个
后端一个二级标题，以及一个可运行的示例。芯片支持表的单元格只能使用 **Supported**、
**Unsupported** 或 **Unverified**（**支持**、**不支持**、**未验证**）；不要把缺失的
实现描述为未验证。

## 11. 提交 PR 前的检查清单

- 算子目录直接位于其 csrc domain 下，Python、implementation、测试和 API 文档的路径
  使用相同的 domain 与算子名称。
- 构建声明只声明实际测量过的架构。
- 不为该算子修改任何共享框架文件（注册表、测试辅助函数、构建基础设施）。
- 文档用英文编写，中文页面以 `_zh` 后缀添加。
- 测试集保持精简：一个生产用例和一个边界用例，并使用描述性的 id。
- PR 描述记录算子的测量环境：设备、CANN 版本、torch/torch_npu 版本和测试结果。

## 12. 后续阅读

若要了解使用 workspace 和真实 tiling-key 路径的算子，请阅读
`csrc/sampler/apply_token_bitmask/`；若要了解针对特定架构的实现，请阅读
`csrc/attention/chunk_gated_delta_rule/`；本教程所遵循的评审规则见
`.opencode/skills/pr-review/SKILL.md`。
