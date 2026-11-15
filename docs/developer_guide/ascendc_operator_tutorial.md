# Ascend C operator tutorial: ELU

This tutorial follows one complete operator, ELU, through the current repository
pipeline: build declaration, Host tiling, Ascend C Kernel, the generated ACLNN
interface, the Torch adapter, the Python API, tests and API documentation. ELU is
deliberately simple, so the workflow stays visible instead of being hidden behind
kernel math.

ELU (Exponential Linear Unit) is element-wise:

$$
\mathrm{elu}(x)=\begin{cases}x, & x > 0 \\ \alpha\left(e^{x}-1\right), & x \le 0\end{cases}
$$

## 1. Files of a complete operator

| Path | Role |
| --- | --- |
| `csrc/examples/elu/CMakeLists.txt` | Build declaration: execution unit, source roles, supported architectures |
| `csrc/examples/elu/op_host/elu.cpp` | OpDef, shape/dtype inference, tiling |
| `csrc/examples/elu/op_kernel/elu_tiling.h` | Tiling layout shared by Host and Kernel |
| `csrc/examples/elu/op_kernel/kernel_elu.cpp` | Ascend C Kernel |
| `csrc/examples/elu/torch_adapter.cpp` | `torch.ops.npu.elu` over the generated ACLNN interface |
| `python/sgl_kernel_npu/sgl_kernel_npu/operators/examples/elu.py` | Public Python API and implementation registry |
| `python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native/examples/elu.py` | Independent PyTorch reference |
| `tests/python/operators/examples/test_elu.py` | Accuracy and performance test |
| `docs/api/{en,zh}/operators/examples/elu.md` | Bilingual API pages |

Place operator directories directly under their domain directory, without another
classification level. `csrc/examples` is already registered as a CANN domain in
`csrc/cmake/domains.cmake`, so adding `csrc/examples/elu` needs no framework edit.

## 2. Declare the operator to the build

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

- `EXECUTION_UNIT` selects the execution unit: `AICORE`, `AICPU` or `HOST`.
- `CANN_OP_TYPE` is the operator type used by GE and by the generated ACLNN
  interface (`SglElu` produces `aclnnSglElu*` and `aclnn_sgl_elu.h`).
- `SUPPORTED_ARCHS` limits the operator to the architectures that were really
  validated. `2201` is arch22 (A2/A3) from
  `csrc/cmake/architectures.json`. arch35 is not declared because no Ascend950 run
  has been recorded; a declared-but-unverified architecture would claim support
  the project cannot back.
- `OPDEF_SOURCES`, `INFER_SOURCES` and `TILING_SOURCES` can name the same file: a
  combined Host implementation that provides the OpDef, shape inference and the
  tiling function.
- `KERNEL_SOURCES` must contain exactly one Kernel entry file for AICORE.
- `TORCH_ADAPTER` is compiled into `libsgl_kernel_npu.so` so Python can reach the
  operator through `torch.ops.npu`.

The same file also shows the sources that are *not* here: no `include/` change and
no edit to `csrc/pytorch_extensions.cpp`. Registration lives in the declaration,
and the torch binding lives in `torch_adapter.cpp`.

## 3. Host side: OpDef and shape inference

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

Each `DataType` list entry is a dtype *combination*: entry one is fp16 and entry
two is fp32, so the SDK compiles the Kernel twice. `SGL_CANN_CONFIGS` lists every
compute unit of the architecture selected at build time, so one declaration
covers both A2 and A3 SoCs. The attribute `alpha` is declared with a default, and
the generated ACLNN interface receives it after the inputs and before the
outputs.

## 4. Host side: tiling

Tiling turns "this tensor has N elements on this device" into concrete Kernel
parameters. ELU has three decisions to make.

1. **Tile length.** Six tiles of UB are live at peak: the double-buffered input
   and output queues (2 + 2) plus two single-buffered scratch tensors (1 + 1).
   The tile length is therefore the largest 512-element multiple that fits six
   live tiles in the usable UB, capped at 4096.
2. **Core count.** One core per tile, up to the number of AIV cores.
3. **Work split.** The element count is divided evenly: every core owns
   `baseLength` elements and the first `extraBlocks` cores own one more. The last
   tile of a core is generally shorter than `tileLength`.

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

The tiling function also validates what the Kernel assumes: the dtype is fp16 or
fp32, the element count is positive and fits `uint32_t`, and `alpha` is finite.
Returning `ge::GRAPH_FAILED` rejects the call before anything is launched.

## 5. The Kernel

### 5.1 Shared tiling layout

Host and Kernel must agree on the layout byte for byte. Put it in a header both
sides include, and assert the size so a field addition cannot silently change
the ABI.

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

### 5.2 The pipeline

The Kernel class follows the standard CopyIn / Compute / CopyOut shape:

- `Init` resolves the core's own range from the tiling data, binds the input and
  output global buffers to it, allocates two double-buffered queues and two
  scratch buffers, and casts `alpha` to the element type once.
- `Process` walks the core's range tile by tile. A tile is `tileLength` elements
  except for the last one, which is whatever remains.
- `CopyIn` moves one tile from global memory to UB with `DataCopyPad`.
- `Compute` runs the branch-free ELU sequence.
- `CopyOut` writes the tile back, again with `DataCopyPad`.

`DataCopyPad` is what makes "any element count" work without host-side padding:
plain `DataCopy` requires a 32-byte multiple, while `DataCopyPad` copies exactly
`curLength * sizeof(T)` bytes. That is why the Host does not need to pad the
input, and the output keeps the caller's exact shape.

### 5.3 Branch-free compute

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

The piecewise definition is rewritten as
`alpha * (exp(min(x, 0)) - 1) + (x - min(x, 0))`, which is exact for every `x`
(including `x == 0`) and never evaluates `exp` on a positive value. `exp` of a
non-positive argument stays in `(0, 1]`, so fp16 cannot overflow.

### 5.4 Kernel entry

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

The entry name equals the OpType in lower snake case, and the argument order is
inputs, outputs, workspace, tiling. `DTYPE_X` is substituted per dtype
combination declared by the OpDef, which is why one entry serves both fp16 and
fp32.

## 6. Torch adapter

The SDK generates the ACLNN interface from the OpDef. The adapter allocates the
output, measures the workspace, and queues the launch on the current NPU stream:

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

Three details are easy to get wrong:

- **Argument order**: inputs, then attributes, then outputs. `alpha` comes after
  `xDesc` and before `yDesc`.
- **Lifetimes**: the lambda captures the descriptors and tensors so the host queue
  keeps them alive, and `record_stream` keeps storage alive for asynchronous
  device use.
- **Registration**: `TORCH_LIBRARY_FRAGMENT(npu, m)` declares
  `elu(Tensor x, float alpha=1.0) -> Tensor` and
  `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` binds it. No shared registration file
  needs editing.

## 7. Python layer

The public function lives in `operators/examples/elu.py` and is registered with
the shared registry:

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

- `auto` is an alias of `ascendc`: it resolves to the same function and never
  falls back silently.
- `supported_generations` makes the registry refuse the Ascend C implementation
  on arch35 instead of failing later inside the launch.
- The backend import is cached in a module-level variable so repeated calls do not
  re-import the reference module.
- The docstring directly under the signature states what the operator does, which
  implementation `auto` selects, and the meaning, dtype and shape of every
  argument.
- The public function validates before dispatch: tensor type, NPU device, dtype,
  rank, element count, contiguity, ND storage format and `alpha`.

## 8. Tests

Tests live under `tests/python/operators/<domain>/test_<operator>.py` and use the
shared accuracy/performance helpers:

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

A demonstration operator keeps its test suite small: one production case and one
meaningful edge case. Here the edge case is an unaligned element count, which
exercises the partial-tile path. Give every parameter set a descriptive id, check
accuracy before performance, and keep performance shapes in a separate list.

Run it on a machine with an NPU and the package installed:

```bash
python -m pytest tests/python/operators/examples/test_elu.py -v -s
python -m pytest tests/python/operators/examples/test_elu.py --enable-perf=true -v
```

## 9. Build and run

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Build the Python wheel with this operator and its CANN artifacts.
python3 setup.py --arch-version 2201 --ops elu --without-third-party

# Or use the default behavior: build every operator supported on the
# architecture, including the third-party operators configured for the CANN
# version.
python3 setup.py --arch-version 2201
```

Install the wheel, then run the example:

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.examples import elu

x = torch.randn(4, 4096, device="npu", dtype=torch.float16)
y = elu(x, alpha=1.0)
print(y.shape, y.dtype)
```

## 10. API documentation

A new operator needs both language pages plus both index rows:

- `docs/api/en/operators/examples/elu.md`
- `docs/api/zh/operators/examples/elu.md`
- a row in `docs/api/en/API_list.md` and `docs/api/zh/API_list.md`

Follow the page order used by the other operators: hardware support table,
signature, description, parameters and return value, one second-level heading per
backend, and a runnable example. Hardware-support cells use only **Supported**,
**Unsupported** or **Unverified** (**支持**, **不支持**, **未验证**); do not
describe a missing implementation as an unverified one.

## 11. Checklist before opening a PR

- The operator directory is directly under its csrc domain, and the Python,
  implementation, test and API-doc paths use the same domain and operator names.
- The build declaration declares only architectures that were measured.
- No shared framework file (registry, test helpers, build infrastructure) is
  modified for the operator.
- Documentation is written in English; a Chinese page is added as `_zh`.
- The test suite stays small: one production case and one edge case, with
  descriptive ids.
- The PR description records where the operator was measured: device, CANN
  version, torch/torch_npu versions and the test result.

## 12. Where to go next

Read `csrc/sampler/apply_token_bitmask/` for an operator that uses a workspace
and a real tiling-key path, `csrc/attention/chunk_gated_delta_rule/` for
architecture-specific implementations, and `.opencode/skills/pr-review/SKILL.md`
for the review rules this tutorial is written against.
