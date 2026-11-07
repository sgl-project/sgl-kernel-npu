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
Kernel 复用 `csrc/utils/kernel/device_prims` 的设备基础接口和 `csrc/utils/kernel/math_utils.h` 的数学辅助函数。

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

在仓库根目录运行 `python3 setup.py --arch-version 2201 --ops all` 构建。
当前构建工具会校验 arch22 两个 CANN 目标的产物，因此先构建该架构完整的迁移算子包，
再在目标 A2/A3 设备上单独执行本算子测试。
测试新 ACLNN 路径前，安装生成的 wheel：
`pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl`。
精度测试：
`python -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py --enable-perf=false -v -rs`。
不传 `--enable-perf` 时，仅对选定的 BF16 MLA shape 采集性能。
用例根据 SGLang MLA 调用布局设置，并标注模型维度、代表性 token 负载，未宣称为运行时采集记录。

## 模型 shape 测试

分析基于本地 SGLang 提交 `2456c9111a121a02e41e625e3930e1eb143fe9bf`，
模型参数于 2026-11-05 核对。测试只构造张量并直接调用当前算子，不启动模型，
不覆盖 NPU Graph、权重加载、attention backend 或完整推理调用链。

调用点位于
[`deepseek_v2_attention_mla_npu.py`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py#L290)：

- 普通 MLA 的 `forward_mla_core_npu`：只有 `M >= 65536`、`B*K >= 65536`
  或 `N >= 65536` 时调用本算子。对于下表的普通 MLA 模型和当前 `M <= 1024`
  的算子限制，只有有效 attention TP=1 的情况满足条件；TP=2/4/8 时走
  `torch_npu.npu_transpose_batchmatmul`。
- DSA 的 `forward_dsa_core_npu`：在 A2/A3 的 decode、target verify 和
  draft extend v2 分支调用本算子，普通 extend 使用 `torch_npu`。
  因此可以覆盖 TP=1/2/4/8。模型识别和分支选择分别见
  [`is_deepseek_dsa`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/configs/model_config.py#L187)
  和 [`handle_attention_ascend`](https://github.com/sgl-project/sglang/blob/43af9fcbb68d2d34a5549635dfcc7b46360e6fc6/python/sglang/srt/models/deepseek_common/attention_backend_handler.py#L75)。

模型相关 shape 为 `(M, B, K, N)`，其中
`B = num_attention_heads / effective_attention_tp`、`K = kv_lora_rank`、
`N = v_head_dim`。attention TP 取调用时生效的值，可能与整个模型的 TP 不同。
模型用例仅保留 DeepSeek 和 GLM 系列。同系列的相同 shape 仅保留一两个较新的模型作为代表。
DeepSeek 保留 V3.1/V3.2，
分别代表普通 MLA 和 DSA；GLM 保留 5.2/5.3-Flash。

| Shape ID 前缀 / 模型配置来源 | 总 head 数 | K | N | 测试 attention TP | 调用分支 |
| --- | --- | --- | --- | --- | --- |
| [`deepseek-v3.1`](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json) | 128 | 512 | 128 | 1 | 普通 MLA |
| [`deepseek-v3.2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) | 128 | 512 | 128 | 1/2/4/8 | DSA |
| [`glm-5.2`](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) | 64 | 512 | 256 | 1/2/4/8 | DSA |
| [`glm-5.3-flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) | 64 | 512 | 256 | 1/2/4/8 | DSA 层 |

GLM-5.3-Flash 读取 `text_config` 的维度，仅其 DSA 层复用这条路径。
这些结论用于选择单算子输入，不表示已经验证相应模型在 A2/A3 上的完整可运行性。

`M >= 65536` 的调用分支超过本算子原有 1024 限制，
本次未将其作为有效输入，也未修改这个限制。

每个模型/TP 组合测试 `M = 1, 8, 36, 128, 1024`。
M 是展平后的输入 token 数，verify 场景可能包含多个候选 token；这些值是代表性负载，
不是模型配置规定的 batch size 或实际采集的请求。DeepSeek-V3.2、GLM-5.2 还在
attention TP=8 下测试 M=15/16/17，覆盖真实维度下的 tile 边界。
每个 shape 都测试 FP16/BF16、随机输入/零输入，并检查输出写入及输入不被修改。
不需要模型权重；即使原 checkpoint 使用量化，本次也只测试算子接受的浮点输入。

例如 `glm-5.2-attn-tp8-tokens36` 对应 A=`[36,8,512]`、W=`[8,512,256]`、
C=`[36,8,256]`。主仓两个调用点均使用默认 ND 和默认 quant 参数。
NZ 的模型前缀用例仅扩展直接 API 的格式覆盖，不表示主仓以 NZ 调用它。
非法输入测试仍标记为 `extra_checks`，默认不执行。

为目标 A2/A3 设备构建并加载新包后，运行全部直接算子测试：

```bash
python -m pytest tests/python/operators/matmul/test_batch_matmul_transpose.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -rs
```

追加 `-k 'glm-5'` 可以筛选 GLM 模型用例；追加 `-k 'deepseek'`
可以筛选 DeepSeek 用例。这里只筛选单算子 shape，不启动对应模型。
