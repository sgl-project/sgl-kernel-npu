# apply_token_bitmask

## 架构支持

代际名称沿用公共 operators 框架的 `arch22`、`arch35`，分别对应架构编号
`2201`、`3510`；构建声明使用 `SUPPORTED_ARCHS 2201 3510`。
下表区分实现声明与实测状态。

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 已实现；本次整改待上机验收 | 已实现；待上机验收 |
| `torch_native` | 已实现；NPU 待验收 | 已实现；NPU 待验收 |

## 接口

```python
from sgl_kernel_npu.operators.sampler import apply_token_bitmask

apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto")
```

`auto` 与 `ascendc` 都调用 `torch.ops.npu.apply_token_bitmask`。未注册该原生算子时，由 Torch 查找时报错，不自动回退；实现名不存在时，由公共注册工具报错。`torch_native` 必须显式选择。请安装与设备架构匹配的 wheel。安装的原生实现仍可通过 `torch.ops.npu.apply_token_bitmask` 直接调用。

## 参数与语义

| 参数 | 形状、类型与约束 |
| --- | --- |
| `logits` | 连续 NPU Tensor `[B, V]`；float32、float16 或 bfloat16；`B >= 0`，`V > 0`。 |
| `bitmask` | 同一 NPU 上连续 int32 Tensor `[B, W]`。每个整数从最低位起控制32个词元。宽度不足时缺少的位视为0，多余列忽略。 |
| `indices` | 可选一维 Tensor，CPU 或同一 NPU；按旧实现转换为 int64 后索引行。推荐 int32/int64，索引必须有效。`None` 和空 Tensor 都表示全部行。 |
| `impl` | `auto`、`ascendc`、`torch_native`；默认 `auto`。 |

bit=0 的位置写成负无穷，bit=1 的位置保留原值；只修改选中的行。函数**原地修改 `logits` 并返回同一存储的 Tensor**；`bitmask` 和 `indices` 不变。普通 C++ Tiling 保留九字段，Kernel 按256元素对齐，适配层负责补齐和裁剪，不要求调用者改变 `V`。

底层 Kernel 使用 uint32 索引；工作张量维度、实际访问偏移和分块计算必须在其可表示范围内。超出范围时报错，不截断输入。此推理算子没有新增自动求导实现，也未宣称图捕获或编译模式已通过验收。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sampler import apply_token_bitmask

logits = torch.randn(4, 320, device="npu", dtype=torch.float16)
mask = torch.full((4, 10), -1, device="npu", dtype=torch.int32)
expected = apply_token_bitmask(logits.clone(), mask, impl="torch_native")
result = apply_token_bitmask(logits, mask)  # auto -> ascendc；原地修改
assert result.data_ptr() == logits.data_ptr()
torch.testing.assert_close(result, expected, atol=0, rtol=0)
```

完整 Python 接口会进入每个 REFACTOR wheel，但原生实现只包含本次 `--ops` 选择的已迁移算子。不同子集 wheel 使用同一发行包身份，安装后相互替换，不会叠加；完整清单见安装包内 `_build_manifest.json`。
