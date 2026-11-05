# chain_speculative_sampling

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 已注册；重构版本待实机验证 | 已注册；重构版本待实机验证 |
| `torch_native` | 已注册；重构版本待实机验证 | 已注册；重构版本待实机验证 |

`auto` 选择 `triton`，不进行失败回退。注册代表支持声明，不代表设备精度或
性能已验证。本次迁移保留原 Triton 计算和启动配置，仍需与迁移前版本进行
NPU 精度和性能对照。

## 接口签名

```python
from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

def chain_speculative_sampling(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在非贪心推测解码中，目标模型完成验证前向、得到概率后，本算子验证一条
**线性草稿链**。SGLang 的 NPU EAGLE/NEXTN 拒绝采样和稠密 DSpark 接受路径
使用它。logits 处理、概率归一化、随机数生成和 KV 缓存更新由调用方完成。

`B` 是实际验证请求数，`D` 是包含第 0 列根节点的候选宽度，`R` 是草稿概率
行数，`V` 是完整词表宽度。依次检查草稿 `x = candidates[b, j]`，其中
`j = 1, ..., D-1`，接受条件为
`uniform_samples[b, j-1] * draft_probs[b, j-1, x] < target_probs[b, j-1, x]`。
第一次拒绝即停止。若共接受 `a` 个草稿，使用第 `a` 行补采一个最终 token：

- 全部接受时，权重 `w = target_probs[b, a]`。
- 发生拒绝时，权重 `w = max(target_probs[b, a] - draft_probs[b, a], 0)`。

选择累计权重严格大于 `uniform_samples_for_final_sampling[b] * sum(w)` 的
第一个 token。无需对残差再次归一化。保留已有累计分布末尾回退行为：
Triton 取所选词块内最后一个正权重位置，块内没有正权重时取 `V-1`；
PyTorch 参考在整行中查找最后一个正权重位置。

## 参数与返回值

参与计算的张量须连续、位于同一 NPU。除明确标注的三只输出缓冲外，输入只读。
非空批要求 `D >= 1`、`R >= max(D-1, 1)`。调用方须提供非空词表及有效的
token/槽号，wrapper 不检查索引范围或概率值。保留旧拼写 `retrive` 以兼容
关键字调用。

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，通常 `[B*D]`，int32 | 输出：在 `retrive_index` 指定的槽位写入接受的草稿和最终 token。 |
| `accept_index` | `[B,D]`，int32 | 输出：根槽位及接受槽位；有效前缀长度为 `accept_token_num[b] + 1`。 |
| `accept_token_num` | `[B]`，int32 | 输出：接受的草稿数，不含最终补采 token；SGLang 加 1 得到本轮推进长度。 |
| `candidates` | `[B,D]`，int64 | 根及草稿 token ID；根 token 本身不参与验证或采样。 |
| `retrive_index` | `[B,D]`，int64 | 有效且互不冲突的全局输出槽位，须落在 `predicts` 范围内且能用 int32 输出索引表示。 |
| `retrive_next_token`、`retrive_next_sibling` | 忽略 | 兼容参数，不启用树遍历。 |
| `uniform_samples` | `[B,D]`，float32 | 调用方提供的 `[0,1)` 接受随机数；最后一列不使用。 |
| `uniform_samples_for_final_sampling` | `[B]`，float32 | 调用方提供的 `[0,1)` 最终采样随机数。 |
| `target_probs` | `[B,D,V]`，float32 | 调用方完成采样变换后的目标概率。 |
| `draft_probs` | `[B,R,V]`，float32 | 同一词表上的草稿概率，允许额外行。 |
| `threshold_single`、`threshold_acc` | float，默认 `1.0` | 忽略的兼容参数，不改变接受条件。 |
| `deterministic` | bool，默认 `True` | 忽略的兼容参数；算子内部不生成随机数。 |
| `impl` | str，默认 `"auto"` | 可选 `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `None`。调用方分配并初始化三只输出缓冲，未写入的槽保留原值；
输出不得与输入或其他输出重叠；允许三只输出使用同一分配中互不重叠的视图。`B == 0` 时，Triton 仅检查候选和目标
张量的维数，随后直接返回，不执行其他参数检查。

## 后端实现

`triton` 顺序执行三个 kernel：接受前缀、按 2048 个词分块求残差权重和、
补采最终 token。临时元数据为 `[B,3]` int64，词块和为
`[B,ceil(V/2048)]` float32。保留原 shape、dtype、连续性和同设备检查。

`torch_native` 复用仓库旧测试的 `chain_rejection_reference` 计算主体，适配
完整公开签名，不重复 Triton 的输入校验。它用 Python float 进行标量接受
判断，并用 PyTorch 整行 `sum`/`cumsum`；Triton 使用 FP32 运算和分块归约。
接近决策边界时，舍入可能改变离散 token 选择。整数输出仍要求精确相等，
出现分歧需分析原因。

旧 `sgl_kernel_npu.sample` 下的算子实现、包导出和原测试保持原样。
现有 `chain_speculative_sampling_triton` 导入继续执行旧目录中的实现副本。
本次新架构保留独立的实现副本，新增接入统一使用 `operators`。

## 测试覆盖

统一测试为两个后端分配独立输出缓冲，用公共精度工具比较完整的三只 int32
输出，再调用公共性能工具。两组原测试完整保留，并以模型 pytest ID 覆盖以下
部署推导的 FP32 场景：

| 模型 | `(D,R,V)` | `B` |
| --- | --- | --- |
| GLM-4.7-Flash | `(4,3,154880)` | 1、2、4、8、16、32 |
| Qwen3.5-9B | `(4,3,248320)` | 1、2、4、8、16 |
| DeepSeek-V4-Flash | `(6,5,129280)`<br>`(7,6,129280)` | 1、2、4、8<br>1、2、4、6、8、10、12 |

执行参数沿用[测试框架说明](../../../../../tests/python/README.md)。
