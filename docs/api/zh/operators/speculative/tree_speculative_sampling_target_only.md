# tree_speculative_sampling_target_only

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 已注册，重构版本待实机验证 | 已注册，重构版本待实机验证 |
| `torch_native` | 已注册，重构版本待实机验证 | 已注册，重构版本待实机验证 |

`auto` 选择 `triton`，实现解析或执行错误直接向调用方报告。当前登记的是支持
范围，NPU 精度和性能仍需在对应设备上与旧实现对照验收。

## 接口签名

```python
from sgl_kernel_npu.operators.speculative import tree_speculative_sampling_target_only

def tree_speculative_sampling_target_only(
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

目标模型完成验证前向并得到目标概率后，本算子沿候选树选择一条接受路径，
然后补采一个最终 token。调用方负责提供树结构、目标概率与随机数，并消费
输出以推进解码。

`B` 是请求数，`D` 是每个请求包含根节点的候选节点数，`S` 是接受索引的宽度，
`V` 是词表宽度。根位于局部节点 0。`retrive_next_token` 指向第一个子节点，
`retrive_next_sibling` 指向下一个兄弟节点，`-1` 表示该链接结束。

开始时当前概率行为根节点行 `r=0`，随机数为 `uniform_samples[b,0]`。
每层按兄弟链接顺序尝试子节点，对候选词 `x` 读取 `p=target_probs[b,r,x]`，
并把 `p` 累加到本层的 `prob_acc`。满足以下任一条件时接受该节点。

- `coin <= prob_acc / max(float(threshold_acc), 1e-9)`。
- `p >= threshold_single`。

接受后将 token 写入上一个接受节点的全局输出槽，登记本节点槽号，将概率行
切换为本节点行，并使用该节点对应的 `uniform_samples`。拒绝时，把该词的
目标概率写入 `draft_probs[b,r,x]`，继续尝试兄弟节点。没有可接受的子节点时
结束路径。接受索引最多容纳根和 `S-1` 个已接受草稿。

在最终概率行 `r`，计算 `w=max(target_probs[b,r]-draft_probs[b,r],0)`。
选择累计权重严格大于 `uniform_samples_for_final_sampling[b] * sum(w)` 的
第一个词，写入最后一个接受节点的输出槽。Triton 到达累计分布末尾时取所选
词块内最后一个正权重词，该块没有正权重时取 `V-1`。PyTorch 参考在整行中
查找最后一个正权重词，整行没有正权重时取 `V-1`。

## 参数与返回值

Triton 执行使用同一 NPU 上连续存储的张量。非空批要求 `D >= 1` 和
`1 <= S <= D`。调用方须提供非空词表、合法无环树、有效的 token ID 和全局
输出槽。原 wrapper 检查形状、类型、设备和连续性，索引范围、树的有效性及
概率值由调用方保证。保留 `retrive_index` 等旧参数名，确保关键字调用兼容。

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，通常 `[B*D]`，int32 | 原地输出接受的草稿及最终 token，写入 `retrive_index` 指定的全局槽。 |
| `accept_index` | `[B,S]`，int32 | 原地输出根槽和随后接受的节点槽，有效前缀长度为 `accept_token_num[b]+1`。 |
| `accept_token_num` | `[B]`，int32 | 原地输出接受的草稿数，范围为 `[0,S-1]`，最终补采 token 另计。 |
| `candidates` | `[B,D]`，int64 | 根和候选节点的词号，遍历时读取子节点的词号。 |
| `retrive_index` | `[B,D]`，int64 | 局部节点对应的全局输出槽，须落在 `predicts` 范围内且可用 int32 表示。 |
| `retrive_next_token` | `[B,D]`，int64 | 第一个子节点的局部编号，`-1` 表示没有子节点。 |
| `retrive_next_sibling` | `[B,D]`，int64 | 下一个兄弟节点的局部编号，`-1` 表示兄弟链结束。 |
| `uniform_samples` | `[B,D]`，float32 | 调用方提供的 `[0,1)` 随机数，按当前接受节点的局部编号读取。 |
| `uniform_samples_for_final_sampling` | `[B]`，float32 | 调用方提供的 `[0,1)` 最终采样随机数。 |
| `target_probs` | `[B,D,V]`，float32 | 只读的目标概率，每个节点提供下一步验证或补采所需的概率行。 |
| `draft_probs` | `[B,D,V]`，float32 | 第四个可变输出，即拒绝概率 scratch。每次调用先清零，再记录被拒绝兄弟的目标概率。 |
| `threshold_single` | float，默认 `1.0` | 单词概率达到该阈值时直接接受，取值范围 `[0,1]`。 |
| `threshold_acc` | float，默认 `1.0` | 累计概率接受阈值，取值范围 `[0,1]`，计算时使用 `max(float(threshold_acc),1e-9)`。 |
| `deterministic` | bool，默认 `True` | 保留参数，当前实现忽略其值，所有随机数均由调用方提供。 |
| `impl` | str，默认 `"auto"` | 可选 `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `None`。三只整数输出缓冲由调用方分配并初始化，未写位置保留原值。
`draft_probs` 允许使用 `empty_like(target_probs)` 分配，调用会覆盖其全部内容。
它须使用与目标概率分离的存储，旧 wrapper 在二者 `data_ptr()` 相等时报错。
四只可变缓冲须与只读输入分离，各自的写入范围也须互不重叠。

`B == 0` 时，Triton 在检查 `candidates` 和 `target_probs` 的维数后直接返回，
此时其他校验及 scratch 清零均跳过。PyTorch 参考先读取候选形状和
`accept_index.shape[1]`，转换累计阈值并执行 scratch 清零，再进入空请求循环。
符合上述形状契约的空批返回 `None`，scratch 此时也为空。

## 后端实现与来源

`triton` 在清零 scratch 后顺序启动三个 kernel，依次执行树遍历接受、按
2048 个词分块求残差权重和、最终采样。临时 `metadata` 为 `[B,2]` int64，
保存最终概率行与最终输出槽，`block_sums` 为 `[B,ceil(V/2048)]` float32。
词块数量补齐到 2 的幂供末尾采样使用。首尾两次启动各有 `B` 个程序，中间
启动数量为 `min(B*ceil(V/2048), 设备向量核数)`，各程序跨步处理词块。
性能测量需要包含 wrapper 中的 scratch 清零及三次 kernel 执行。

新 Triton 计算与完整 wrapper 复制自本仓原 `sample` 实现，仅将设备属性工具
导入改到实现层。来源记录为 #623 合入提交。新旧目录各自保留独立实现副本。
[原始 Triton 实现](https://github.com/sgl-project/sgl-kernel-npu/blob/15ea25766eb0f9a30951d7d127af8600998272ea/python/sgl_kernel_npu/sgl_kernel_npu/sample/tree_speculative_sampling_target_only.py)

`torch_native` 的计算主体复制自本仓旧测试中的 `target_only_tree_reference`，
该参考已经包含一般树的子节点和兄弟遍历、阈值以及 scratch 写入。本轮将完整
签名对齐公开接口，令原参考变量 `rejected_probs` 指向 `draft_probs`，忽略
`deterministic`，并把原参考返回四元组的接口适配为原地写入后返回 `None`。
[原始 PyTorch 树参考](https://github.com/sgl-project/sgl-kernel-npu/blob/15ea25766eb0f9a30951d7d127af8600998272ea/tests/python/sgl_kernel_npu/test_tree_speculative_sampling_target_only.py#L10-L77)

显式输入校验位于 Triton wrapper。PyTorch 后端直接执行已有参考计算，
输入须满足接口契约，错误检查时机与 Triton 有所区别。它用 Python float 累加概率和判断接受条件，并用
PyTorch 整行 `sum`/`cumsum`。Triton 使用 FP32 累加和分块归约。接近接受阈值
或累计分布边界时，已有舍入差可能改变 token 选择，精确整数比较发现分歧后
需要定位原因。

旧 `sgl_kernel_npu.sample` 目录中的实现、包导出及旧测试原样保留。原有导入
继续使用旧副本，新增接入通过 `sgl_kernel_npu.operators.speculative` 选择
本次新增的独立实现。

## 测试覆盖

完整回归保留旧测试的 24 组输入，包括 3 组一般树阈值、18 组线性树和 3 组
线性树阈值。两个后端使用各自的整数输出及 scratch，统一精度工具比较四个
可变缓冲。三只整数输出精确比较，scratch 沿用原一般树测试的零容差比较。
只读输入同时检查调用前后一致性。

另外覆盖以下 11 组 FP32 模型形状。均使用 NEXTN、TP2、topk=1 的线性候选树，
关闭 rejection sampling，两个接受阈值均为默认 `1.0`。

| 场景 | `(D,S,V)` | `B` |
| --- | --- | --- |
| GLM-4.7-Flash，NEXTN，TP2，topk=1 | `(4,4,154880)` | 1、2、4、8、16、32 |
| Qwen3.5-9B，NEXTN，TP2，topk=1 | `(4,4,248320)` | 1、2、4、8、16 |

B 档位参考 [GLM NPU CI](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/test/registered/npu/accuracy/glm4_7_flash/test_npu_glm4_7_flash_1p_gsm8k.py#L21-L52)
和 [Qwen NPU CI](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/test/registered/npu/accuracy/qwen3_5_9b/test_npu_qwen3_5_9b_bf16_1p_gsm8k.py#L24-L48)。
Qwen 的 1、2、4、8 为补充覆盖档位；两份 NPU CI 本身未开启推测解码。
D/S 来自 SGLang 的 NEXTN 配置与路径容量约定，V 来自模型官方配置，具体来源
见测试文件。它们是组合构造的回归工况，不能表述为已有 target-only 整网调用 trace。

新模型输入复用旧线性树测试的随机种子公式、概率构造和稳定最终采样随机数
生成规则，没有复制生产概率。原有多分支树用例完整保留，tree 套件共 35 项。

默认只运行 3 项：GLM `(B,D,S,V)=(1,4,4,154880)`、Qwen
`(16,4,4,248320)`，以及旧多分支树 `(2,6,4,20)`，阈值均为 `(1.0,1.0)`。
这样保留模型覆盖，以及 `S<D` 的非线性树遍历。其余 32 项标记为
`extra_checks`；`-m extra_checks` 只运行补充检查，`-m ""` 运行全部 35 项。

性能 shape 列表仍保留每个模型最小和最大 B。默认 case 只采集上述两个模型
shape；其他性能档位需要选中对应补充用例。公共 CLI 可以覆盖选中 case 是否
计时，但不覆盖 pytest 的 marker 选择。

固定输入下，重复调用会再次清零 scratch 并写入相同槽位，可复用预分配输出
进行公共性能测量。NPU 新旧直接对照、各自对独立参考的比较以及性能验收
仍须在目标设备完成。执行参数沿用[测试框架说明](../../../../../tests/python/README.md)。
