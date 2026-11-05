# verify_tree_greedy

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 已注册，待本轮实机验证 | 已注册，待本轮实机验证 |
| `torch_native` | 已注册，待本轮实机验证 | 已注册，待本轮实机验证 |

`auto` 选择 `triton`，不根据树形状回退到其他实现。当前注册声明不代表 NPU
精度或性能验收已通过。

## 接口签名

```python
from sgl_kernel_npu.operators.speculative import verify_tree_greedy

def verify_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    *,
    impl="auto",
):
    ...
```

## 功能与实际支持范围

在 SGLang 贪心验证路径中，先对目标模型的 logits 做 argmax，再判断草稿是否与目标
token 一致，并提交最长接受前缀及一个最终 token。随机数、概率表和词表大小
都不是本算子的输入。词表 argmax、模型前向及 KV 状态更新由调用方完成。

**迁入的旧 Triton 实际验证线性链。** 它按列比较候选与前一列目标，不读取
子节点或兄弟链接；`accept_index` 的物理行跨度固定为候选宽度 `D`。共同精度
比较范围为连续的 `[B,D]` 线性链、`accept_index[B,D]` 以及
`retrive_index[b,j] = b*D+j` 的标准全局槽位映射。

它不支持任意多分支树，也不支持将接受索引宽度缩为 `S<D`。旧测试中
`B=2,D=6,S=4` 的多叉树用例只执行原 native；本轮保留该用例，并在新公开接口
下单独检查 `torch_native`，不对该数据启动 Triton。旧 wrapper 没有参数校验，
本轮保留其行为；错误的行跨度可能导致越界写入，调用方须满足上述范围。

记 `c_j=candidates[b,j]`、`t_j=target_predict[b,j]`。从 `j=1` 起，接受条件是
`c_j == t_(j-1)`；第一次不等即停止。若接受草稿数为 `a`，则：

- `accept_token_num[b] = a`，范围为 `[0,D-1]`。
- 写入 `accept_index[b,0:a+1]`，包含根槽及接受槽。
- 在标准槽位映射下，写入 `predicts[b*D+j] = t_j`，其中 `0 <= j <= a`。
  最后一项是补出的目标 token，SGLang 计算解码推进长度时把它计入。
- 其他输出位置保留原值，返回 `None`。

## 参数

| 参数 | shape / dtype | 含义 |
| --- | --- | --- |
| `predicts` | 一维，长度至少 `B*D`，int32 | 调用方分配的原地 token 输出。 |
| `accept_index` | `[B,D]`，int32 | 原地根槽及接受槽索引；未写位置保留初值。 |
| `accept_token_num` | `[B]`，int32 | 接受的草稿数，不含最终补出 token。 |
| `candidates` | `[B,D]`，int32 或 int64 | 含根的候选 token，根列的 token 值不参与比较。 |
| `retrive_index` | `[B,D]`，整数 | 标准映射 `b*D+j`。 |
| `retrive_next_token` | `[B,D]`，整数 | 线性链的首子节点局部编号；Triton 忽略此参数，native 使用。 |
| `retrive_next_sibling` | `[B,D]`，整数 | 线性链中为 `-1`；Triton 忽略此参数，native 使用。 |
| `target_predict` | `[B,D]`，int32 或 int64 | 目标模型为每个节点预测的下一个 token ID。 |
| `impl` | str，默认 `"auto"` | `"auto"`、`"triton"`、`"torch_native"`。 |

使用同一 NPU 上连续存储的张量，`B>=1`、`D>=1`。本轮保留旧 int32 用例；
模型形状使用 int64 输入，与 SGLang 的索引缓冲和 argmax token ID 类型对齐。
两类输入在本轮 NPU 上的执行均待验证。输出须与输入及其他输出范围互不重叠，
token 和槽位须可用 int32 输出表示。不声明空批和任意非连续布局的支持。

## 后端与来源

`triton` 从本仓 `sample/verify_tree_greedy.py` 原样复制 kernel 和完整 wrapper。
每个请求一个程序，先读取补齐到 2 的幂宽度的候选、槽位和目标 token，随后
执行顺序前缀比较。没有临时概率缓冲、归约或随机数。记录的引入提交为
`96dfd27fbc15bf893f96d8d7837d818086d23d04`（#165）；直接复制版本为本分支基线
`26ec66efe01f8acef43db55da679107702876b23`，包含 #473 的 CANN 接口适配。

`torch_native` 复制本仓 `speculative.py` 的 `verify_tree_greedy_native` 作为私有
函数，计算主体及原返回行为保留。新 wrapper 按原默认 `topk=-1` 调用其原地
遍历路径，丢弃返回的三个张量以适配公开接口的 `None` 返回。原函数的
`topk=1,D=2` 快捷分支会分配替代输出，因此新公开 wrapper 不选择该分支。
这不改变候选树的 topk；公开接口和旧 Triton 本来就没有 `topk` 参数。

native 使用子节点和兄弟链接，能够表达旧多叉树参考用例。它在共同线性范围
内充当独立精度参考；更广的参考能力不扩大 Triton 的支持范围。两个后端之间
没有互相调用，原算子、原 `speculative.py` 和原测试均保持原样。

## 测试与验证状态

默认测试包括旧 int32 线性用例和 11 组模型形状各三种接受路径，共 34 项：

| 场景 | `D=S` | 词号范围 `V` | `B` |
| --- | --- | --- | --- |
| GLM-4.7-Flash，NEXTN greedy，TP2，topk1 | 4 | 154880 | 1、2、4、8、16、32 |
| Qwen3.5-9B，NEXTN greedy，TP2，topk1 | 4 | 248320 | 1、2、4、8、16 |

`V` 只用于构造合法 token ID，不是输入张量维度。形状沿用已确认的模型及部署
覆盖范围，候选、匹配位置和最终 token 是本轮构造的数据。两份来源 NPU CI
本身未启用 NEXTN，不能将这些组合形状称为生产调用 trace。源码记录固定的
SGLang 配方及官方模型配置链接。默认性能子集取各模型最小、最大 B，公共 CLI
可覆盖该选择。

`extra_checks` 包含 6 个根节点/非 2 的幂宽度及 dtype 检查，以及一个明确只测
`torch_native` 的原多叉树结果回归，共 41 项。后者按旧测试的固定期望值比较，
不受待测后端选项切换，也不进行性能采集。
公共 CLI 跳过精度时，该固定结果回归也不执行。

两个后端分别分配三只 int32 输出，使用公共整数精确比较，同时检查返回值、
输入只读。CPU 参考回放仅核对 native 迁移和输入，不代表 Triton
执行。NPU 的新旧直接对照、对独立参考的精度及同机性能仍待验收。

来源：[旧 Triton](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/python/sgl_kernel_npu/sgl_kernel_npu/sample/verify_tree_greedy.py)、
[旧 native](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/python/sgl_kernel_npu/sgl_kernel_npu/speculative.py#L135)、
[SGLang greedy 调用](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/python/sglang/srt/speculative/eagle_utils.py#L733-L745)、
[NPU 分派](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/python/sglang/kernels/ops/speculative/spec_tree.py#L523-L536)。
执行参数沿用[测试框架说明](../../../../../tests/python/README.md)。
