# MegaMoe 源码接口

贡献者：chengyue、zhaoyuxin、hanggongmao、chenyukai。

`source/` 下的文件来自 [ops-transformer](https://gitcode.com/hanggongmao1/ops-transformer)
的 `mega_moe` 分支提交 `d4c5f65d93591f26895c126fc590f0d622440819`。
原 CANN Open Software License Agreement 2.0 及版权声明予以保留；完整许可
见本目录的 `LICENSE`，仓库根目录的 MIT 许可不取代这些文件的原许可。

独立构建脚本获取固定版本的 ops-transformer MC2 构建框架，用仓内源码覆盖
对应文件，生成自定义 OPP 安装包和 Python wheel。它需要 Linux、Ascend 950、
匹配的 CANN SDK、PyTorch、torch_npu、Python 的 `build`、`setuptools` 和
`wheel` 包，以及访问固定依赖的网络。可运行
`python3 csrc/megamoe/build.py --plan` 查看构建命令而不执行构建；在上述
环境中运行 `python3 csrc/megamoe/build.py` 执行构建。此脚本独立于标准的
sgl-kernel-npu wheel 构建，也不会自动安装生成的文件。

本文描述导入的 ops-transformer MegaMoe 源码所带的 Python 接口；它与
现有 DeepEP `FuseMode.MEGA_MOE` 接口的参数及支持模式不同。该实现面向
Ascend 950，提供 eager 接口；共享专家参数尚未纳入 Torch 图模式 schema。

此实现与其他 MegaMoe 提供方使用同名 CANN OpType 和 ACLNN 符号。请只在
**全新隔离进程**中使用，且该进程不安装或加载其他 MegaMoe 提供方。在
启动 Python 前，将其匹配的自定义 OPP 配置为 `ASCEND_CUSTOM_OPP_PATH`
的唯一条目，并且不要导入会修改此路径或加载其他提供方的 `deep_ep`、
`sgl_kernel_npu` 等包。不同 vendor 目录并不能保证符号隔离。这是使用
前提，并非自动检查。

在上述环境中，从同一个源码版本构建并安装匹配的自定义 OPP 和 Python
wheel 后，启动 Python 前启用最小导入：

```sh
export ASCEND_CUSTOM_OPP_PATH="<isolated-install>/vendors/sgl_megamoe_transformer"
export NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE=minimal
```

需要已初始化的 HCCL `torch.distributed.ProcessGroup`。实际入口为：

```python
from npu_ops_transformer.ops.mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
)
```

`get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank,
num_topk, hidden, intermediate_hidden, ...)` 创建通信状态。同一进程中，
同一通信组至多创建一个 context；所有调用复用它，最后一次操作结束后
再调用 `destroy()`。销毁后不能认为该进程可以为同一组安全地再次创建。

`mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, ...)`
返回 `(y, expert_token_nums)`。`x` 和 `y` 为 BF16 `[B, H]`，`topk_ids`
为 int32 `[B, K]`，每个 token 的专家编号必须有效且互不重复；
`topk_weights` 为 BF16 `[B, K]`。两组专家权重均以一个堆叠 Tensor
组成的列表传入，形状分别为 `[E_local, 2 * I, H]` 和 `[E_local, H, I]`。
本 rank 专家计数为 int32 `[E_local]`。BF16 模式不使用权重 scale；
MXFP8 模式使用 FP8 权重和 E8M0 scale。本源码版本要求 `scales` 和
`x_active_mask` 为 `None`。可选共享专家属于本 rank，输出直接相加，
不乘路由权重。

`tests/python/megamoe_source/reference.py` 提供独立 CPU 参考，覆盖全部
rank 的 BF16 路由和共享专家及 BF16 中间舍入，不覆盖 MXFP8，不能替代
设备精度、连续调用稳定性和性能验证。本次源码贡献未编译或执行设备测试。
