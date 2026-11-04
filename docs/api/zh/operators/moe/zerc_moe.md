# ZeRCMoE (zerc_moe)

本文描述导入的 Ascend 950 eager 实现及正在开发的公开接口。构建、设备执行和
同进程共存均尚未验证。现有 `deep_ep.FuseMode.MEGA_MOE` 使用另一外部实现，
接口不同。此源码适用 CANN Open Software License Agreement 2.0，
详见 [`csrc/zerc_moe/LICENSE`](../../../../../csrc/zerc_moe/LICENSE)。

## 硬件支持

| 实现 | arch22 | arch35（Ascend 950） |
| --- | --- | --- |
| `ascendc` / `auto` | 不支持 | 注册至 arch35；构建和设备执行未验证 |
| 测试中的独立 BF16 CPU 参考 | 不适用 | 仅供 CPU 参考，未注册为后端 |

EP 的所有 peer 必须具备 UB_MEM 链路，源码要求至少两个 rank，当前没有
URMA/RDMA 回退。本次贡献尚未验证设备精度、稳定性和性能。

## 接口原型

`auto` 在 arch35 选择 `ascendc` 实现。标准构建需显式选择
位于 CANN 9.2.0 配置中的 `zerc_moe` profile：
`python3 setup.py --arch-version 3510 --third-party zerc_moe`。
打包和实际构建尚未验证。`--third-party` 指定完整 profile 集合，而不是在默认
集合上追加；如需同时打包已有 ops-transformer profile，请使用
`--third-party ops_transformer,zerc_moe`。
构建还需要 Python 的 `build`、`setuptools` 和 `wheel` 软件包。

```python
from sgl_kernel_npu.operators.moe import (
    create_zerc_moe_buffer,
    zerc_moe,
)

def create_zerc_moe_buffer(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden,
    intermediate_hidden, max_recv_token_num=0, dispatch_quant_mode=0,
    dispatch_quant_out_dtype=28, combine_quant_mode=0, comm_alg="",
): ...

def zerc_moe(
    x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer,
    scales=None, l1_weights_sf=None, l2_weights_sf=None, x_active_mask=None,
    weight1_type=28, weight2_type=28, timer=None,
    l1_weights_shared=None, l2_weights_shared=None,
    l1_weights_shared_sf=None, l2_weights_shared_sf=None, *, impl="auto",
): ...
```

参数名称、顺序和默认值与导入的原版 MegaMoE 一致；新增 `impl` 仅能以关键字传入。

## 功能描述

算子融合 Dispatch、第一层专家矩阵乘、SwiGLU、第二层专家矩阵乘和 Combine。
忽略有限精度舍入时，专家计算为 `f_e(x) = W2_e @ (silu(W1_gate_e @ x) *
(W1_up_e @ x))`；每个 token 输出为其选中专家 `p_e * f_e(x)` 之和。
可选本地共享专家的 `f_s(x)` 直接加入输出，不乘路由权重。BF16 与 MXFP8
路径的中间舍入和存储不同。

## 参数与返回值

`B` 为本 rank token 数，`H` 为隐藏维度，`I` 为专家中间维度，`E` 为本地
专家数，`K` 为选中专家数。除非另有说明，张量为 ND 存储格式的连续 NPU 张量。源码声明
`B` 范围为 `[1, 262144]`；实际可用范围可能因 workspace 和通信内存小得多。
专家 ID 是按 rank 排列的全局索引，同一 token 的专家 ID 必须有效且不重复。

| 参数 | 类型 / 形状 / dtype | 默认值 | 含义和约束 |
| --- | --- | --- | --- |
| `x` | Tensor `[B,H]`，BF16 | 必填 | 本地 token 输入。 |
| `topk_ids` | Tensor `[B,K]`，int32 | 必填 | 全局专家 ID，`1 <= K <= num_experts`。 |
| `topk_weights` | Tensor `[B,K]`，BF16 | 必填 | 路由权重。 |
| `l1_weights` | 含一个 Tensor `[E,2*I,H]` 的列表 | 必填 | 路由专家 gate/up 权重，BF16 或选定的 FP8。 |
| `l2_weights` | 含一个 Tensor `[E,H,I]` 的列表 | 必填 | 路由专家 down 权重，dtype 与第一层相同。 |
| `sym_buffer` | `ZercMoeBuffer` | 必填 | 使用匹配的通信组和参数创建。 |
| `scales`、`x_active_mask` | Tensor 或 `None` | `None` | 原接口保留参数；当前必须为 `None`。 |
| `weight1_type`、`weight2_type` | int | `28` | 原接口保留参数；当前必须为 `28`。 |
| `timer` | Tensor 或 `None` | `None` | 透传给底层；仅供启用 `ENABLE_TIMER=1` 的内部 profiling，当前构建与设备运行未验证。 |
| `l1_weights_sf` | 含一个 Tensor `[E,2*I,ceil(H/64),2]` 的列表，E8M0 | `None` | MXFP8 模式必填；BF16 模式不传。 |
| `l2_weights_sf` | 含一个 Tensor `[E,H,ceil(I/64),2]` 的列表，E8M0 | `None` | MXFP8 模式必填；BF16 模式不传。 |
| `l1_weights_shared`、`l2_weights_shared` | 含一个 Tensor `[S,2*I_s,H]`、`[S,H,I_s]` 的列表 | `None` | 可选本地共享专家，须成对提供，dtype 与路由专家权重一致。 |
| `l1_weights_shared_sf`、`l2_weights_shared_sf` | 含一个 E8M0 Tensor 的列表 | `None` | MXFP8 共享权重 scale，形状分别为 `[S,2*I_s,ceil(H/64),2]` 和 `[S,H,ceil(I_s/64),2]`；BF16 模式不传。 |
| `impl` | str，仅关键字参数 | `"auto"` | `auto` 或 `ascendc`，两者均选择此源码实现。 |

返回 `(y, expert_token_nums)`：`y` 为 BF16 `[B,H]`，`expert_token_nums` 为
int32 `[E]`，统计本 rank 各专家接收的 token 行数。

`timer` 须为与输入同设备的连续 ND 格式一维 int64 Tensor；长度由同版本的
`AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE` 决定，
由底层 host 校验。默认构建会编译掉计时逻辑，传入 `timer` 不保证产生计时数据。

### 对称缓冲区参数

| 参数 | 类型 | 默认值 | 含义和约束 |
| --- | --- | --- | --- |
| `group` | HCCL `torch.distributed.ProcessGroup` | 必填 | 已初始化的 EP 通信组，`2 <= world_size <= 768`。 |
| `num_experts` | int | 必填 | 全局专家数，等于 `E * world_size` 且不超过 1024。 |
| `num_max_tokens_per_rank` | int | 必填 | 正的各 rank 共同容量上界，用于计算缓冲区。 |
| `num_topk`、`hidden`、`intermediate_hidden` | int | 必填 | Top-k 与维度；前两者参与缓冲区容量计算。 |
| `max_recv_token_num` | int | `0` | `0` 为自动上界；非零值必须覆盖所有实际接收行，过小可能静默少算。 |
| `dispatch_quant_mode` | int | `0` | `0` 为 BF16，`4` 为 MXFP8。BF16 要求 `H % 16 == 0`。 |
| `dispatch_quant_out_dtype` | int | `28` | MXFP8 必须显式传 `23`（E5M2）或 `24`（E4M3FN）；BF16 忽略此参数。 |
| `combine_quant_mode`、`comm_alg` | int、str | `0`、`""` | 原接口保留参数；必须分别为 `0` 和空字符串。 |

共享专家要求 `1 <= S <= 4`，`I_s` 在 BF16 下为 16 的倍数，在 MXFP8
下为 64 的倍数。非零 `max_recv_token_num` 每次调用还不得超过
`B * world_size * K`。

`ZercMoeBuffer` 提供原 buffer 的只读元数据属性及存活期间的 `context`。不要在销毁后使用已取得的 context，也不要将其交给其他后端或并发调用；它的类型和可写属性与原 buffer 并非完全相同。

同一进程中，同一 HCCL group 至多创建一个 ZeRCMoE MC2 context；所有调用复用
缓冲区，在最后一次调用完成并同步后、通信组销毁前销毁缓冲区。销毁缓冲区后
不能在同一进程中为该 group 再次创建 context。其他通信后端须使用独立的进程组；
与现有 MegaMoE 提供方共用同一组尚未验证。共享专家参数未进入 Torch 图模式
schema，不能据此认为支持图模式。
若通信组已预留后创建 context 失败，同一进程中也不能为该组重试；预留前的
参数校验或导入失败则可修正后重试。

每次调用时，所有 rank 的实际 `B` 应一致；不同 batch size 尚未验证。如需
padding，调用方须自行完成，padding 行也必须使用有效且不重复的专家 ID，
并自行丢弃 padding 输出；专家计数包含 padding 行。同一 buffer 的调用不得
跨 stream 重叠。Python 锁仅串行化主机侧的启动和销毁，不能等待设备任务或
自动防止异步 stream 重叠。销毁前所有 rank 须完成并同步设备工作，再进行
组内同步并再次同步设备；异常时不要盲目进入 collective 或释放仍在使用的
buffer。

## Ascend C 后端

计划中的私有适配使用 `ZercMoe` OpType、`aclnnZercMoe` 及独立 Torch
命名空间，vendor 名为 `vendor_zerc_moe_transformer`。这些名称与现有实现
分开，但尚未通过设备构建或同进程测试验证。导入后端时 JIT 构建扩展，需要
Ninja、C++ 编译器和已初始化的 CANN 环境。在完成运行时验证前，两套实现使用
不同 HCCL 进程组。

## CPU 参考

`tests/python/zerc_moe/reference.py` 是从 CPU 列表接收所有 rank 输入的
BF16 数学参考。它不是公开注册的实现或设备精度结果，也不覆盖 MXFP8。

## 示例

以下是正在开发的公开接口未验证示例，需要使用 `torchrun` 在至少两个具备链路的
Ascend 950 rank 上启动。实际支持的形状和内存容量须在设备上验证。

```python
import os
import torch
import torch.distributed as dist
import torch_npu
from sgl_kernel_npu.operators.moe import create_zerc_moe_buffer, zerc_moe

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl")
rank, world = dist.get_rank(), dist.get_world_size()
group = dist.new_group(ranks=list(range(world)), backend="hccl")
dist.barrier(group=group)  # Ensure the HCCL communicator is initialized.
B, H, I, K = 4, 4096, 1536, 1
x = torch.randn(B, H, device="npu", dtype=torch.bfloat16)
ids = torch.full((B, K), rank, device="npu", dtype=torch.int32)
weights = torch.ones(B, K, device="npu", dtype=torch.bfloat16)
w1 = [torch.randn(1, 2 * I, H, device="npu", dtype=torch.bfloat16)]
w2 = [torch.randn(1, H, I, device="npu", dtype=torch.bfloat16)]
buffer = create_zerc_moe_buffer(group, world, B, K, H, I)
y, counts = zerc_moe(x, ids, weights, w1, w2, buffer, impl="auto")
torch.npu.synchronize()
dist.barrier(group=group)
torch.npu.synchronize()
buffer.destroy()
dist.destroy_process_group(group)
dist.destroy_process_group()
```
