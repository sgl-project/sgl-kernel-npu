# sparse_flash_mla

下文描述 CANN 9.2 原生接口；公共 API 同时适配 CANN 9.1。

上游原始文档：[Sparse Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/sparse_flash_mla/docs/torchapi_sparse_flash_mla.md)。


## CANN 9.1 兼容

`cann_ops_transformers_cann920` 是随包 CANN 9.2 实现的注册名。
CANN 9.1 的窗口输入走 `ops_transformers_cann910`；显式原始 KV 稀疏输入
使用 `ascendc` 计算；metadata 的 `impl="auto"` 在内部选择 CPU 调度路径，
该适配 API 不再单独注册 `host` 实现名。显式稀疏输入不能同时提供压缩 KV。
CPU metadata 需要已有的 CPU int32 `cu_seqlens_q_cpu`、
`seqused_ori_kv_cpu` 镜像。生产绑定均随 wheel 打包，无需独立算子包。
A5 使用单独的 mixed-quant sparse API。

迁移测试中的 `torch_native` metadata 返回独立参考描述符，不生成原生 tiling
字节；配对测试分别构造参考描述符。主仓计算参考也接受不透明的原生 metadata，
并覆盖 SWA/CSA/HCA 与 BSND。原生 metadata 的字节布局取决于 SDK。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 不支持 |
| CANN 9.1 `ops_transformers_cann910`、`ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

在 arch22 上，`auto` 根据当前 SDK 选择 CANN 9.1 或 9.2 实现。下列完整参数说明对应 CANN 9.2；未传参数使用所选绑定的默认值。

```python
from sgl_kernel_npu.operators.attention import sparse_flash_mla

def sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

## 功能说明

使用共享 KV（Key 与 Value 是同一份输入）计算稀疏 MLA 注意力，覆盖三类场景：

- **SWA**：只用 `ori_kv`，对原始 KV 做滑动窗口注意力，不传 `cmp_kv`，`cmp_ratio=1`。
- **CSA**：同时使用 `ori_kv` 窗口和由 `cmp_sparse_indices` 选出的压缩 KV，`cmp_ratio` 取 1、2 或 4。
- **HCA**：同时使用 `ori_kv` 窗口和连续的压缩 KV 段，`cmp_ratio` 取 128。

设第 $i$ 个 query 位置的有效 query 长度与 KV 长度分别为 $L_q$、$L_{ori}$、$L_{cmp}$，压缩前长度为 $L_{restored}=L_{cmp}\cdot r+\rho$（$r$ 为 `cmp_ratio`，$\rho$ 为 `cmp_residual_kv`），则：

$$
\theta_i = L_{ori} - L_q + i + 1,\qquad
\Omega_i = \left[\max(\theta_i - w_{left} - 1, 0),\ \min(\max(\theta_i + w_{right}, 0), L_{ori})\right)
$$

其中 $w_{left}$、$w_{right}$ 为 `ori_win_left`、`ori_win_right`。压缩侧上限为

$$
\kappa_i = \min\left(\max\left(\left\lfloor\frac{L_{restored} - L_q + i + 1}{r}\right\rfloor, 0\right), L_{cmp}\right)
$$

`cmp_mask_mode=3` 时只取小于 $\kappa_i$ 的压缩 token，`0` 时取全部 $L_{cmp}$ 个。CSA 按 `cmp_sparse_indices` 给出的 token 编号取数，编号 `-1` 表示终止；HCA 直接取 $[0, \kappa_i)$。参与 softmax 的键为 $K = K_{ori}[\Omega_i] \Vert K_{cmp}[\text{选中}]$，Value 取同一份数据：

$$
A_{ij}=\alpha Q_iK_j^\top,\qquad
Z_i=\sum_j e^{A_{ij}}+\delta,\qquad
O_i=\frac{\sum_j e^{A_{ij}}V_j}{Z_i},\qquad
\mathrm{LSE}_i=\log Z_i.
$$

$\alpha$ 为 `softmax_scale`。无 sink 时 $\delta=0$；存在第 $n$ 个 head 的 sink logit $s_n$ 时 $\delta=e^{s_n}$。既无 `ori_kv` 窗口也无压缩 token 时，该行输出为零。

## 参数与返回值

所有 tensor 使用相同设备；CANN 实现要求 NPU tensor。下文 `B` 为请求数，`T` 为打包后的 query token 总数，`S` 为填充后的 query 长度，`N1`/`N2` 为 query/KV 头数，`D` 为头维度，`P` 为物理页数。

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `q` | Tensor；`[T,N1,D]`（TND）或 `[B,S,N1,D]`；BF16/FP16 | 必填 | 非空 Query；CANN 要求 `D=512`。 |
| `ori_kv` | Tensor；按 `layout_kv` 为 `[T,N2,D]`、`[B,S,N2,D]` 或 `[P,page_size,N2,D]`；与 `q` 同 dtype | 必填 | 未压缩的共享 K/V。 |
| `cmp_kv` | 同上；可选 | `None` | 压缩后的共享 K/V；SWA 不传。 |
| `ori_sparse_indices` | 可选 Tensor；int32 | `None` | Atlas A2/A3 暂不支持，必须不传。 |
| `cmp_sparse_indices` | 可选 Tensor；int32；TND 为 `[T,N2,K]`，BSND 为 `[B,S,N2,K]` | `None` | CSA 必传；编号指向压缩 token，`-1` 终止。 |
| `ori_block_table`、`cmp_block_table` | 可选 Tensor `[B,L]`；int32 | `None` | `PA_BBND` 必传。 |
| `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv` | 可选 Tensor `[B+1]`；int32 | `None` | TND 布局的打包边界，首元素为 0。 |
| `seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv` | 可选 Tensor `[B]`；int32 | `None` | 每个请求的有效长度；`PA_BBND` 必传 KV 侧两个。 |
| `cmp_residual_kv` | Tensor `[B]`；int32 | `None` | 压缩余数，每个元素小于 `cmp_ratio`；`cmp_mask_mode=3` 且 `cmp_ratio!=1` 时必传，其余情况不允许传。 |
| `ori_topk_length` | 可选 Tensor；int32 | `None` | 对应 `ori_sparse_indices` 的有效长度，A2/A3 不使用。 |
| `cmp_topk_length` | 可选 Tensor；int32；TND 为 `[T,N2]`，BSND 为 `[B,S,N2]` | `None` | `cmp_sparse_indices` 的有效长度；Atlas A2/A3 为保留字段，必须不传。 |
| `sinks` | Tensor `[N1]`；float32 | `None` | 每个 head 的 softmax sink logit；Atlas A2/A3 的 kernel 要求必须传入（不能为空、也不能省略）。 |
| `metadata` | Tensor `[2048]`；int32 | 必填 | [sparse_flash_mla_metadata](sparse_flash_mla_metadata.md) 生成的分核信息。 |
| `softmax_scale` | `float` | `1.0` | QK 分数缩放系数，不会根据头维度自动推断。 |
| `cmp_ratio` | `int` | `1` | 压缩倍率；SWA 为 1，CSA 为 1/2/4，HCA 为 128。 |
| `ori_mask_mode` | `int` | `0` | `0` 无掩码；`3` 因果；`4` 滑动窗口。 |
| `cmp_mask_mode` | `int` | `0` | `0` 无掩码；`3` 因果；SWA 用 0，CSA/HCA 用 3。 |
| `ori_win_left`、`ori_win_right` | `int` | `-1` | 模式 4 的窗口左右边界；Atlas A3 要求 127 和 0。 |
| `layout_q` | `str` | `"BSND"` | `BSND` 或 `TND`。 |
| `layout_kv` | `str` | `"BSND"` | `BSND`、`TND` 或 `PA_BBND`。 |
| `topk_value_mode` | `int` | `1` | 索引取值约定；Atlas A2/A3 仅支持 1。 |
| `return_softmax_lse` | `bool` | `False` | 是否返回 FP32 log-sum-exp。 |
| `impl` | `str`，仅关键字传入 | `"auto"` | `auto`、`cann_ops_transformers_cann920` 或 `torch_native`。 |

返回 `(attention_out, softmax_lse)`，不修改输入。

| 返回值 | 类型 / Shape / Dtype | 描述 |
| --- | --- | --- |
| `attention_out` | Tensor；与 `q` 同 shape、同 dtype、同设备 | 注意力输出。 |
| `softmax_lse` | Tensor；float32；TND 为 `[N2,T,N1/N2]`，BSND 为 `[B,N2,S,N1/N2]` | 启用时为 log-sum-exp；关闭时为空 tensor `[0]`。 |

## 约束

- `metadata` 必须由 [sparse_flash_mla_metadata](sparse_flash_mla_metadata.md) 用与本次调用完全一致的布局、长度、压缩率、TopK、Mask、窗口和 KV 存在性标志生成；不一致时行为未定义。
- Atlas A2/A3 的 `num_heads_q` 只能取 1、2、4、8、16、32、64、128，`num_heads_kv` 只能为 1，`head_dim` 只能为 512。
- Atlas A2/A3 的 `ori_win_left` 必须为 127、`ori_win_right` 必须为 0；`cmp_ratio` 在 SWA 必须为 1，CSA 只能为 1/2/4，HCA 只能为 128。
- Atlas A2/A3 不支持 `ori_sparse_indices`（`ori_topk` 只能为 0），`topk_value_mode` 只能为 1。
- `sinks` 必须是一维、长度等于 `num_heads_q`、dtype 为 float32 的非空张量；Atlas A2/A3 的 kernel 不接受 `None`。
- `cmp_topk_length` 在 Atlas A2/A3 上是保留字段，即使 CSA 场景也必须不传。
- `cmp_residual_kv[i] < cmp_ratio`；`cmp_kv` 存在且 `cmp_mask_mode=3`、`cmp_ratio!=1` 时必须传入，`cmp_mask_mode=0` 或 `cmp_ratio=1` 时不允许传入。
- `PA_BBND` 要求 `seqused_ori_kv`、`ori_block_table`（以及对应压缩侧参数）同时传入；block size 为 16 的倍数，最大 1024。

## cann_ops_transformers_cann920 后端

需要 CANN 9.2.0，以及 wheel 内置的 ops-transformer 依赖。首次调用可能编译适配代码，请保留 C++ 编译环境。

不要在同一个 Python 环境里另外安装独立构建的 ops-transformer 及其 vendor：它与 wheel 内置的 vendor 声明同名算子，两者会互相覆盖。

CANN 9.2.0 自带同名算子 `SparseFlashMla`。请在导入 `torch_npu` 之前先导入 `sgl_kernel_npu`（或在启动进程前把 `ASCEND_CUSTOM_OPP_PATH` 指向 wheel 内的 vendor），否则可能命中 CANN 内置实现，出现 `metadata` 尺寸校验失败。

## torch_native 后端

用于精度对比，在输入设备上使用 FP32 计算，输出转换为 `q.dtype`。支持 `BSND`/`TND` Query、`BSND`/`TND`/`PA_BBND` KV、SWA/CSA/HCA 三类场景、sink logit 和 `return_softmax_lse`；不支持 `ori_sparse_indices` 与 `topk_value_mode!=1`。`metadata` 只作为不透明调度数据忽略，不适合图捕获；与 CANN 实现可能存在浮点舍入差异。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)

device = "npu"
q = torch.randn((16, 8, 512), device=device, dtype=torch.bfloat16) * 0.1
ori_kv = torch.randn((1, 128, 1, 512), device=device, dtype=torch.bfloat16) * 0.1
ori_table = torch.tensor([[0]], device=device, dtype=torch.int32)
seqused_ori = torch.tensor([128], device=device, dtype=torch.int32)
sinks = torch.zeros((8,), device=device, dtype=torch.float32)
boundaries = torch.tensor([0, 16], device=device, dtype=torch.int32)

metadata = sparse_flash_mla_metadata(
    8, 1, 512,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    max_seqlen_q=16,
    max_seqlen_ori_kv=128,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)
out, lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    ori_block_table=ori_table,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=512**-0.5,
    ori_mask_mode=4,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=True,
)
print(out.shape, lse.shape)  # [16, 8, 512], [1, 16, 8]
```
