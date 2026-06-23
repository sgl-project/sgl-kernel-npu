# Platform Guide

<div align="center">

[![Platforms](https://img.shields.io/badge/Platforms-A2%20%7C%20A3%20%7C%20A5-green)]()

English | [中文](#中文)

</div>

Deployment, configuration, and testing guide for each platform.

---

## English

### A2 Single Node

**Applicable**: P/D node ranks = 8 (supports PD separation or mixed deployment).
**Not recommended**: ranks < 8 (insufficient EP parallelism).

**Build**:
```bash
bash build.sh -a deepep2
```

> [!WARNING]
> A2 requires mandatory HCCL settings. Without these, DeepEP will fail.

<details>
<summary><b>Required HCCL Settings</b></summary>

```bash
export HCCL_BUFFSIZE=1024        # Must set! Default 200MB insufficient.
unset HCCL_OP_EXPANSION_MODE     # Must unset/disable!
```

</details>

**Performance limits**:
- Normal dispatch/combine: `bs ≤ 8000`
- Low-latency dispatch/combine: `bs ≤ 512`

<details>
<summary><b>Optional Quantization (deprecated)</b></summary>

Prefill INT8 per-token via env var (deprecated — recommend tuple input):
```bash
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1   # INT8 per-token (deprecated but still functional)
```

Decode BF16 dispatch (controlled by SGLang framework, not deep_ep):
```bash
export SGLANG_DEEPEP_BF16_DISPATCH=1       # Disable quantization in low_latency_dispatch
```
> ⚠️ `SGLANG_DEEPEP_BF16_DISPATCH` is read by the SGLang framework, NOT by deep_ep code.

</details>

**Testing**:
```bash
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8
```

---

### A2 Dual Node

**Applicable**: P/D node ranks > 8 (cross-node communication).

**Build**: Same as single node (`bash build.sh -a deepep2`).

<details>
<summary><b>Required HCCL Settings</b></summary>

```bash
export HCCL_BUFFSIZE=1024
export HCCL_INTRA_PCIE_ENABLE=1    # Must set for hierarchical
export HCCL_INTRA_ROCE_ENABLE=0    # Must set for hierarchical
unset HCCL_OP_EXPANSION_MODE
```

</details>

> [!IMPORTANT]
> Normal mode does NOT support quantization on dual-node. Low-latency mode supports INT8 per-token quantization.

**Performance limits**:
- Normal dispatch/combine: `bs ≤ 4096`
- Low-latency dispatch/combine: `bs ≤ 512`

**Testing**:
```bash
# Set primary node IP in run_test_internode.sh first
bash tests/python/deepep/run_test_internode.sh
```

---

### A3

**Build**:
```bash
bash build.sh -a deepep
```

**CANN**: 8.5+ or 9.0+.

**Communication**: Pure HCCS (intranode + internode full-mesh). No hierarchical configuration needed.

<details>
<summary><b>Low-latency ops strategy comm_alg</b></summary>

| `comm_alg` | Available | Notes |
|:---|:---:|:---|
| `hierarchy` | ✅ | Requires `topk_weights` |
| `fullmesh_v1` | ✅ | Standard full-mesh |
| `fullmesh_v2` | ✅ | Has restrictions |
| `ccu` | — | Not validated on A3 |

</details>

> [!NOTE]
> Fused MoE is available on A3 with INT8 quantization (`quant_mode=1`). See [Fused MoE API](FUSED_DEEP_MOE.md).

**Testing**:
```bash
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py
python3 tests/python/deepep/test_fused_deep_moe.py
```

---

### A5 (C310)

**Build**:
```bash
bash build.sh -a deepep Ascend950
```

**CANN**: 9.0 only.

**Quantization**:

| Mode | quant_mode | Trigger | Available |
|:---|:---:|:---|:---:|
| INT8 per-token | `2` | Normal: tuple `(bf16, int8_empty)`; LL: `use_fp8=True` | ✅ |
| MXFP8 per-block | `3` | Normal: tuple `(bf16, fp8_e4m3fn_empty)`; LL: `use_fp8=True, use_ue8m0=True` | ✅ |

<details>
<summary><b>MXFP8 Usage Examples</b></summary>

**Normal mode** — pass tuple as `x` (second element is type discriminator):
```python
# Second element dtype selects quantization type:
#   torch.float8_e4m3fn → MXFP8 per-block (quant_mode=3)
#   torch.int8          → INT8 per-token (quant_mode=2)
# First element is always BF16 data — kernel quantizes internally.
x_tuple = (x_bf16, torch.tensor([], dtype=torch.float8_e4m3fn, device="npu"))
recv_x, ..., handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x is a tuple: (float8_e4m3fn_data, float8_e8m0fnu_scales)
```

**Low-latency mode** — use flags:
```python
recv_x, ... = buffer.low_latency_dispatch(x=x, use_fp8=True, use_ue8m0=True, ...)
# recv_x is a tuple: (float8_e4m3fn, float8_e8m0fnu)
```

</details>

> [!NOTE]
> Fused MoE is not yet available on A5. FP8/MXFP8 support planned for future release.

**Testing**:
```bash
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py
```

---

<a id="中文"></a>

## 中文

### A2 单机

**适用条件**：P/D 节点 ranks = 8（支持 PD 分离或混部）。
**不推荐**：ranks < 8（EP 并行度不足）。

**构建**：
```bash
bash build.sh -a deepep2
```

> [!WARNING]
> A2 必须设置 HCCL 变量，否则 DeepEP 会报错。

<details>
<summary><b>必须设置的 HCCL 变量</b></summary>

```bash
export HCCL_BUFFSIZE=1024        # 必须设置！默认 200MB 不够。
unset HCCL_OP_EXPANSION_MODE     # 必须禁用！
```

</details>

**性能上限**：
- Normal dispatch/combine：`bs ≤ 8000`
- Low-latency dispatch/combine：`bs ≤ 512`

<details>
<summary><b>可选量化（已弃用）</b></summary>

Prefill INT8 per-token 通过环境变量（已弃用——建议使用 tuple 输入）：
```bash
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1   # INT8 per-token（已弃用但仍生效）
```

Decode BF16 dispatch（由 SGLang 框架控制，非 deep_ep 读取）：
```bash
export SGLANG_DEEPEP_BF16_DISPATCH=1       # 关闭 low_latency_dispatch 量化
```
> ⚠️ `SGLANG_DEEPEP_BF16_DISPATCH` 由 SGLang 框架读取，deep_ep 代码本身不读取此变量。

</details>

**测试**：
```bash
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8
```

---

### A2 双机

**适用条件**：P/D 节点 ranks > 8（跨节点通信）。

**构建**：同单机（`bash build.sh -a deepep2`）。

<details>
<summary><b>必须设置的 HCCL 变量</b></summary>

```bash
export HCCL_BUFFSIZE=1024
export HCCL_INTRA_PCIE_ENABLE=1    # 分层通信必须设置
export HCCL_INTRA_ROCE_ENABLE=0    # 分层通信必须设置
unset HCCL_OP_EXPANSION_MODE
```

</details>

> [!IMPORTANT]
> 双机 Normal 模式**不支持量化**。Low-latency 模式支持 INT8 per-token 量化。

**性能上限**：
- Normal dispatch/combine：`bs ≤ 4096`
- Low-latency dispatch/combine：`bs ≤ 512`

**测试**：
```bash
# 先设置 run_test_internode.sh 中的主节点 IP
bash tests/python/deepep/run_test_internode.sh
```

---

### A3

**构建**：
```bash
bash build.sh -a deepep
```

**CANN**：8.5+ 或 9.0+。

**通信**：纯 HCCS（节点内 + 节点间全互联），无需分层配置。

<details>
<summary><b>Low-latency ops 策略 comm_alg</b></summary>

| `comm_alg` | 可用 | 说明 |
|:---|:---:|:---|
| `hierarchy` | ✅ | 需要 `topk_weights` |
| `fullmesh_v1` | ✅ | 标准 full-mesh |
| `fullmesh_v2` | ✅ | 有额外限制 |
| `ccu` | — | 未在 A3 验证 |

</details>

> [!NOTE]
> Fused MoE 仅 A3 可用，支持 INT8 量化（`quant_mode=1`）。详见 [Fused MoE API](FUSED_DEEP_MOE.md)。

**测试**：
```bash
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py
python3 tests/python/deepep/test_fused_deep_moe.py
```

---

### A5 (C310)

**构建**：
```bash
bash build.sh -a deepep Ascend950
```

**CANN**：仅 9.0。

**量化**：

| 模式 | quant_mode | 触发方式 | 可用 |
|:---|:---:|:---|:---:|
| INT8 per-token | `2` | Normal: tuple `(bf16, int8空tensor)`；LL: `use_fp8=True` | ✅ |
| MXFP8 per-block | `3` | Normal: tuple `(bf16, fp8_e4m3fn空tensor)`；LL: `use_fp8=True, use_ue8m0=True` | ✅ |

<details>
<summary><b>MXFP8 使用示例</b></summary>

**Normal 模式** — 以 tuple 作为 `x`（第二个元素是类型标识）：
```python
# 第二个元素 dtype 选择量化类型：
#   torch.float8_e4m3fn → MXFP8 per-block（quant_mode=3）
#   torch.int8          → INT8 per-token（quant_mode=2）
# 第一个元素始终是 BF16 数据——内核内部执行量化。
x_tuple = (x_bf16, torch.tensor([], dtype=torch.float8_e4m3fn, device="npu"))
recv_x, ..., handle, event = buffer.dispatch(x=x_tuple, ...)
# recv_x 为 tuple: (float8_e4m3fn_data, float8_e8m0fnu_scales)
```

**Low-latency 模式** — 使用 flags：
```python
recv_x, ... = buffer.low_latency_dispatch(x=x, use_fp8=True, use_ue8m0=True, ...)
# recv_x 为 tuple: (float8_e4m3fn, float8_e8m0fnu)
```

</details>

> [!NOTE]
> A5 暂不支持 Fused MoE。FP8/MXFP8 计划在未来版本支持。

**测试**：
```bash
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py
```
