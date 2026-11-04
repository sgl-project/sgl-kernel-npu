# ZeRCMoE (zerc_moe)

This page describes the in-development public interface for the imported Ascend
950 eager implementation. Build, device execution, and same-process coexistence
have not been validated. The separate `deep_ep.FuseMode.MEGA_MOE` integration
uses another provider and has a different interface. This source is under the
CANN Open Software License Agreement 2.0; see
[`csrc/zerc_moe/LICENSE`](../../../../../csrc/zerc_moe/LICENSE).

## Hardware support

| Implementation | arch22 | arch35 (Ascend 950) |
| --- | --- | --- |
| `ascendc` / `auto` | Unsupported | Registered for arch35; build and device execution unverified |
| Independent BF16 CPU reference in tests | N/A | CPU reference only; not a registered backend |

All EP peers must have UB_MEM links. The source requires at least two ranks and
does not provide a URMA/RDMA fallback. Device accuracy, stability and
performance have not been validated for this contribution.

## Signature

`auto` selects the `ascendc` implementation on arch35. The standard build
requires CANN 9.2.0 and explicitly selecting the `zerc_moe` profile:
`python3 setup.py --arch-version 3510 --third-party zerc_moe`. The package
and build have not been verified on device. This option selects the complete
set of profiles rather than appending to the default set; to include the
existing ops-transformer profile too, use
`--third-party ops_transformer,zerc_moe`.
The build additionally requires the Python `build`, `setuptools` and
`wheel` distributions.
The default `--ops all` also selects the repository's AICPU example and thus
requires its HCC toolchain and AICPU dependencies. Use `--ops` to select the
required native operators explicitly.

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
    activation_clamp=None,
): ...
```

The imported original MegaMoE parameter names, order, and defaults are retained; the added `impl` and `activation_clamp` arguments are keyword-only.

## Description

The operator fuses dispatch, the first expert matrix multiplication, SwiGLU,
the second expert matrix multiplication, and combine. Ignoring finite-precision
rounding, an expert computes `f_e(x) = W2_e @ (silu(W1_gate_e @ x) *
(W1_up_e @ x))`; each token's output sums `p_e * f_e(x)` for its selected
experts. Optional local shared experts add `f_s(x)` without routing weights.
The BF16 and MXFP8 paths have different intermediate rounding and storage.

For a positive `activation_clamp` value `c`, gate is capped only above and up
is clamped on both sides before SwiGLU: `silu(min(gate, c)) * clamp(up, -c, c)`.
The same parameter applies to routed experts and shared experts fused by this
interface, and all ranks must use the same value in each call. `None` or `0`
disables clamping, preserving existing behavior.
This implementation has not been compiled or tested on device.

## Parameters and return values

`B` is the local token count, `H` the hidden width, `I` the expert intermediate
width, `E` the local expert count and `K` the number of selected experts.
Tensors are contiguous NPU tensors in ND storage format unless noted. The source documents `B` in
`[1, 262144]`; available workspace and communication memory may impose a much
smaller usable limit. Expert IDs are global rank-major indices and must be
valid and distinct within each token.

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `x` | Tensor `[B,H]`, BF16 | Required | Local token inputs. |
| `topk_ids` | Tensor `[B,K]`, int32 | Required | Selected global expert IDs; `1 <= K <= num_experts`. |
| `topk_weights` | Tensor `[B,K]`, BF16 | Required | Routing weights. |
| `l1_weights` | List of one tensor `[E,2*I,H]` | Required | Routed gate and up weights, BF16 or selected FP8. |
| `l2_weights` | List of one tensor `[E,H,I]` | Required | Routed down weights, same dtype as first weights. |
| `sym_buffer` | `ZercMoeBuffer` | Required | Created with the matching group and configuration below. |
| `scales`, `x_active_mask` | Tensor or `None` | `None` | Reserved by the original interface; both must be `None`. |
| `weight1_type`, `weight2_type` | int | `28` | Reserved by the original interface; both must be `28`. |
| `timer` | Tensor or `None` | `None` | Forwarded to the backend; for internal profiling builds with `ENABLE_TIMER=1`. Build and device execution are unverified. |
| `l1_weights_sf` | List of one tensor `[E,2*I,ceil(H/64),2]`, E8M0 | `None` | Required in MXFP8 mode; absent in BF16 mode. |
| `l2_weights_sf` | List of one tensor `[E,H,ceil(I/64),2]`, E8M0 | `None` | Required in MXFP8 mode; absent in BF16 mode. |
| `l1_weights_shared`, `l2_weights_shared` | Lists of one tensor `[S,2*I_s,H]`, `[S,H,I_s]` | `None` | Optional local shared experts; both must be provided together with matching routed dtype. |
| `l1_weights_shared_sf`, `l2_weights_shared_sf` | Lists of one E8M0 tensor | `None` | Shared weight scales for MXFP8; shapes `[S,2*I_s,ceil(H/64),2]` and `[S,H,ceil(I_s/64),2]`. Absent in BF16 mode. |
| `activation_clamp` | float, keyword-only | `None` | DSV4 limit; a positive finite float32 value enables clamping; `None` or `0` disables it. Negative, nonfinite and positive values not representable as nonzero finite float32 are rejected. |
| `impl` | str, keyword-only | `"auto"` | `auto` or `ascendc`; both select the source implementation. |

Returns `(y, expert_token_nums)`: BF16 `y` has shape `[B,H]`, and the int32
`expert_token_nums` has shape `[E]` and counts tokens received per local expert.

`timer` must be a contiguous 1D int64 tensor in ND format on the input device.
The host validates its version-specific length,
`AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE`.
Timing is compiled out by default, so passing a timer does not ensure timing data.

### Symmetric buffer parameters

| Parameter | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `group` | HCCL `torch.distributed.ProcessGroup` | Required | Initialized EP group; `2 <= world_size <= 768`. |
| `num_experts` | int | Required | Global expert count, equal to `E * world_size` and at most 1024. |
| `num_max_tokens_per_rank` | int | Required | Positive common upper bound used to size the buffer. |
| `num_topk`, `hidden`, `intermediate_hidden` | int | Required | The first two size the buffer; `intermediate_hidden` is metadata, and each call uses the actual weight shape. |
| `max_recv_token_num` | int | `0` | `0` selects the automatic bound; a nonzero bound must cover all actual received rows. An undersized bound may silently omit rows. |
| `dispatch_quant_mode` | int | `0` | `0` for BF16, `4` for MXFP8. BF16 requires `H % 16 == 0`. |
| `dispatch_quant_out_dtype` | int | `28` | For MXFP8 explicitly use `23` (E5M2) or `24` (E4M3FN); ignored for BF16. |
| `combine_quant_mode`, `comm_alg` | int, str | `0`, `""` | Reserved by the original interface; must be `0` and the empty string respectively. |

Unused BF16 scales and unused shared expert weights and scales may be `None`
or empty lists. Nonempty BF16 scales and supplying only one side of nonempty
shared arguments are rejected to avoid silently ignoring data. Creation-time `hidden` and
`num_topk` size the buffer; each call uses its actual dimensions, and host
tiling checks the CCL memory requirement. Shapes larger than the creation
parameters are not guaranteed to fit the allocated buffer.

For shared experts, `1 <= S <= 4` and `I_s` must be divisible by 16 for
BF16 or 64 for MXFP8. A nonzero `max_recv_token_num` must also not exceed
`B * world_size * K` for each call.

`ZercMoeBuffer` exposes writable `rank_id`, `group_name`,
`num_max_tokens_per_rank`, `num_topk`, `hidden`, and `intermediate_hidden`
metadata. Changing them does not alter the actual rank, group or allocated
capacity; the token limit remains the creation-time value. Metadata used
directly by the launch remains read-only to prevent validation and
communication state from diverging.
The context is available while live; do not use it after destruction or share
it with another backend or overlapping calls. The buffer type and writable
attributes are not identical to the original buffer.

Create at most one ZeRCMoE MC2 context per process and HCCL group; reuse the
buffer, synchronize after its last use, then destroy it before group teardown.
Destroying the buffer does not allow a second context for the same group in the
same process. Use a separate process group from other communication backends;
sharing a group with the existing MegaMoE provider is not validated.
If context creation fails after the group is reserved, do not retry it for that
group in the same process; pre-reservation validation or import failures may
be corrected and retried.
For each call, all ranks must use the same actual `B`, `H` and `K`; differing
batch sizes are not yet validated. Padding, if needed, is the caller's responsibility:
padding routes must use valid, distinct IDs, outputs must be discarded by the
caller, and expert counts include padding. Do not overlap calls on the same
buffer across streams. The Python lock serializes host launch and destruction
only; it does not wait for device work or prevent asynchronous stream overlap.
All ranks must complete and synchronize device work, synchronize the group,
and synchronize the device again before destroying the buffer. On error, do
not blindly enter a collective or release a buffer with outstanding work.
Shared-expert arguments are not represented in the Torch graph
schema; do not assume graph-mode support.

## Ascend C backend

The planned private adapter uses the `ZercMoe` OpType,
`aclnnZercMoe` and a distinct Torch namespace. Its vendor is
`vendor_zerc_moe_transformer`. This separates names from the existing
provider, but is not yet verified by a device build or same-process test.
Backend import JIT-builds a native extension and requires Ninja, a C++ compiler
and an initialized CANN environment. The two implementations should use
separate HCCL process groups pending runtime validation.

## CPU reference

`tests/python/zerc_moe/reference.py` is a small BF16 reference that takes
all ranks' inputs in CPU lists. It is not an exported implementation or a device
accuracy result; it does not cover MXFP8.

## Example

This is an unverified usage example for the in-development API, launched with
`torchrun` on at least two connected Ascend 950 ranks.
Actual supported dimensions and memory availability must be checked on device.

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
