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

```python
from sgl_kernel_npu.operators.moe import (
    create_zerc_moe_buffer,
    zerc_moe,
)

def create_zerc_moe_buffer(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden,
    intermediate_hidden, *, max_recv_token_num=0, dispatch_quant_mode=0,
    dispatch_quant_out_dtype=28,
): ...

def zerc_moe(
    x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer,
    *, l1_weights_sf=None, l2_weights_sf=None,
    l1_weights_shared=None, l2_weights_shared=None,
    l1_weights_shared_sf=None, l2_weights_shared_sf=None, impl="auto",
): ...
```

## Description

The operator fuses dispatch, the first expert matrix multiplication, SwiGLU,
the second expert matrix multiplication, and combine. Ignoring finite-precision
rounding, an expert computes `f_e(x) = W2_e @ (silu(W1_gate_e @ x) *
(W1_up_e @ x))`; each token's output sums `p_e * f_e(x)` for its selected
experts. Optional local shared experts add `f_s(x)` without routing weights.
The BF16 and MXFP8 paths have different intermediate rounding and storage.

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
| `l1_weights_sf` | List of one tensor `[E,2*I,ceil(H/64),2]`, E8M0 | `None` | Required in MXFP8 mode; absent in BF16 mode. |
| `l2_weights_sf` | List of one tensor `[E,H,ceil(I/64),2]`, E8M0 | `None` | Required in MXFP8 mode; absent in BF16 mode. |
| `l1_weights_shared`, `l2_weights_shared` | Lists of one tensor `[S,2*I_s,H]`, `[S,H,I_s]` | `None` | Optional local shared experts; both must be provided together with matching routed dtype. |
| `l1_weights_shared_sf`, `l2_weights_shared_sf` | Lists of one E8M0 tensor | `None` | Shared weight scales for MXFP8; shapes `[S,2*I_s,ceil(H/64),2]` and `[S,H,ceil(I_s/64),2]`. Absent in BF16 mode. |
| `impl` | str, keyword-only | `"auto"` | `auto` or `ascendc`; both select the source implementation. |

Returns `(y, expert_token_nums)`: BF16 `y` has shape `[B,H]`, and the int32
`expert_token_nums` has shape `[E]` and counts tokens received per local expert.

### Symmetric buffer parameters

| Parameter | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `group` | HCCL `torch.distributed.ProcessGroup` | Required | Initialized EP group; `2 <= world_size <= 768`. |
| `num_experts` | int | Required | Global expert count, equal to `E * world_size` and at most 1024. |
| `num_max_tokens_per_rank` | int | Required | Positive common upper bound used to size the buffer. |
| `num_topk`, `hidden`, `intermediate_hidden` | int | Required | Top-k and dimensions; the first two are used for buffer sizing. |
| `max_recv_token_num` | int | `0` | `0` selects the automatic bound; a nonzero bound must cover all actual received rows. An undersized bound may silently omit rows. |
| `dispatch_quant_mode` | int | `0` | `0` for BF16, `4` for MXFP8. BF16 requires `H % 16 == 0`. |
| `dispatch_quant_out_dtype` | int | `28` | For MXFP8 explicitly use `23` (E5M2) or `24` (E4M3FN); ignored for BF16. |

For shared experts, `1 <= S <= 4` and `I_s` must be divisible by 16 for
BF16 or 64 for MXFP8. A nonzero `max_recv_token_num` must also not exceed
`B * world_size * K` for each call.

Create at most one ZeRCMoE MC2 context per process and HCCL group; reuse the
buffer, synchronize after its last use, then destroy it before group teardown.
Destroying the buffer does not allow a second context for the same group in the
same process. Use a separate process group from other communication backends;
sharing a group with the existing MegaMoE provider is not validated.
If context creation fails after the group is reserved, do not retry it for that
group in the same process; pre-reservation validation or import failures may
be corrected and retried.
For each call, all ranks must use the same actual `B`; differing batch sizes
are not yet validated. Padding, if needed, is the caller's responsibility:
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
