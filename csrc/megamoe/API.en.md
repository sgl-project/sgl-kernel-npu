# MegaMoe source API

Contributors: chengyue, zhaoyuxin, hanggongmao, chenyukai.

The files under `source/` originate from the `mega_moe` branch of
[ops-transformer](https://gitcode.com/hanggongmao1/ops-transformer) at
`d4c5f65d93591f26895c126fc590f0d622440819`. Their original CANN Open
Software License Agreement 2.0 and copyright notices are retained; see
`LICENSE` in this directory. The repository's root MIT license does not replace
that license for these files.

The standalone build recipe obtains the pinned ops-transformer MC2 framework,
overlays the checked-in source, and produces a custom OPP installer and Python
wheel. It requires Linux, Ascend 950, a matching CANN SDK, PyTorch, torch_npu,
the Python `build`, `setuptools` and `wheel` packages, and network access to
the pinned dependencies. To inspect the commands without building, run
`python3 csrc/megamoe/build.py --plan`. To build in that environment, run
`python3 csrc/megamoe/build.py`. This recipe is separate from the standard
sgl-kernel-npu wheel build and does not install its outputs.
Importing the Python binding JIT-builds its extension and requires Ninja, a
working C++ compiler, and an initialized CANN environment including
`ASCEND_HOME_PATH`.

This document describes the Python interface included with the imported
ops-transformer MegaMoe source. It does not describe the existing DeepEP
`FuseMode.MEGA_MOE` interface, whose argument and mode contracts differ.
The source implementation targets Ascend 950 and provides an eager interface;
its shared-expert arguments are not declared in the Torch graph schema.

The CANN OpType and ACLNN names are the same as those of another MegaMoe
provider. Use this implementation only in a **fresh, isolated process** in
which no other MegaMoe provider is installed or loaded. Configure its matching
custom OPP as the only `ASCEND_CUSTOM_OPP_PATH` entry before Python starts and
do not import `deep_ep`, `sgl_kernel_npu`, or another package that modifies
that path or loads another provider in that process. Separate vendor directories
alone do not guarantee symbol isolation. This is a deployment requirement, not
an automatic check.

After building and installing the matching custom OPP and Python wheel from
the same source revision in that environment, enable the minimal source import
before starting Python:

```sh
export ASCEND_CUSTOM_OPP_PATH="<isolated-install>/vendors/sgl_megamoe_transformer"
export NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE=minimal
```

An initialized HCCL `torch.distributed.ProcessGroup` is required. The actual
entry points are:

```python
from npu_ops_transformer.ops.mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
)
```

`get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank,
num_topk, hidden, intermediate_hidden, ...)` creates communication state.
Create at most one context for a group in a process, reuse it across all calls,
and call `destroy()` only after the last operation completes. Destroying it
does not make a second context for that group safe in the same process.

`mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, ...)`
returns `(y, expert_token_nums)`. `x` and `y` are BF16 `[B, H]`; `topk_ids`
is int32 `[B, K]` with valid, distinct expert IDs per token, and
`topk_weights` is BF16 `[B, K]`. Each expert weight list contains one stacked
tensor: the two shapes are `[E_local, 2 * I, H]` and `[E_local, H, I]`.
The local expert counts are int32 `[E_local]`. BF16 mode has no weight scales;
MXFP8 mode uses FP8 weights with E8M0 scales. `scales` and `x_active_mask`
must be `None` in this source revision. Optional shared weights are local to
each rank and their output is added without a routing weight.

The independent CPU reference at `tests/python/megamoe_source/reference.py`
models BF16 routed and shared experts across all ranks, including BF16
intermediate rounding. It does not model MXFP8 and is not a substitute for
device correctness, repeated-launch, or performance measurements. No device
build or execution was performed as part of this source contribution.
