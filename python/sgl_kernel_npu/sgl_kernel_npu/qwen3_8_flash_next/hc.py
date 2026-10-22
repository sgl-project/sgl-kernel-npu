"""Model-scoped hyperconnection operators for Qwen3.8-Flash-Next.

Inputs and outputs
------------------
    R is the physical token-row count, including caller graph padding.
    HC=4 branches, H=2560 hidden columns per branch, HC*H=10240, L=320.

    API           Inputs                                    Output
    ------------  ----------------------------------------  ----------
    grouped_norm  x[R,10240], weight[10240]                   [R,10240]
    mix           x[R,10240], down[320,10240], up[10240,320]   [R,2560]
    combine       block[R,2560], residual/normed[R,10240],    [R,10240]
                  weight[4,10240]

    In mix, x is the upstream learned grouped-norm output. In combine, weight
    is the branch-injection projection. group_size=2560 means four independent
    norm groups per row. hc and hs must be 4 and 2560, respectively. eps is a
    finite positive scalar, currently 1e-6. R counts tokens, not requests:
    eight requests with four verification tokens can produce R32 before padding.
    Decode, verification and prefill do not select different algorithms.

Upstream contract and validation
-------------------------------
    Public tensors are contiguous BF16 on the same NPU. The wrapper checks
    tensor type, device, dtype, contiguity, ranks, exact shapes, matching row
    counts and scalar metadata before selecting a path, including for R0.
    HC/H/L are model constraints, not tuning parameters or attention-TP shards.
    The model uses replicated dense gate weights and learned per-branch norm;
    mix and combine's normed input come from that norm. The broader GPU API
    does not authorize other shapes, dtypes, strides or out= in this wrapper.

    There is no device-value scan or host copy for dispatch. Finite values are
    an upstream assumption, not a runtime value check or a guarantee that every
    magnitude passes numerical tolerances. Learned norm weights affect the
    range; synthetic arbitrary inputs are not actual checkpoint activations.
    Each physical row is independent, including padding. No valid-row inference
    or zero-padding-output promise is made. Request metadata, padding policy
    and communication remain with the caller.

Paths and examples
------------------
    Operation     Rows       Path                    Example
    ------------  ---------  ----------------------  -------------------------
    all           R == 0     Empty, after validation No kernel computation
    grouped_norm  R > 0      Ordinary FP32 Triton    [4096,10240] -> [4096,10240]
    mix           1..8192    Native FP32 + Triton    [32,10240] -> [32,2560]
    combine       1..8192    Native FP32 + Triton    [32,2560] plus branch inputs
    mix/combine   R > 8192   Same hybrid, chunked    R8193 -> 4096+4096+1

    Norm needs no second algorithm. Native projections are an intended part
    of the hybrid, not an escape to the old framework Torch reference. Large
    R stays in this wrapper; unsupported metadata raises without silent fallback.
    R8192 and R8193 have the same mathematical contract. The 8192 boundary and
    4096-row block bound projection scratch growth; they are resource scheduling
    choices, not model row limits or decode/prefill boundaries.

Tiles and temporary storage
---------------------------
    Norm has R*4 logical 2560-element group tasks. Each uses 4096 lanes; masked
    lanes load zero, do not contribute to the sum, and are not stored. Programs
    loop over tasks beyond the available vector-core launch count. Mix's final
    stage handles 256 hidden columns across four branches per task; combine's
    final stage handles 512. The SiLU stage processes 256 flattened elements.

    Mix/combine above R8192 process consecutive slices of at most 4096 rows,
    including the tail, using the same computations. Mix bounds the rows of
    converted activations, down-projection, SiLU and up-projection temporaries;
    combine bounds converted normed activations and gate-projection rows.
    Chunked calls convert weights once, reuse them across chunks, and write
    into slices of one full output. Chunking does not bound full output storage,
    converted weights, captured graph residency or total process memory.
    Output size and captured work still grow with R; OOM is not caught/retried.

Computation and precision
-------------------------
    Norm: conceptually view each row as [4,2560]; compute mean(x*x) in FP32,
    inverse_rms=rsqrt(mean+eps), then (x*inverse_rms)*(1+weight), store BF16.
    No additional precision compensation. GPU arithmetic stages are followed,
    not CUDA bitwise parity: reduction order, rsqrt and conversion can differ.

    Mix: a = x @ down.T / 4; b = SiLU(a); gates = sigmoid(b @ up.T);
    output = mean_over_4_branches(x * gates), preserving hidden-column indices.
    Native projections, SiLU storage, gate multiplication and reduction use
    FP32; output is BF16. This does not reproduce the BF16 SiLU storage boundary
    of every GPU fused path, nor the GPU framework's small-row dispatch rules.

    Combine: gates = 2*sigmoid(normed @ weight.T / 4);
    output[r,branch,col] = residual[r,branch,col] + gates[r,branch]*block[r,col].
    Projection and elementwise intermediates use FP32; output is BF16 [R,10240].

Graph and ownership
-------------------
    Inputs/weights are not modified. Each eager invocation owns a fresh output;
    no public output aliases an input. Weight conversions are per invocation,
    not cached across calls in a way that hides in-place updates. Only immutable
    device metadata is cached. Warm up before capture. Replay may change tensor
    contents in place while preserving captured shapes, layouts, addresses and
    scalar parameters. A different R requires another capture.

    Captured outputs/storage live with their graph and caller references; replay
    overwrites that graph's output. Copy results if they must survive its next
    replay. Callers manage graph/buffer lifetimes and stream synchronization.
    Numerical acceptance requires separate validation; successful capture alone
    proves neither numerical nor model-level accuracy.
"""

import math

import torch

from . import hc_core as core


def _tensors(values):
    if any(not isinstance(x, torch.Tensor) for x in values):
        raise ValueError("Expected tensors")
    device = values[0].device
    if device.type != "npu" or any(x.device != device for x in values):
        raise ValueError("Expected tensors on the same NPU")
    if any(x.dtype != torch.bfloat16 or not x.is_contiguous() for x in values):
        raise ValueError("Expected contiguous BF16 tensors")


def _dimensions(hc, hs):
    if type(hc) is not int or type(hs) is not int or (hc, hs) != (4, 2560):
        raise ValueError("Current model requires HC=4 and H=2560")


def grouped_norm(x, weight, group_size, eps):
    _tensors((x, weight))
    if x.ndim != 2 or x.shape[1] != 10240 or weight.shape != (10240,):
        raise ValueError("Expected x[R,10240], weight[10240]")
    if type(group_size) is not int or group_size != 2560:
        raise ValueError("Expected group_size=2560")
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected finite positive epsilon")
    if x.shape[0] == 0:
        return torch.empty_like(x)
    return core.grouped_norm(x, weight, group_size, eps)


def mix(x, down, up, hc, hs):
    _dimensions(hc, hs)
    _tensors((x, down, up))
    if (
        x.ndim != 2
        or x.shape[1] != 10240
        or down.shape != (320, 10240)
        or up.shape != (10240, 320)
    ):
        raise ValueError("Expected x[R,10240], down[320,10240], up[10240,320]")
    if x.shape[0] == 0:
        return x.new_empty((0, hs))
    implementation = core.mix if x.shape[0] <= core.DIRECT_ROWS else core.mix_chunked
    return implementation(x, down, up, hc, hs)


def combine(block, residual, normed, weight, hc, hs):
    _dimensions(hc, hs)
    _tensors((block, residual, normed, weight))
    if (
        residual.ndim != 2
        or residual.shape[1] != 10240
        or normed.shape != residual.shape
        or block.shape != (residual.shape[0], hs)
        or weight.shape != (4, 10240)
    ):
        raise ValueError(
            "Expected block[R,2560], residual/normed[R,10240], weight[4,10240]"
        )
    if residual.shape[0] == 0:
        return torch.empty_like(residual)
    implementation = (
        core.combine if residual.shape[0] <= core.DIRECT_ROWS else core.combine_chunked
    )
    return implementation(block, residual, normed, weight, hc, hs)


def path_name(op, args):
    # Diagnostic only: tests/bench call this outside the timed/captured region.
    if args[0].shape[0] == 0:
        return "empty"
    if op == "grouped_norm":
        return "fp32_triton_norm"
    return (
        "fp32_native_hybrid"
        if args[0].shape[0] <= core.DIRECT_ROWS
        else "chunked_fp32_native_hybrid"
    )
