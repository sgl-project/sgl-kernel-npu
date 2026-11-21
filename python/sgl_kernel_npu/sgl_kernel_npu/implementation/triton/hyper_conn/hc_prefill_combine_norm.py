# SPDX-License-Identifier: Apache-2.0
"""Qwen HC combine and grouped normalization."""

# Adapted from ops/triton/qwen4_exp/hc_fusion.py in the vLLM NPU plugin.
import triton
import triton.language as tl
import torch

@triton.jit(do_not_specialize=["ROWS"])
def _hc_sp_combine_norm_kernel(
    X, B, I, W, O, N,
    XS: tl.constexpr, BS: tl.constexpr, IS: tl.constexpr,
    OS: tl.constexpr, NS: tl.constexpr, ROWS,
    SHARED_WEIGHT: tl.constexpr, EPS: tl.constexpr,
):
    program = tl.program_id(0)
    row = (program // 4) * 4 + tl.arange(0, 4)[:, None]
    stream = program % 4
    col = tl.arange(0, 512)[None, :]
    offset = stream * 2560
    mask = row < ROWS
    x0 = tl.load(X + row * XS + offset + 0 + col, mask, 0).to(tl.float32)
    x1 = tl.load(X + row * XS + offset + 512 + col, mask, 0).to(tl.float32)
    x2 = tl.load(X + row * XS + offset + 1024 + col, mask, 0).to(tl.float32)
    x3 = tl.load(X + row * XS + offset + 1536 + col, mask, 0).to(tl.float32)
    x4 = tl.load(X + row * XS + offset + 2048 + col, mask, 0).to(tl.float32)
    injection = tl.load(I + row * IS + stream, mask, 0).to(tl.float32)
    injection = 2.0 * tl.sigmoid(injection / 4)
    b0 = tl.load(B + row * BS + 0 + col, mask, 0).to(tl.float32)
    b1 = tl.load(B + row * BS + 512 + col, mask, 0).to(tl.float32)
    b2 = tl.load(B + row * BS + 1024 + col, mask, 0).to(tl.float32)
    b3 = tl.load(B + row * BS + 1536 + col, mask, 0).to(tl.float32)
    b4 = tl.load(B + row * BS + 2048 + col, mask, 0).to(tl.float32)
    combined0 = tl.fma(b0, injection, x0).to(O.dtype.element_ty)
    combined1 = tl.fma(b1, injection, x1).to(O.dtype.element_ty)
    combined2 = tl.fma(b2, injection, x2).to(O.dtype.element_ty)
    combined3 = tl.fma(b3, injection, x3).to(O.dtype.element_ty)
    combined4 = tl.fma(b4, injection, x4).to(O.dtype.element_ty)
    tl.store(O + row * OS + offset + 0 + col, combined0, mask)
    tl.store(O + row * OS + offset + 512 + col, combined1, mask)
    tl.store(O + row * OS + offset + 1024 + col, combined2, mask)
    tl.store(O + row * OS + offset + 1536 + col, combined3, mask)
    tl.store(O + row * OS + offset + 2048 + col, combined4, mask)
    x0 = combined0.to(tl.float32)
    x1 = combined1.to(tl.float32)
    x2 = combined2.to(tl.float32)
    x3 = combined3.to(tl.float32)
    x4 = combined4.to(tl.float32)
    # Match the SGLang norm reduction, which pads 2560 columns to 4096.
    # The first pair is 2048 columns apart, then 1024, then 512.
    partial = ((x0 * x0 + x4 * x4) + x2 * x2) + (x1 * x1 + x3 * x3)
    for stage in tl.static_range(0, 3):
        partial = tl.sum(tl.reshape(partial, (4, 2, 256 >> stage)), axis=1)
    square_sum = tl.sum(partial, axis=1)
    inverse = (1.0 / tl.sqrt(square_sum / 2560 + EPS))[:, None]
    weight_offset = 0 if SHARED_WEIGHT else offset
    w0 = tl.load(W + weight_offset + 0 + col).to(tl.float32)
    w1 = tl.load(W + weight_offset + 512 + col).to(tl.float32)
    w2 = tl.load(W + weight_offset + 1024 + col).to(tl.float32)
    w3 = tl.load(W + weight_offset + 1536 + col).to(tl.float32)
    w4 = tl.load(W + weight_offset + 2048 + col).to(tl.float32)
    tl.store(N + row * NS + offset + 0 + col, (x0 * inverse) * (1.0 + w0), mask)
    tl.store(N + row * NS + offset + 512 + col, (x1 * inverse) * (1.0 + w1), mask)
    tl.store(N + row * NS + offset + 1024 + col, (x2 * inverse) * (1.0 + w2), mask)
    tl.store(N + row * NS + offset + 1536 + col, (x3 * inverse) * (1.0 + w3), mask)
    tl.store(N + row * NS + offset + 2048 + col, (x4 * inverse) * (1.0 + w4), mask)


def hc_sp_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count):
    rows, width = residual.shape
    assert hc_count == 4 and width == 10240
    assert residual.dtype == torch.bfloat16 and residual.stride(1) == 1
    assert block_output.shape == (rows, 2560) and block_output.stride(1) == 1
    assert injection_logits.shape == (rows, 4) and injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous() and norm_weight.numel() in (2560, width)
    combined = residual.new_empty(residual.shape)
    normalized = residual.new_empty(residual.shape)
    if rows:
        _hc_sp_combine_norm_kernel[(triton.cdiv(rows, 4) * hc_count,)](
            residual, block_output, injection_logits, norm_weight, combined, normalized,
            residual.stride(0), block_output.stride(0), injection_logits.stride(0),
            combined.stride(0), normalized.stride(0), rows, norm_weight.numel() == 2560, eps,
            enable_fp_fusion=False,
        )
    return combined, normalized
