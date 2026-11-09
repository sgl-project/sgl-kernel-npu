# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-ascend:
# vllm_ascend/ops/triton/glm5_next_cann_keypool.py.
# Changes: use native Triton imports and adapt the Python wrapper for
# sgl-kernel-npu. The Triton kernel is unchanged.
# License text: LICENSES/Apache-2.0.txt at the repository root.
"""Pooled index packing adapted from vllm-ascend's glm5_next_cann_keypool."""

import triton
import triton.language as tl


# Keep the upstream kernel unchanged.
# fmt: off
@triton.jit(do_not_specialize=["num_reqs"], do_not_specialize_on_alignment=["positions"])
def _pack_indices_kernel(
    indices,
    positions,
    cumulative,
    output,
    INPUT_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    num_reqs,
    OUTPUT_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid_row = row < tl.load(cumulative + num_reqs - 1)
    position = tl.load(positions + row, valid_row, other=0).to(tl.int64)
    tail_start = (position + 1) // POOL_SIZE * POOL_SIZE
    history = tl.minimum(tail_start, TOPK).to(tl.int32)
    value = tl.load(indices + row * INPUT_STRIDE + cols, valid_row & (cols < history), other=-1)
    tail_columns = cols - history
    tail_count = (position + 1 - tail_start).to(tl.int32)
    # Keep wide index vectors in int32: Ascend scalarizes int64 vector
    # conversion to the int32 SFA output. Only position arithmetic is int64.
    tail_value = tail_start.to(tl.int32) + tail_columns
    value = tl.where(
        valid_row & (tail_columns >= 0) & (tail_columns < tail_count) & (cols < TOPK + POOL_SIZE - 1),
        tail_value,
        value,
    )
    tl.store(output + row * OUTPUT_STRIDE + cols, value, cols < OUTPUT_COLS)


# fmt: on
def pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, *, output_buffer=None
):
    output = indices if output_buffer is None else output_buffer[: indices.shape[0]]
    if indices.shape[0]:
        _pack_indices_kernel[(indices.shape[0],)](
            indices,
            positions,
            query_ends,
            output,
            indices.stride(0),
            output.stride(0),
            topk,
            pool_size,
            query_ends.numel(),
            output.shape[1],
            triton.next_power_of_2(output.shape[1]),
        )
    return output[:, : topk + pool_size - 1]
