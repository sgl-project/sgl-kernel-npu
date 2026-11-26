# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PyTorch translation of the legacy sparse_block_estimate AscendC stages.

Source: csrc/attentions/csrc/ops/sparse_block_estimate at 064448af8f67.
Keep its FP16 intermediates, raw layout addressing and score-selection quirks.
This is an independent reference; it never calls the native operator.
"""

import struct

import torch

# The legacy Tiling callback mutates these two process-wide values too.
_single_core_mbase = 128
_single_core_nbase = 1024


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _vector_cores(device):
    if device.type == "cpu":
        # CPU execution is useful for stage diagnostics, not device parity:
        # use one core rather than guessing a particular accelerator model.
        return 1
    import torch_npu

    return int(torch_npu.npu.get_device_properties(device).vector_core_num)


def _tiling_factors(seq_q, seq_k, heads, stride, sparse_size, vector_cores):
    global _single_core_mbase, _single_core_nbase
    while seq_k < stride * 2 * _single_core_nbase:
        _single_core_nbase //= 2
    _single_core_nbase = max(_single_core_nbase, 512)
    tasks = (heads * seq_q + vector_cores - 1) // vector_cores
    tasks = (tasks + sparse_size - 1) // sparse_size
    _single_core_mbase = min(_single_core_mbase, tasks * sparse_size // stride)
    return _single_core_mbase, _single_core_nbase


def _whole_sum(values):
    """Half reduction with a rounded result at every addition level."""
    work = values.clone()
    width = work.shape[-1]
    while width > 1:
        split = (width + 1) // 2
        pairs = width - split
        work[..., :pairs] = work[..., :pairs] + work[..., split:width]
        width = split
    return work[..., 0]


def _row_sum(values):
    """RowSum's 128-half vector folding followed by WholeReduceSum."""
    work = values.clone()
    width = work.shape[-1]
    blocks, remain = divmod(width, 128)
    if blocks and remain:
        work[..., :remain] += work[..., blocks * 128 : blocks * 128 + remain]
    while blocks // 2:
        pairs = blocks // 2
        blocks = (blocks + 1) // 2
        for index in range(pairs):
            first = index * 128
            other = (index + blocks) * 128
            work[..., first : first + 128] += work[..., other : other + 128]
    return _whole_sum(work[..., : min(width, 128)])


def _column_sum(values, reduce_size):
    """ColumnSum/AccumulateByRow retain the legacy binary addition order."""
    work = values.clone()
    rows = work.shape[0]
    step = reduce_size
    while step:
        if rows & step:
            if rows > step:
                work[: rows - step] += work[step:rows]
            while step > 1:
                step //= 2
                work[:step] += work[step : 2 * step]
            return work[0]
        step //= 2
    return work[0]


def _score_mask(scores, active, threshold, causal, keep_sink, keep_recent, row_sparse):
    """ScoreCompute, including forced first column even with keep_sink=False."""
    if active <= 2 and causal:
        return torch.ones(active, dtype=torch.int8, device=scores.device)
    cols = (active + 31) // 32 * 32
    work = torch.zeros(cols, dtype=torch.float16, device=scores.device)
    work[:active] = scores[:active]
    if keep_recent or not causal:
        work[active - 1] = 0
    if keep_sink:
        work[0] = 0
    rs = float(_row_sum(work[: active - 1 if keep_recent else active]).item())
    ordered = work.sort(descending=True).values
    guard = 0.0
    count = active
    if threshold < 1.0:
        total, block = 0.0, 0
        piece_sums = _row_sum(ordered.reshape(-1, 16)).tolist()
        target = _f32(threshold * rs)
        for index, value in enumerate(piece_sums):
            if _f32(total + value) > target:
                block = index * 16
                break
            total = _f32(total + value)
        for index in range(16):
            guard = float(ordered[block + index].item())
            total = _f32(total + guard)
            count = block + index
            if total >= target:
                break
    if count > _f32(active * row_sparse) and active >= 10:
        kk = int(_f32(active * row_sparse))
        guard = float(ordered[kk - 1].item())
    guard = _f32(guard - _f32(1e-5))
    if keep_sink:
        work[0] = _f32(guard + 1.0)
    if keep_recent or not causal:
        work[active - 1] = _f32(guard + 1.0)
    positive = (work[:active].float() - guard).relu()
    mask_half = (positive / (positive + _f32(0.001))).half()
    mask_half[0] = 1  # The original Duplicate is unconditional.
    return mask_half.ceil().to(torch.int8)


def _query_tile(flat, offset, length, dim, heads, stride, interleaved):
    """ReorderQKTail uses 32-byte block strides, including its BSH stride."""
    padded = (length + stride - 1) // stride * stride
    tile = torch.empty((padded, dim), dtype=flat.dtype, device=flat.device)
    capacity = 4 * 32768 // 2 // dim
    copied = dim // 16 * 16
    pitch = copied + (heads - 1) * dim // 32 * 16 if interleaved else dim
    for start in range(0, length, capacity):
        size = min(capacity, length - start)
        base = offset + start * (heads if interleaved else 1) * dim
        view = flat.as_strided((size, copied), (pitch, 1), flat.storage_offset() + base)
        tile[start : start + size, :copied] = view
    tile[length:] = 0
    return tile.reshape(-1, stride, dim).flip(1).reshape(-1, stride * dim)


def _block_scores(
    q,
    flat_k,
    k_offset,
    key_length,
    dim,
    kv_heads,
    stride,
    sparse_size,
    nbase,
    interleaved,
    scale,
):
    """Res1VecCompute and SecondReduce, keeping online maxima and half sums."""
    reduce_size = sparse_size // stride
    samples = key_length // stride  # The legacy kernel does not evaluate a K tail.
    running_max = torch.full((q.shape[0],), -float("inf"), device=q.device)
    running_sum = torch.zeros_like(running_max)
    stages = []
    for first in range(0, samples, nbase):
        size = min(nbase, samples - first)
        offset = k_offset + first * stride * dim * (kv_heads if interleaved else 1)
        # Matmul's Kb is dim*stride even for BSND/BSH. Do not substitute an
        # idealized head-wise gather for the original raw contiguous B matrix.
        k = flat_k.as_strided(
            (size, stride * dim), (stride * dim, 1), flat_k.storage_offset() + offset
        )
        logits = (q.float() @ k.float().T) * scale
        maximum = torch.maximum(running_max, logits.amax(dim=-1))
        exponents = (logits - maximum[:, None]).exp()
        running_sum = running_sum * (running_max - maximum).exp() + exponents.sum(-1)
        running_max = maximum
        padded = (size + reduce_size - 1) // reduce_size * reduce_size
        half = torch.zeros((q.shape[0], padded), dtype=torch.float16, device=q.device)
        half[:, :size] = exponents.half()
        first_sum = _row_sum(half.reshape(q.shape[0], -1, reduce_size))
        stages.append((maximum, first_sum))
    rows = (q.shape[0] + reduce_size - 1) // reduce_size
    columns = (key_length + sparse_size - 1) // sparse_size
    result = torch.zeros((rows, columns), dtype=torch.float16, device=q.device)
    for index, (maximum, first_sum) in enumerate(stages):
        factor = ((maximum - running_max).exp() / running_sum).half()
        weighted = first_sum * factor[:, None]
        first_col = index * (nbase // reduce_size)
        for row in range(rows):
            value = _column_sum(
                weighted[row * reduce_size : (row + 1) * reduce_size], reduce_size
            )
            size = min(value.numel(), columns - first_col)
            result[row, first_col : first_col + size] = value[:size]
    return result


def _empty(shape, query, dtype):
    if query.device.type == "npu":
        import torch_npu

        return torch_npu.empty_with_format(
            shape,
            dtype=dtype,
            device=query.device,
            acl_format=torch_npu.get_npu_format(query),
        )
    return torch.empty(shape, dtype=dtype, device=query.device)


def sparse_block_estimate(
    query,
    key,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    input_layout="BNSD",
    stride=8,
    sparse_size=128,
    num_heads=1,
    num_key_value_heads=1,
    scale_value=1,
    threshold=1,
    causal=True,
    keep_sink=True,
    keep_recent=True,
    row_sparse=1,
):
    """Translate the old kernel; final integer outputs require exact comparison.

    CPU calls support arithmetic diagnostics only. NPU calls use the actual
    vector-core count to reproduce Tiling's query tile size. Like the original,
    rows beyond actual_seq_lengths are uninitialized.
    """
    if num_heads == 0:
        raise RuntimeError("num_heads must be nonzero.")
    if sparse_size == 0:
        raise RuntimeError("sparse_size must be nonzero.")
    interleaved = input_layout != "BNSD"
    if input_layout == "BNSD":
        batch, nq, seq_q, dim = query.shape
        seq_k = key.shape[2]
    elif input_layout == "BSND":
        batch, seq_q, nq, dim = query.shape
        seq_k = key.shape[1]
    elif input_layout == "BSH":
        batch, seq_q, hidden = query.shape
        nq, dim, seq_k = num_heads, hidden // num_heads, key.shape[1]
    else:
        raise NotImplementedError(
            "The legacy Torch wrapper does not define outputs for this layout"
        )
    blocks = (seq_q + sparse_size - 1) // sparse_size
    aligned = (blocks + 31) // 32 * 32
    mask = _empty((batch, nq, blocks, aligned), query, torch.int8)
    count = _empty((batch, nq, blocks), query, torch.int32)
    mbase, nbase = _tiling_factors(
        seq_q, seq_k, num_heads, stride, sparse_size, _vector_cores(query.device)
    )
    scale, threshold, row_sparse = map(_f32, (scale_value, threshold, row_sparse))
    qflat, kflat = query.contiguous().reshape(-1), key.contiguous().reshape(-1)
    q_lengths = actual_seq_lengths or [seq_q]
    k_lengths = actual_seq_lengths_kv or [seq_k]
    for b in range(batch):
        qlen = q_lengths[b if len(q_lengths) > 1 else 0]
        klen = k_lengths[b if len(k_lengths) > 1 else 0]
        for head in range(num_heads):
            kv_head = head // (num_heads // num_key_value_heads)
            if interleaved:
                qbase = b * seq_q * num_heads * dim + head * dim
                kbase = b * seq_k * num_key_value_heads * dim + kv_head * dim
            else:
                qbase = (b * num_heads + head) * seq_q * dim
                kbase = (b * num_key_value_heads + kv_head) * seq_k * dim
            for start in range(0, qlen, mbase * stride):
                length = min(mbase * stride, qlen - start)
                effective_k = start + length // stride * stride if causal else klen
                if effective_k // stride == 0:
                    continue  # Original inner loop writes nothing in this case.
                offset = qbase + start * dim * (num_heads if interleaved else 1)
                q = _query_tile(
                    qflat, offset, length, dim, num_heads, stride, interleaved
                )
                scores = _block_scores(
                    q,
                    kflat,
                    kbase,
                    effective_k,
                    dim,
                    num_key_value_heads,
                    stride,
                    sparse_size,
                    nbase,
                    interleaved,
                    scale,
                )
                for row in range(scores.shape[0]):
                    output_row = start // sparse_size + row
                    active = (
                        output_row + 1
                        if causal
                        else (klen + sparse_size - 1) // sparse_size
                    )
                    values = _score_mask(
                        scores[row],
                        active,
                        threshold,
                        causal,
                        keep_sink,
                        keep_recent,
                        row_sparse,
                    )
                    # MaskCopyOut writes all padded K columns for a valid row.
                    padded_k = (
                        ((seq_k + sparse_size - 1) // sparse_size + 31) // 32 * 32
                    )
                    mask[b, head, output_row, :padded_k] = 0
                    mask[b, head, output_row, :active] = values
                    count[b, head, output_row] = values.to(torch.int32).sum()
    return mask, count
