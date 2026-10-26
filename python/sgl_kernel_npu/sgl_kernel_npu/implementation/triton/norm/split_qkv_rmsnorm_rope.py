import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    NORMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    DO_PARTIAL: tl.constexpr,
    DO_HALF: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
    if NORMS:
        weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        if NORMS:
            squares = input_values * input_values
            variances = tl.sum(squares, axis=1) / HEAD_DIM
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
                Q_BLOCK_SIZE // HEAD_DIM, 1
            )
            normalized_values = (
                input_values * reciprocal_std
            )  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
            if BIAS:
                normalized_values = normalized_values * weight_values + bias_values
            else:
                normalized_values = normalized_values * weight_values
        else:
            normalized_values = input_values

        # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        if not DO_HALF:
            sin_half = al.extract_slice(
                sin, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            cos_half = al.extract_slice(
                cos, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            sin = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

            cos = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
        if DO_PARTIAL:
            rot_x = al.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            pass_x = al.extract_slice(
                normalized_values,
                offsets=(0, ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            rot_x = normalized_values
        if DO_HALF:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, 1),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        if DO_HALF:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, 1),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        roped_q = cat_x * sin + rot_x * cos
        if DO_PARTIAL:
            normalized_values = al.insert_slice(
                normalized_values,
                roped_q,
                offsets=(0, 0),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            normalized_values = al.insert_slice(
                normalized_values,
                pass_x,
                offsets=(0, ROPE_DIM),
                sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            normalized_values = roped_q
        # store
        tl.store(
            q_ptr + output_offset + col_indices,
            normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # k
    if NORMS:
        weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        if NORMS:
            squares = input_values * input_values
            variances = tl.sum(squares, axis=1) / HEAD_DIM
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(
                KV_BLOCK_SIZE // HEAD_DIM, 1
            )
            normalized_values = (
                input_values * reciprocal_std
            )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
            if BIAS:
                normalized_values = normalized_values * weight_values + bias_values
            else:
                normalized_values = normalized_values * weight_values
        else:
            normalized_values = input_values

        # # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)

        if not DO_HALF:
            sin_half = al.extract_slice(
                sin, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            cos_half = al.extract_slice(
                cos, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 1)
            )
            sin = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            sin = al.insert_slice(
                sin, sin_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

            cos = tl.zeros((1, ROPE_DIM), dtype=tl.float32)
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 0), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )
            cos = al.insert_slice(
                cos, cos_half, offsets=(0, 1), sizes=(1, HALF_ROPE_DIM), strides=(1, 2)
            )

        if DO_PARTIAL:
            rot_x = al.extract_slice(
                normalized_values,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            pass_x = al.extract_slice(
                normalized_values,
                offsets=(0, ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            rot_x = normalized_values
        if DO_HALF:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            x1 = al.extract_slice(
                rot_x,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            x2 = al.extract_slice(
                rot_x,
                offsets=(0, 1),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        if DO_HALF:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, HALF_ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 1),
            )
        else:
            cat_x = al.insert_slice(
                cat_x,
                -x2,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
            cat_x = al.insert_slice(
                cat_x,
                x1,
                offsets=(0, 1),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
                strides=(1, 2),
            )
        roped_k = cat_x * sin + rot_x * cos
        if DO_PARTIAL:
            normalized_values = al.insert_slice(
                normalized_values,
                roped_k,
                offsets=(0, 0),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
                strides=(1, 1),
            )
            normalized_values = al.insert_slice(
                normalized_values,
                pass_x,
                offsets=(0, ROPE_DIM),
                sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
                strides=(1, 1),
            )
        else:
            normalized_values = roped_k

        # store
        tl.store(
            k_ptr + output_offset + col_indices,
            normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # v
    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(
            input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0
        )
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


kernels = {}


@triton.jit
def split_qkvgate_gemma_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    q_weight_ptr,
    k_weight_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    Q_GATE_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    # q
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    qgate_hidden_size = q_hidden_size * 2

    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_GATE_BLOCK_SIZE + tl.arange(0, Q_GATE_BLOCK_SIZE)
        valid_mask = col_indices < qgate_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM * 2)
        )
        q = al.extract_slice(
            input_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
            strides=(1, 1),
        ).reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)

        gate = al.extract_slice(
            input_values,
            offsets=(0, HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM),
            strides=(1, 1),
        ).reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)

        squares = q * q
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = q * reciprocal_std  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        normalized_values = normalized_values * weight_values

        # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        rot_x = al.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        pass_x = al.extract_slice(
            normalized_values,
            offsets=(0, ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )
        x1 = al.extract_slice(
            rot_x,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = al.extract_slice(
            rot_x,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        cat_x = al.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = al.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_q = cat_x * sin + rot_x * cos
        normalized_values = al.insert_slice(
            normalized_values,
            roped_q,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        normalized_values = al.insert_slice(
            normalized_values,
            pass_x,
            offsets=(0, ROPE_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )

        # store
        col_indices_out = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask_out = col_indices_out < q_hidden_size
        tl.store(
            q_ptr + output_offset + col_indices_out,
            normalized_values.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask_out,
        )
        tl.store(
            gate_ptr + output_offset + col_indices_out,
            gate.reshape(Q_BLOCK_SIZE).to(gate_ptr.dtype.element_ty),
            mask=valid_mask_out,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # k
    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32) + 1.0
    input_offset = row_pid * total_hidden_size + q_hidden_size * 2
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = tl.rsqrt(variances + eps).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = (
            input_values * reciprocal_std
        )  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        normalized_values = normalized_values * weight_values

        # # rope
        sc_offsets = row_idx * ROPE_DIM + tl.arange(0, ROPE_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).to(tl.float32).reshape(1, ROPE_DIM)

        rot_x = al.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        pass_x = al.extract_slice(
            normalized_values,
            offsets=(0, ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )
        x1 = al.extract_slice(
            rot_x,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        x2 = al.extract_slice(
            rot_x,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM), dtype=tl.float32)
        cat_x = al.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        cat_x = al.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_ROPE_DIM),
            strides=(1, 1),
        )
        roped_k = cat_x * sin + rot_x * cos
        normalized_values = al.insert_slice(
            normalized_values,
            roped_k,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, ROPE_DIM),
            strides=(1, 1),
        )
        normalized_values = al.insert_slice(
            normalized_values,
            pass_x,
            offsets=(0, ROPE_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, PASS_DIM),
            strides=(1, 1),
        )

        # store
        tl.store(
            k_ptr + output_offset + col_indices,
            normalized_values.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    # v
    input_offset = row_pid * total_hidden_size + q_hidden_size * 2 + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(
            input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0
        )
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


kernels = {}
