"""K=V=128 target verify using two K64 state halves and BV=128."""

import triton
import triton.language as tl


@triton.jit
def _kda_target_verify_k128_fused_kernel(
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    snapshot_ptr,
    snapshot_indices_ptr,
    out_ptr,
    scale,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_a_dim: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    initial_stride_0,
    initial_stride_1,
    initial_stride_2,
    initial_stride_3,
    snapshot_stride_0,
    snapshot_stride_1,
    snapshot_stride_2,
    snapshot_stride_3,
    snapshot_stride_4,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GATES_ARE_PREACTIVATED: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    # A5's K=128 vector path operates naturally as two 64-element halves.
    # The caller selects K=V=128: one program per (batch, value-head).
    offset_k0 = tl.arange(0, 64)
    offset_k1 = offset_k0 + 64
    offset_v = pid_v * BV + tl.arange(0, BV)
    mask_k0 = offset_k0 < K
    mask_k1 = offset_k1 < K
    mask_v = offset_v < V
    mask_state0 = mask_v[:, None] & mask_k0[None, :]
    mask_state1 = mask_v[:, None] & mask_k1[None, :]

    q_ratio = H_V // H_Q
    k_ratio = H_V // H_K
    q_head = pid_hv // q_ratio
    k_head = pid_hv // k_ratio
    initial_idx = tl.load(initial_indices_ptr + pid_batch).to(tl.int64)
    snapshot_idx = tl.load(snapshot_indices_ptr + pid_batch).to(tl.int64)

    initial_offsets0 = (
        initial_idx * initial_stride_0
        + pid_hv * initial_stride_1
        + offset_v[:, None] * initial_stride_2
        + offset_k0[None, :] * initial_stride_3
    )
    initial_offsets1 = (
        initial_idx * initial_stride_0
        + pid_hv * initial_stride_1
        + offset_v[:, None] * initial_stride_2
        + offset_k1[None, :] * initial_stride_3
    )
    state0 = tl.load(
        initial_state_ptr + initial_offsets0,
        mask=(initial_idx >= 0) & mask_state0,
        other=0.0,
    ).to(tl.float32)
    state1 = tl.load(
        initial_state_ptr + initial_offsets1,
        mask=(initial_idx >= 0) & mask_state1,
        other=0.0,
    ).to(tl.float32)

    A_log = tl.zeros((), dtype=tl.float32)
    dt_bias0 = tl.zeros((64,), dtype=tl.float32)
    dt_bias1 = tl.zeros((64,), dtype=tl.float32)
    exp_A = tl.zeros((), dtype=tl.float32)
    neg_exp_A = tl.zeros((), dtype=tl.float32)
    if not GATES_ARE_PREACTIVATED:
        A_log = tl.load(A_log_ptr + k_head).to(tl.float32)
        exp_A = tl.exp(A_log)
        neg_exp_A = -exp_A
        dt_bias0 = tl.load(
            dt_bias_ptr + k_head * K + offset_k0,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        dt_bias1 = tl.load(
            dt_bias_ptr + k_head * K + offset_k1,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)

    for step in range(0, STEPS):
        token = pid_batch * STEPS + step

        # Phase 1: fire all loads as early as possible — no inter-load deps.
        q0 = tl.load(
            q_ptr
            + token * stride_q_token
            + q_head * stride_q_head
            + offset_k0 * stride_q_dim,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        q1 = tl.load(
            q_ptr
            + token * stride_q_token
            + q_head * stride_q_head
            + offset_k1 * stride_q_dim,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)
        k0 = tl.load(
            k_ptr
            + token * stride_k_token
            + k_head * stride_k_head
            + offset_k0 * stride_k_dim,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        k1 = tl.load(
            k_ptr
            + token * stride_k_token
            + k_head * stride_k_head
            + offset_k1 * stride_k_dim,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)
        a0 = tl.load(
            a_ptr
            + token * stride_a_token
            + k_head * stride_a_head
            + offset_k0 * stride_a_dim,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        a1 = tl.load(
            a_ptr
            + token * stride_a_token
            + k_head * stride_a_head
            + offset_k1 * stride_a_dim,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)
        beta_input = tl.load(
            b_ptr + token * stride_b_token + pid_hv * stride_b_head
        ).to(tl.float32)

        # Phase 2: q/k norm and gate computation are independent — overlap.
        # Deliberately retain the measured rsqrt(sum + eps) convention.
        q_scale = scale * tl.rsqrt(tl.sum(q0 * q0 + q1 * q1, axis=0) + 1e-12)
        k_scale = tl.rsqrt(tl.sum(k0 * k0 + k1 * k1, axis=0) + 1e-12)
        q0 *= q_scale
        q1 *= q_scale
        k0 *= k_scale
        k1 *= k_scale

        if GATES_ARE_PREACTIVATED:
            gate0 = tl.exp(a0)
            gate1 = tl.exp(a1)
            beta = beta_input
        else:
            gate_input0 = a0 + dt_bias0
            gate_input1 = a1 + dt_bias1
            softplus0 = tl.where(
                gate_input0 <= 20.0,
                tl.log(1.0 + tl.exp(gate_input0)),
                gate_input0,
            )
            softplus1 = tl.where(
                gate_input1 <= 20.0,
                tl.log(1.0 + tl.exp(gate_input1)),
                gate_input1,
            )
            gate0 = tl.exp(neg_exp_A * softplus0)
            gate1 = tl.exp(neg_exp_A * softplus1)
            beta = 1.0 / (1.0 + tl.exp(-beta_input))

        # Pass 1: decay state and reduce state @ k together. The addition of
        # the two K64 products happens before a single 64-wide reduction.
        state0 *= gate0[None, :]
        state1 *= gate1[None, :]
        value = tl.load(
            v_ptr
            + token * stride_v_token
            + pid_hv * stride_v_head
            + offset_v * stride_v_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        value -= tl.sum(state0 * k0[None, :] + state1 * k1[None, :], axis=1)
        value *= beta

        # Pass 2: update state and reduce state @ q together.
        state0 += value[:, None] * k0[None, :]
        state1 += value[:, None] * k1[None, :]
        output = tl.sum(state0 * q0[None, :] + state1 * q1[None, :], axis=1)

        # Phase 4: stores.
        tl.store(
            out_ptr + (token * H_V + pid_hv) * V + offset_v,
            output,
            mask=mask_v,
        )
        snapshot_offsets0 = (
            snapshot_idx * snapshot_stride_0
            + step * snapshot_stride_1
            + pid_hv * snapshot_stride_2
            + offset_v[:, None] * snapshot_stride_3
            + offset_k0[None, :] * snapshot_stride_4
        )
        snapshot_offsets1 = (
            snapshot_idx * snapshot_stride_0
            + step * snapshot_stride_1
            + pid_hv * snapshot_stride_2
            + offset_v[:, None] * snapshot_stride_3
            + offset_k1[None, :] * snapshot_stride_4
        )
        tl.store(
            snapshot_ptr + snapshot_offsets0,
            state0,
            mask=(snapshot_idx >= 0) & mask_state0,
        )
        tl.store(
            snapshot_ptr + snapshot_offsets1,
            state1,
            mask=(snapshot_idx >= 0) & mask_state1,
        )
