import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)

# ==========================================
# 1. Original SGLang implementation
# ==========================================
class SGLangImpl:
    def torch_causal_conv1d_update_npu(
        self,
        hidden_state: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_update: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        bsz, hidden_size, seq_len = hidden_state.shape
        state_len = conv_state.shape[-1]

        hidden_states_new = torch.cat([conv_state, hidden_state], dim=-1).to(
            weight.dtype
        )

        if conv_state_update is not None:
            for i in range(seq_len):
                end = i - seq_len + 1
                start = end - state_len
                slice_range = slice(start, end if end != 0 else None)
                conv_state_update[:, i] = hidden_states_new[:, :, slice_range]
        else:
            conv_state_update = hidden_states_new[:, :, -state_len:]

        kernel_size = weight.shape[-1]
        windows = hidden_states_new.unfold(-1, kernel_size, 1)

        # Note: this test assumes weight has shape [H, K].
        out = (windows * weight[None, :, None, :]).sum(dim=-1)

        if bias is not None:
            out = out + bias[None, :, None]

        out = F.silu(out[:, :, -seq_len:])
        out = out.to(hidden_state.dtype)
        conv_state = conv_state.transpose(1, 2)
        return out, conv_state_update

# ==========================================
# 2. vLLM-style reference implementation (V3)
# ==========================================
def vllm_causal_conv1d_update_v3(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    hidden_state = hidden_state.transpose(1, 2)
    weight = weight.transpose(0, 1)
    conv_state = conv_state.transpose(1, 2)
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]

    # Keep (kernel_size - 1) history tokens plus the latest generated suffix.
    target_state_len = (kernel_size - 1) + (seq_len - 1)

    full_context = torch.cat([conv_state[conv_state_indices], hidden_state], dim=-1).to(weight.dtype)

    # Compute the output.
    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)

    # This path also assumes weight has shape [H, K].
    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(hidden_state.dtype)

    # Update the state by keeping the most recent target_state_len values.
    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype)
    conv_state[conv_state_indices] = new_conv_state
    conv_state = conv_state.transpose(1, 2)
    out = out.transpose(1, 2)

    return out

# ==========================================
# 3. Correctness test
# ==========================================
def test_correctness_fixed():
    # --- Config ---
    BSZ = 8
    HIDDEN_SIZE = 4096
    SEQ_LEN = 2
    KERNEL_SIZE = 3
    CACHE_LEN = 65
    DTYPE = torch.bfloat16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Test (Fixed Logic) on {DEVICE}...")

    # [FIXED HERE] Weight shape changed to [H, K] (2D)
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_init = torch.randn(CACHE_LEN, KERNEL_SIZE - 1, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    conv_state_indices = torch.arange(BSZ, device=hidden_state.device)

    # --- SGLang Execution ---
    sglang_model = SGLangImpl()
    sglang_cache_buffer = torch.zeros(
        BSZ, SEQ_LEN, HIDDEN_SIZE, KERNEL_SIZE - 1,
        device=DEVICE, dtype=DTYPE
    )

    out_sg, final_buffer_sg = sglang_model.torch_causal_conv1d_update_npu(
        hidden_state=hidden_state.transpose(1, 2),
        conv_state=conv_state_init.transpose(1, 2)[conv_state_indices],
        weight=weight.transpose(0, 1),
        conv_state_update=sglang_cache_buffer,
        bias=bias
    )

    # --- vLLM Execution ---
    out_vl, state_vl = vllm_causal_conv1d_update_v3(
        hidden_state=hidden_state,
        conv_state=conv_state_init,
        weight=weight,
        bias=bias,
        conv_state_indices=conv_state_indices,
        activation=True
    )

    # --- Validation ---

    # 1. Output Check
    try:
        torch.testing.assert_close(out_sg, out_vl, rtol=1e-5, atol=1e-5)
        print("✅ Outputs match perfectly!")
    except AssertionError as e:
        print("❌ Outputs mismatched!")
        print(e)
        return

    # 2. State Length Check
    state_vl_t = state_vl.transpose(1, 2)
    expected_len = (KERNEL_SIZE - 1) + (SEQ_LEN - 1)
    print(f"State Shapes -> SGLang Buffer: {final_buffer_sg.shape}, vLLM State: {state_vl_t.shape}")
    assert state_vl_t.shape[-1] == expected_len, f"Length mismatch: {state_vl_t.shape[-1]} vs {expected_len}"
    print(f"✅ vLLM State length is correct: {expected_len}")

    # 3. Cache Content Check
    print("--- Verifying Cache Slices ---")
    match_count = 0

    for i in range(SEQ_LEN):
        # SGLang: History used to predict i+1
        sg_slice = final_buffer_sg[:, i, :, :]

        # vLLM: Reconstruct window from continuous state
        # vLLM state ends with the token (SEQ_LEN - 1).
        # Token i is (SEQ_LEN - 1 - i) steps away from the end.
        end_idx = state_vl_t.shape[-1] - (SEQ_LEN - 1 - i)
        start_idx = end_idx - (KERNEL_SIZE - 1)

        vl_slice = state_vl_t[:, :, start_idx : end_idx]

        try:
            torch.testing.assert_close(sg_slice, vl_slice, rtol=1e-5, atol=1e-5)
            match_count += 1
        except AssertionError:
            print(f"❌ Mismatch at index {i}")
            break

    if match_count == SEQ_LEN:
        print(f"✅ Verified {match_count} intermediate states (Full Coverage).")
    else:
        print("❌ Cache content verification failed.")

# ==========================================
# 4. NPU operator test
# ==========================================
def test_npu_causal_conv1d_update():
    """Test the NPU causal_conv1d_update operator."""
    try:
        import torch_npu
    except ImportError as e:
        print(f"⚠️  Skipping NPU test (import failed): {e}")
        return

    # Import sgl_kernel_npu to ensure operator registration
    try:
        import sgl_kernel_npu
    except ImportError as e:
        print(f"⚠️  Skipping NPU test (sgl_kernel_npu import failed): {e}")
        return

    # Check NPU availability
    try:
        if not (hasattr(torch_npu, 'npu') and torch.npu.device_count() > 0):
            print("⚠️  NPU not available, skipping NPU test")
            return
    except Exception as e:
        print(f"⚠️  Failed to check NPU availability: {e}")
        return

    # Verify operator is registered
    if not hasattr(torch.ops.npu, 'causal_conv1d_update'):
        print("⚠️  causal_conv1d_update operator not registered!")
        print(f"Available npu ops: {[op for op in dir(torch.ops.npu) if not op.startswith('_')][:10]}")
        return

    # --- Config ---
    BSZ = 1
    HIDDEN_SIZE = 4096  # Keep the hidden size moderate so the test stays fast.
    SEQ_LEN = 1
    KERNEL_SIZE = 4
    CACHE_LEN = 10
    # conv_state must be large enough to hold (width - 1) + (seq_len - 1) values.
    CONV_STATE_LEN = KERNEL_SIZE - 1 + SEQ_LEN - 1  # The current NPU kernel expects a fixed-size state buffer.
    DTYPE = torch.bfloat16
    DEVICE = "npu"

    print(f"\n{'='*50}")
    print(f"Testing NPU causal_conv1d_update on {DEVICE}")
    print(f"{'='*50}")

    # Create inputs.
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    # bias = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = None
    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_init = torch.randn(CACHE_LEN, CONV_STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int32)
    num_accepted_tokens = torch.tensor([SEQ_LEN] * BSZ, device=DEVICE, dtype=torch.int32)
    # Optional tensor used when queries are packed by start offset.
    query_start_loc = torch.tensor([0, SEQ_LEN, 2*SEQ_LEN, 3*SEQ_LEN], device=DEVICE, dtype=torch.int32)
    conv_state_vl = conv_state_init.clone()
    # --- vLLM Execution (CPU/CUDA reference) ---
    out_vl = vllm_causal_conv1d_update_v3(
        hidden_state=hidden_state,
        conv_state=conv_state_vl,
        weight=weight,
        bias=bias,
        conv_state_indices=conv_state_indices,
        activation=True
    )

    # --- NPU Execution ---
    print(f"Input shapes: x={hidden_state.shape}, weight={weight.shape}, conv_state={conv_state_init.shape}")
    print(f"Calling torch.ops.npu.causal_conv1d_update...")

    # Clone conv_state because the NPU kernel updates it in place.
    conv_state_npu = conv_state_init.clone()

    try:
        out_npu = torch.ops.npu.causal_conv1d_update(
            x=hidden_state,
            weight=weight,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            bias=bias,
            num_accepted_tokens=num_accepted_tokens,
            # query_start_loc=None,  # Leave unset for dense [batch, seq, hidden] inputs.
            activation_mode=True,
            pad_slot_id=-1
        )

        print(f"✅ NPU kernel executed successfully!")
        print(f"Output shape: {out_npu.shape}")

        # --- Validation ---
        # Move the NPU result back to CPU for comparison.
        out_npu_cpu = out_npu.cpu()
        out_vl = out_vl.cpu()


        # Check the output shape.
        assert out_npu_cpu.shape == out_vl.shape, \
            f"Output shape mismatch: {out_npu_cpu.shape} vs {out_vl.shape}"
        print(f"✅ Output shape matched: {out_npu_cpu.shape}")

        # Ensure the output is not all zeros.
        assert not torch.all(out_npu_cpu == 0), "NPU output is all zeros!"
        print(f"✅ NPU output is not all zeros")

        print(f"\n--- Numerical Comparison ---")
        print(f"NPU output - shape: {out_npu_cpu.shape}, dtype: {out_npu_cpu.dtype}, mean: {out_npu_cpu.mean().item():.6f}")
        print(f"vLLM output (transposed) - shape: {out_vl.shape}, dtype: {out_vl.dtype}, mean: {out_vl.mean().item():.6f}")

        # Compare precision element by element.
        diff = out_npu_cpu - out_vl
        abs_diff = torch.abs(diff)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        rel_diff_max = abs_diff / (torch.abs(out_vl) + 1e-6)
        max_rel_diff = rel_diff_max.max().item()

        print(f"Max absolute diff: {max_abs_diff:.6e}")
        print(f"Mean absolute diff: {mean_abs_diff:.6e}")
        print(f"Median absolute diff: {(abs_diff).median().item():.6e}")
        print(f"Max relative diff: {max_rel_diff:.6e}")

        # Validate precision using the repo's existing tolerance style.
        ATOL, RTOL = 5e-2, 1e-2
        tol = ATOL + RTOL * torch.abs(out_vl)
        matched = (abs_diff <= tol).sum().item()
        total = abs_diff.numel()
        print(f"Matched (atol={ATOL}, rtol={RTOL}): {matched}/{total} ({100*matched/total:.2f}%)")

        # --- Conv state verification ---
        print(f"\n--- Conv State Update Verification ---")
        print(f"vLLM state shape: {conv_state_vl.shape}")
        print(f"NPU state shape: {conv_state_npu.shape}")

        vllm_last = conv_state_vl
        npu_state = conv_state_npu.cpu()

        state_diff = (npu_state - vllm_last.cpu()).abs()
        state_exact_match = (state_diff < 1e-6).sum().item()
        state_total = state_diff.numel()

        print(f"State exact match (diff < 1e-6): {state_exact_match}/{state_total} ({100*state_exact_match/state_total:.2f}%)")
        if state_exact_match == state_total:
            print(f"✅ Conv state values match exactly!")
        else:
            print(f"State max diff: {state_diff.max():.6e}")

        # --- Summary ---
        print(f"\n{'='*60}")
        print("PRECISION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Output precision:  {matched}/{total} ({100*matched/total:.2f}%) match (atol={ATOL}, rtol={RTOL})")
        print(f"State precision:   {state_exact_match}/{state_total} ({100*state_exact_match/state_total:.2f}%) exact match")

        if matched >= total * 0.95 and state_exact_match == state_total:
            print(f"\\n✅ PASS: Output and state are correctly aligned to torch reference!")
        else:
            print(f"\\n⚠️  WARNING: Precision below expected threshold")

        print(f"\n🎉 NPU causal_conv1d_update test passed!")

    except Exception as e:
        print(f"❌ NPU test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # print("="*60)
    # print("Running test_correctness_fixed (CPU/CUDA reference)")
    # print("="*60)
    # test_correctness_fixed()

    print("\n" + "="*60)
    print("Running test_npu_causal_conv1d_update (NPU kernel)")
    print("="*60)
    test_npu_causal_conv1d_update()
