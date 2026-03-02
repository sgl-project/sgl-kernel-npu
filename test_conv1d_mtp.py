"""
测试 causal_conv1d_update 在 MTP (Multi-Token Prediction) 场景下的 num_accepted_tokens 参数
"""
import torch
import torch.nn.functional as F
from typing import Optional

# 设定随机种子，保证结果可复现
torch.manual_seed(42)

# ==========================================
# vLLM 风格实现 (用于参考)
# ==========================================
def vllm_causal_conv1d_update_with_accept(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
    num_accepted: Optional[int] = None,
) -> torch.Tensor:
    """
    模拟 MTP 场景下的因果卷积更新
    num_accepted: 有多少个 token 被接受
    """
    hidden_state = hidden_state.transpose(1, 2)
    weight = weight.transpose(0, 1)
    conv_state = conv_state.transpose(1, 2)

    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]
    orig_state_len = conv_state.shape[-1]

    # 如果没有 num_accepted，默认全接受
    if num_accepted is None:
        num_accepted = seq_len

    # 逻辑: (K-1) + (accepted - 1)
    target_state_len = (kernel_size - 1) + (num_accepted - 1)

    full_context = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)

    # 计算 output - 使用所有生成的 token
    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)
    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(hidden_state.dtype)

    # 更新 State: 只保留被接受的 token 对应的状态
    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype)

    # 更新到新的 conv_state 张量（返回出来）
    final_state = new_conv_state.transpose(1, 2)
    out = out.transpose(1, 2)

    return out, final_state


# ==========================================
# 测试不同 num_accepted 值
# ==========================================
def test_num_accepted_tokens_scenario():
    """
    测试不同的 num_accepted 值对结果的影响
    """
    print("="*70)
    print("Testing num_accepted_tokens parameter in MTP scenario")
    print("="*70)

    # 配置
    BSZ = 1
    HIDDEN_SIZE = 256
    SEQ_LEN = 5  # 生成了 5 个候选 token
    KERNEL_SIZE = 4
    STATE_LEN = KERNEL_SIZE - 1 + SEQ_LEN - 1  # 3 + 4 = 7
    DTYPE = torch.float32  # 使用 float32 以便精确比较
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfiguration:")
    print(f"  BSZ = {BSZ}")
    print(f"  HIDDEN_SIZE = {HIDDEN_SIZE}")
    print(f"  SEQ_LEN (generated tokens) = {SEQ_LEN}")
    print(f"  KERNEL_SIZE = {KERNEL_SIZE}")
    print(f"  STATE_LEN = {STATE_LEN}")
    print(f"  DTYPE = {DTYPE}")
    print(f"  DEVICE = {DEVICE}")

    # 初始化参数
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_init = torch.randn(BSZ, STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    print(f"\n{'='*70}")
    print("Scenario Analysis")
    print(f"{'='*70}")

    # 测试不同的 num_accepted 值
    test_cases = [
        (1, "Only 1 token accepted (worst case)"),
        (2, "2 tokens accepted"),
        (3, "3 tokens accepted (60%)"),
        (5, "All 5 tokens accepted (best case - no MTP penalty)"),
    ]

    for num_accepted, description in test_cases:
        print(f"\n--- Test: num_accepted = {num_accepted} ({description}) ---")

        # 复制初始状态
        conv_state = conv_state_init.clone()

        # 执行卷积更新
        out, final_state = vllm_causal_conv1d_update_with_accept(
            hidden_state=hidden_state,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=True,
            num_accepted=num_accepted,
        )

        print(f"  Output shape: {out.shape}")
        print(f"  Final state shape: {final_state.shape}")

        # 计算期望的 state 长度
        expected_state_len = (KERNEL_SIZE - 1) + (num_accepted - 1)
        print(f"  Expected state length: {expected_state_len}")
        print(f"  Actual state length: {final_state.shape[-1]}")

        # 验证输出总是有 SEQ_LEN 个
        assert out.shape[1] == SEQ_LEN, f"Output seq_len mismatch: {out.shape[1]} vs {SEQ_LEN}"

        # 验证 state 长度符合预期
        assert final_state.shape[-1] == expected_state_len, \
            f"State length mismatch: {final_state.shape[-1]} vs {expected_state_len}"

        print(f"  Check output shape: (batch={BSZ}, seq={SEQ_LEN}, dim={HIDDEN_SIZE})")
        print(f"  Check state shape: (batch={BSZ}, seq={expected_state_len}, dim={HIDDEN_SIZE})")
        print(f"  State contains: [{KERNEL_SIZE-1} history] + [{num_accepted-1} accepted] tokens")

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("Key findings:")
    print("1. num_accepted_tokens controls how many candidate tokens are 'kept'")
    print("2. Output ALWAYS has SEQ_LEN tokens (all generated tokens are computed)")
    print("3. Conv state length = (KERNEL_SIZE-1) + (num_accepted-1)")
    print("4. Lower num_accepted = more wasted computation (candidates rejected)")
    print(f"{'='*70}\n")


# ==========================================
# 测试 NPU 算子
# ==========================================
def test_npu_with_num_accepted():
    """
    测试 NPU 算子的 num_accepted_tokens 参数
    """
    try:
        import torch_npu
    except ImportError:
        print("⚠️  torch_npu not available, skipping NPU test")
        return

    try:
        import sgl_kernel_npu
    except ImportError:
        print("⚠️  sgl_kernel_npu not available, skipping NPU test")
        return

    print("\n" + "="*70)
    print("Testing NPU causal_conv1d_update with num_accepted_tokens")
    print("="*70)

    # 配置
    BSZ = 1
    HIDDEN_SIZE = 4096
    SEQ_LEN = 5  # MTP 场景，生成 5 个候选 token
    KERNEL_SIZE = 4
    STATE_LEN = KERNEL_SIZE - 1 + SEQ_LEN - 1  # 3 + 4 = 7
    DTYPE = torch.bfloat16
    DEVICE = "npu"

    print(f"\nConfiguration:")
    print(f"  BSZ = {BSZ}")
    print(f"  SEQ_LEN (candidates) = {SEQ_LEN}")
    print(f"  KERNEL_SIZE = {KERNEL_SIZE}")
    print(f"  num_accepted_tokens = 3 (3 tokens accepted, 2 rejected)")

    # 参数
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # NPU 算子需要更大的 conv_state buffer
    # 对于 MTP 场景，需要 (width-1) + (seq_len-1) 的空间
    conv_state = torch.randn(10, STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int32)

    # num_accepted_tokens 指示有多少个 token 被接受
    # 这是一个形状为 (batch,) 的张量
    num_accepted_tokens = torch.tensor([3], device=DEVICE, dtype=torch.int32)

    try:
        print(f"\nCalling NPU op with:")
        print(f"  x shape: {hidden_state.shape}")
        print(f"  weight shape: {weight.shape}")
        print(f"  conv_state shape: {conv_state.shape}")
        print(f"  conv_state_indices: {conv_state_indices}")
        print(f"  num_accepted_tokens: {num_accepted_tokens} (shape: {num_accepted_tokens.shape})")

        # 第一个问题：查询 NPU 算子的签名
        import inspect
        sig = inspect.signature(torch.ops.npu.causal_conv1d_update)
        print(f"\nNPU op signature: {sig}")

        # 调用 NPU 算子
        out = torch.ops.npu.causal_conv1d_update(
            x=hidden_state,
            weight=weight,
            conv_state=conv_state,
            conv_state_indices=conv_state_indices,
            bias=bias,
            num_accepted_tokens=num_accepted_tokens,  # 这是一个 (batch,) 的 tensor
            activation_mode=True,
            # 注意：当使用 num_accepted_tokens 时，可能还需要 query_start_loc
            # 但这取决于具体实现
            pad_slot_id=-1,
        )

        print(f"\n✅ NPU execution succeeded!")
        print(f"  Output shape: {out.shape}")
        print(f"  Output mean: {out.mean().item():.6f}")

        # 验证输出形状
        assert out.shape == (BSZ, SEQ_LEN, HIDDEN_SIZE), \
            f"Output shape mismatch: {out.shape} vs ({BSZ}, {SEQ_LEN}, {HIDDEN_SIZE})"

    except Exception as e:
        print(f"\n❌ NPU execution failed:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 先运行参考实现，了解 behavior
    test_num_accepted_tokens_scenario()

    # 然后测试 NPU 算子
    test_npu_with_num_accepted()
