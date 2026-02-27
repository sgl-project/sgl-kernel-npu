"""
测试不同维度下 NPU Fusion 算子 vs NPU Atomic 算子的精度对比
"""
import torch
import torch.nn.functional as F
import torch_npu
import sgl_kernel_npu

def compute_atomic_npu(x, conv_state, weight, bias, indices, seq_len, kernel_size):
    """在NPU上逐步计算causal_conv1d"""
    # x: [BSZ, SEQ_LEN, HIDDEN_SIZE]
    # conv_state: [CACHE_LEN, KERNEL_SIZE-1, HIDDEN_SIZE] -> [BSZ, HIDDEN_SIZE, KERNEL_SIZE-1]
    # weight: [KERNEL_SIZE, HIDDEN_SIZE]

    # 构建 full context
    conv_t = conv_state.transpose(1, 2)  # [BSZ, HIDDEN_SIZE, KERNEL_SIZE-1]
    hidden_t = x.transpose(1, 2)  # [BSZ, HIDDEN_SIZE, SEQ_LEN]

    # 根据indices选择对应的conv_state
    conv_selected = conv_t[indices.long()]  # [BSZ, HIDDEN_SIZE, KERNEL_SIZE-1]

    # 拼接: [BSZ, HIDDEN_SIZE, KERNEL_SIZE-1] + [BSZ, HIDDEN_SIZE, SEQ_LEN] = [BSZ, HIDDEN_SIZE, KERNEL_SIZE-1+SEQ_LEN]
    full_context = torch.cat([conv_selected, hidden_t], dim=-1).to(weight.dtype)

    # 计算滑动窗口
    windows = full_context.unfold(-1, kernel_size, 1)  # [BSZ, HIDDEN_SIZE, new_seq_len, kernel_size]

    # 卷积计算
    out_t = (windows * weight.transpose(0, 1).view(1, 1, -1, kernel_size)).sum(dim=-1)  # [BSZ, HIDDEN_SIZE, new_seq_len]

    # 加 bias (注意 bias的形状)
    out_t = out_t + bias.unsqueeze(2)  # [BSZ, HIDDEN_SIZE, new_seq_len]

    # SiLU激活
    out_t = F.silu(out_t)  # [BSZ, HIDDEN_SIZE, new_seq_len]

    # 转换回 [BSZ, SEQ_LEN, HIDDEN_SIZE]
    out = out_t.transpose(1, 2).to(x.dtype)

    return out

def test_all_dimensions():
    """测试不同的维度"""
    if not (hasattr(torch, 'npu') and torch.npu.device_count() > 0):
        print("⚠️ NPU not available")
        return

    BSZ = 4
    SEQ_LEN = 2
    KERNEL_SIZE = 3
    CACHE_LEN = 10
    DEVICE = 'npu'
    DTYPE = torch.bfloat16

    print('='*80)
    print('NPU Fusion vs NPU Atomic 算子精度对比 - 不同维度测试')
    print('='*80)

    for HIDDEN_SIZE in [64, 128, 256, 512, 1024]:
        torch.manual_seed(42)

        # 创建测试数据
        weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
        bias = torch.randn(HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
        conv_state = torch.randn(CACHE_LEN, KERNEL_SIZE - 1, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.3
        x = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.5
        indices = torch.arange(BSZ, device='cpu', dtype=torch.int32)

        # Fusion 算子
        out_fusion = torch.ops.npu.causal_conv1d_update(
            x=x.clone().to(DEVICE),
            weight=weight.clone().to(DEVICE),
            conv_state=conv_state.clone().to(DEVICE),
            conv_state_indices=indices.clone().to(DEVICE),
            bias=bias.clone().to(DEVICE),
            num_accepted_tokens=torch.empty(0, device=DEVICE, dtype=torch.int32),
            query_start_loc=torch.empty(0, device=DEVICE, dtype=torch.int32),
            activation_mode=True,
            pad_slot_id=-1
        )

        # Atomic 算子 (NPU)
        out_atomic = compute_atomic_npu(
            x.clone().to(DEVICE),
            conv_state.clone().to(DEVICE),
            weight.clone().to(DEVICE),
            bias.clone().to(DEVICE),
            indices.clone().to(DEVICE),
            SEQ_LEN,
            KERNEL_SIZE
        )

        # 计算误差
        abs_error = torch.abs(out_fusion.cpu() - out_atomic.cpu())
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        std_error = abs_error.std().item()

        # 容差测试
        tolerance_001_count = (abs_error <= 0.001).sum().item()
        tolerance_003_count = (abs_error <= 0.003).sum().item()
        tolerance_01_count = (abs_error <= 0.01).sum().item()
        total = abs_error.numel()

        print(f"HIDDEN_SIZE={HIDDEN_SIZE:4d}: ", end="")
        print(f"max_err={max_error:.6f}, ", end="")
        print(f"mean_err={mean_error:.6f}, ", end="")
        print(f"std_err={std_error:.6f}, ", end="")
        print(f"match_0.001={tolerance_001_count:5d}/{total} ({100*tolerance_001_count/total:5.1f}%)")

    print()
    print('='*80)
    print("结论: NPU Fusion算子与NPU Atomic算子在不同维度下的精度对比")
    print("="*80)

if __name__ == "__main__":
    test_all_dimensions()
