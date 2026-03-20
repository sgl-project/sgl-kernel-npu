"""
测试 fusion (融合) 算子 vs atomic (小算子) 的精度对比
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import torch_npu
import sgl_kernel_npu

def vllm_causal_conv1d_update_v3(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """vLLM风格的atomic算子实现 - 在指定设备上运行"""
    hidden_state = hidden_state.transpose(1, 2)
    weight = weight.transpose(0, 1)
    conv_state = conv_state.transpose(1, 2)
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]

    target_state_len = (kernel_size - 1) + (seq_len - 1)

    full_context = torch.cat([conv_state[conv_state_indices.long()], hidden_state], dim=-1).to(weight.dtype)

    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)

    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(hidden_state.dtype)

    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype)
    new_conv_state = new_conv_state.transpose(1, 2)
    return out, new_conv_state


def test_fusion_vs_atomic():
    """测试融合算子 vs 原子算子的精度对比"""
    if not (hasattr(torch, 'npu') and torch.npu.device_count() > 0):
        print("⚠️  NPU not available")
        return

    # 配置
    BSZ = 4
    HIDDEN_SIZE = 1024
    SEQ_LEN = 2
    KERNEL_SIZE = 3
    CACHE_LEN = 10
    DTYPE = torch.bfloat16
    DEVICE = "npu"

    print("=" * 70)
    print("Fusion vs Atomic 算子对比测试 (NPU)")
    print("=" * 70)
    print(f"配置: BSZ={BSZ}, HIDDEN_SIZE={HIDDEN_SIZE}, SEQ_LEN={SEQ_LEN}, KERNEL_SIZE={KERNEL_SIZE}")
    print(f"设备: {DEVICE}, 数据类型: {DTYPE}")
    print()

    # 固定种子确保可重复性
    torch.manual_seed(42)

    # 创建测试数据 (CPU)
    weight_cpu = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
    bias_cpu = torch.randn(HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
    hidden_state_cpu = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.5
    conv_state_init_cpu = torch.randn(CACHE_LEN, KERNEL_SIZE - 1, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.3
    conv_state_indices_cpu = torch.arange(BSZ, device='cpu', dtype=torch.int32)

    print("=" * 70)
    print("方法 1: Fusion 算子 (融合的NPU内核)")
    print("=" * 70)

    # Fusion 算子 - 使用 clone 确保不修改原始数据
    weight_fusion = weight_cpu.clone().to(DEVICE)
    bias_fusion = bias_cpu.clone().to(DEVICE)
    hidden_state_fusion = hidden_state_cpu.clone().to(DEVICE)
    conv_state_fusion = conv_state_init_cpu.clone().to(DEVICE)
    conv_state_indices_fusion = conv_state_indices_cpu.clone().to(DEVICE)

    # 执行融合算子
    out_fusion = torch.ops.npu.causal_conv1d_update(
        x=hidden_state_fusion,
        weight=weight_fusion,
        conv_state=conv_state_fusion,
        conv_state_indices=conv_state_indices_fusion,
        bias=bias_fusion,
        num_accepted_tokens=torch.empty(0, device=DEVICE, dtype=torch.int32),
        query_start_loc=torch.empty(0, device=DEVICE, dtype=torch.int32),
        activation_mode=True,
        pad_slot_id=-1
    )

    print(f"✓ Fusion 算子执行成功")
    print(f"  输出形状: {out_fusion.shape}")
    print(f"  输出设备: {out_fusion.device}")

    print()
    print("=" * 70)
    print("方法 2: Atomic 算子 (PyTorch原子操作)")
    print("=" * 70)

    # Atomic 算子 - 在NPU上逐步执行
    weight_atomic = weight_cpu.clone().to(DEVICE)
    bias_atomic = bias_cpu.clone().to(DEVICE)
    hidden_state_atomic = hidden_state_cpu.clone().to(DEVICE)
    conv_state_atomic = conv_state_init_cpu.clone().to(DEVICE)
    conv_state_indices_atomic = conv_state_indices_cpu.clone().to(DEVICE)

    # 执行原子算子 (在NPU上)
    out_atomic, _ = vllm_causal_conv1d_update_v3(
        hidden_state=hidden_state_atomic,
        conv_state=conv_state_atomic,
        weight=weight_atomic,
        bias=bias_atomic,
        conv_state_indices=conv_state_indices_atomic,
        activation=True
    )

    print(f"✓ Atomic 算子执行成功")
    print(f"  输出形状: {out_atomic.shape}")
    print(f"  输出设备: {out_atomic.device}")

    print()
    print("=" * 70)
    print("精度对比分析")
    print("=" * 70)

    # 转换到CPU进行比较
    out_fusion_cpu = out_fusion.cpu()
    out_atomic_cpu = out_atomic.cpu()

    # 基本统计信息
    print("\n统计信息对比:")
    print(f"{'指标':<20} {'Fusion算子':<15} {'Atomic算子':<15} {'差异':<10}")
    print("-" * 60)
    print(f"{'均值 (mean)':<20} {out_fusion_cpu.mean():<15.6f} {out_atomic_cpu.mean():<15.6f} {abs(out_fusion_cpu.mean() - out_atomic_cpu.mean()):<10.6f}")
    print(f"{'标准差 (std)':<20} {out_fusion_cpu.std():<15.6f} {out_atomic_cpu.std():<15.6f} {abs(out_fusion_cpu.std() - out_atomic_cpu.std()):<10.6f}")
    print(f"{'最小值 (min)':<20} {out_fusion_cpu.min():<15.6f} {out_atomic_cpu.min():<15.6f} {abs(out_fusion_cpu.min() - out_atomic_cpu.min()):<10.6f}")
    print(f"{'最大值 (max)':<20} {out_fusion_cpu.max():<15.6f} {out_atomic_cpu.max():<15.6f} {abs(out_fusion_cpu.max() - out_atomic_cpu.max()):<10.6f}")

    # 直接误差分析
    abs_error = torch.abs(out_fusion_cpu - out_atomic_cpu)
    print("\n误差统计:")
    print(f"  最大绝对误差: {abs_error.max():.8f}")
    print(f"  平均绝对误差: {abs_error.mean():.8f}")
    print(f"  中位数绝对误差: {abs_error.median():.8f}")
    print(f"  RMS误差: {torch.mean((out_fusion_cpu - out_atomic_cpu) ** 2).sqrt():.8f}")

    # 最大误差位置分析
    max_error_idx = torch.argmax(abs_error)
    fusion_val = out_fusion_cpu.flatten()[max_error_idx].item()
    atomic_val = out_atomic_cpu.flatten()[max_error_idx].item()
    print(f"\n最大误差位置: {max_error_idx.item()}")
    print(f"  Fusion值: {fusion_val:.10f}")
    print(f"  Atomic值: {atomic_val:.10f}")
    print(f"  误差: {abs(fusion_val - atomic_val):.10f}")

    # 容差测试
    print("\n容差测试:")
    for rtol, atol in [(0.01, 0.003), (0.005, 0.001), (0.002, 0.0005), (0.001, 0.0003)]:
        match = torch.allclose(out_fusion_cpu, out_atomic_cpu, rtol=rtol, atol=atol)
        match_count = torch.sum(abs_error <= atol + rtol * torch.abs(out_atomic_cpu)).item()
        pct = 100 * match_count / out_atomic_cpu.numel()
        status = "✅" if match else "❌"
        total = out_atomic_cpu.numel()
        print(f"  {status} rtol={rtol:}, atol={atol:}: ({pct:6.2f}%) = {match_count}/{total}")

    # 逐元素不匹配统计
    ATOL_STRICT = 0.0003
    mismatch_mask = abs_error > ATOL_STRICT
    mismatch_count = mismatch_mask.sum().item()
    if mismatch_count > 0:
        print(f"\n不匹配元素 (误差 > {ATOL_STRICT}): {mismatch_count}/{out_atomic_cpu.numel()} ({100*mismatch_count/out_atomic_cpu.numel():.4f}%)")

        # 按batch和seq统计
        mismatch_by_batch_seq = torch.sum(mismatch_mask.view(BSZ, SEQ_LEN, HIDDEN_SIZE), dim=2)
        print("\n每个 (batch, seq) 位置的不匹配元素数:")
        for b in range(BSZ):
            for s in range(SEQ_LEN):
                count = mismatch_by_batch_seq[b, s].item()
                if count > 0:
                    print(f"  batch={b}, seq={s}: {count}/{HIDDEN_SIZE} ({100*count/HIDDEN_SIZE:.2f}%)")
    else:
        print(f"\n✅ 所有元素误差都在 {ATOL_STRICT} 以内！")

    # 样本值对比
    print("\n样本值对比 (每个batch第一个seq的前10个维度):")
    for b in range(BSZ):
        print(f"\nBatch {b}, Seq 0:")
        for i in range(min(10, HIDDEN_SIZE)):
            fusion_val = out_fusion_cpu[b, 0, i].item()
            atomic_val = out_atomic_cpu[b, 0, i].item()
            diff = abs(fusion_val - atomic_val)
            symbol = "✓" if diff <= 0.0005 else "✗" if diff > 0.001 else "~"
            print(f"    dim {i:3d}: Fusion={fusion_val:10.6f}, Atomic={atomic_val:10.6f}, diff={diff:8.6f} {symbol}")

    print()
    print("=" * 70)
    if torch.allclose(out_fusion_cpu, out_atomic_cpu, rtol=0.01, atol=0.003):
        print("✅ Fusion算子与Atomic算子精度匹配！")
    else:
        print("⚠️  Fusion算子与Atomic算子存在精度差异")
    print("=" * 70)


if __name__ == "__main__":
    test_fusion_vs_atomic()
