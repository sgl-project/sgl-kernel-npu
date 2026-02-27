import torch
import torch.nn.functional as F
import torch_npu
import sgl_kernel_npu

BSZ = 4
HIDDEN_SIZE = 64  # 减小维度以便观察
SEQ_LEN = 2
KERNEL_SIZE = 3
CACHE_LEN = 10
DTYPE = torch.bfloat16
DEVICE = 'npu'

torch.manual_seed(42)

weight_cpu = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
bias_cpu = torch.randn(HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.1
hidden_state_cpu = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.5
conv_state_init_cpu = torch.randn(CACHE_LEN, KERNEL_SIZE - 1, HIDDEN_SIZE, device='cpu', dtype=DTYPE) * 0.3
conv_state_indices_cpu = torch.arange(BSZ, device='cpu', dtype=torch.int32)

# Fusion算子
weight_f = weight_cpu.clone().to(DEVICE)
bias_f = bias_cpu.clone().to(DEVICE)
hidden_f = hidden_state_cpu.clone().to(DEVICE)
conv_f = conv_state_init_cpu.clone().to(DEVICE)
idx_f = conv_state_indices_cpu.clone().to(DEVICE)

out_fusion = torch.ops.npu.causal_conv1d_update(
    x=hidden_f, weight=weight_f, conv_state=conv_f, conv_state_indices=idx_f, bias=bias_f,
    num_accepted_tokens=torch.empty(0, device=DEVICE, dtype=torch.int32),
    query_start_loc=torch.empty(0, device=DEVICE, dtype=torch.int32),
    activation_mode=True, pad_slot_id=-1
)

# Atomic算子
weight_a = weight_cpu.clone().to(DEVICE)
bias_a = bias_cpu.clone().to(DEVICE)
hidden_a = hidden_state_cpu.clone().to(DEVICE)
conv_a = conv_state_init_cpu.clone().to(DEVICE)
idx_a = conv_state_indices_cpu.clone().to(DEVICE)

# 手动atomic: conv_state [cache_len, state_len, hidden_size]
# 第一: 构造full_context
conv_a_t = conv_a.transpose(1, 2)  # [BSZ, HIDDEN_SIZE, state_len]
full_context = torch.cat([conv_a_t[idx_a.long()], hidden_a.transpose(1, 2)], dim=-1).to(weight_a.dtype)
# 第二: 计算窗口卷积
computation_input = full_context[:, :, -(KERNEL_SIZE - 1 + SEQ_LEN):]
windows = computation_input.unfold(-1, KERNEL_SIZE, 1)
out = (windows * weight_a.transpose(0, 1)[None, :, None, :]).sum(dim=-1)
# 第三: 加 bias
out = out + bias_a.unsqueeze(-1)
# 第四: Silu激活
out = F.silu(out)
out_atomic = out.transpose(1, 2).to(hidden_a.dtype)

out_f_cpu = out_fusion.cpu()
out_a_cpu = out_atomic.cpu()

abs_error = torch.abs(out_f_cpu - out_a_cpu)

print('='*60)
print('逐步诊断 - NPU Fusion vs NPU Atomic')
print('='*60)
print(f'数据规模: BSZ={BSZ}, HIDDEN_SIZE={HIDDEN_SIZE}, SEQ_LEN={SEQ_LEN}')
print()

# 找出最大错误的位置
max_error_idx = torch.argmax(abs_error)
max_error_val = abs_error.max()

batch_idx = max_error_idx // (SEQ_LEN * HIDDEN_SIZE)
seq_idx = (max_error_idx % (SEQ_LEN * HIDDEN_SIZE)) // HIDDEN_SIZE
dim_idx = max_error_idx % HIDDEN_SIZE

print(f'最大误差位置: batch={batch_idx}, seq={seq_idx}, dim={dim_idx}')
print(f'  Fusion值: {out_f_cpu[batch_idx, seq_idx, dim_idx]:.8f}')
print(f'  Atomic值: {out_a_cpu[batch_idx, seq_idx, dim_idx]:.8f}')
print(f'  误差: {max_error_val:.8f}')
print()

# 显示该batch该seq的所有元素对比
print(f'Batch {batch_idx}, Seq {seq_idx} 的所有维度对比:')
for d in range(HIDDEN_SIZE):
    f_val = out_f_cpu[batch_idx, seq_idx, d].item()
    a_val = out_a_cpu[batch_idx, seq_idx, d].item()
    err = abs(f_val - a_val)
    marker = '>>> ' if d == dim_idx else '    '
    status = '✗' if err > 0.01 else '✓' if err > 0.003 else ''
    print(f'{marker}Dim {d:2d}: Fusion={f_val:10.6f}, Atomic={a_val:10.6f}, Error={err:8.6f} {status}')

# 找出误差>0.01的元素
large_errors = (abs_error > 0.01).sum()
print(f'\n误差>0.01的元素: {large_errors}/{out_f_cpu.numel()}')

# 输入验证
print()
print('='*60)
print('输入数据验证 (batch 0, seq 0 的前5个维度):')
print('='*60)
for d in range(5):
    print(f'Dim {d}:')
    print(f'  hidden_state:  {hidden_state_cpu[0,0,d].item():.6f}')
    print(f'  conv_state[0]: {conv_state_init_cpu[0,0,d].item():.6f}')
    print(f'  conv_state[1]: {conv_state_init_cpu[0,1,d].item():.6f}')
    print(f'  weight[0]:     {weight_cpu[0,d].item():.6f}')
    print(f'  weight[1]:     {weight_cpu[1,d].item():.6f}')
    print(f'  weight[2]:     {weight_cpu[2,d].item():.6f}')
    print(f'  bias [{d}]:    {bias_cpu[d].item():.6f}')
    print()
