# fused_scatter_copy_sparse_flash_attention_split

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Verified on Ascend A3 | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.attention import (
    fused_scatter_copy_sparse_flash_attention_split,
)
```

```python
def fused_scatter_copy_sparse_flash_attention_split(
    query, query_rope, hbm_ckv, hbm_rope, actual_seq_lengths_query,
    actual_seq_lengths_kv, num_cache_tokens, topk_src_ids, topk_dst_slots,
    topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, dram_kv_dev_ptr, attention_out,
    scale_value, *, full_visible=False, resident_refresh_tail=False,
    current=None, positions=None, valid=None, request_slots=None,
    req_to_token=None, resident_end=None, host_token_capacity=0,
    dram_rope_dev_ptr=0, impl="auto",
) -> None:
    ...
```

## Description

This operator consumes a LIM plan, copies missing KV to the split HBM cache and
computes sparse attention. `impl="auto"` selects `"ascendc"` on ARCH22 with the
native operator included in the wheel.

## Parameters and return value

| Arguments | Contract |
| --- | --- |
| `query`, `query_rope`, `attention_out` | FP16 or BF16 TND tensors, shapes `[T,H,512]`, `[T,H,64]`, `[T,H,512]`. |
| `hbm_ckv`, `hbm_rope` | Mutable paged caches, shapes `[P,128,1,512]` and `[P,128,1,64]`, with the query dtype. |
| `actual_seq_lengths_query`, `actual_seq_lengths_kv`, `num_cache_tokens` | Int32 `[B]` cumulative query ends, request-local KV lengths and cache capacities. |
| `topk_src_ids`, `topk_dst_slots`, `topk_miss_counts` | Int32 `[T,2048]`, `[T,2048]` and `[T]` query plans. |
| `miss_src_ids`, `miss_dst_slots`, `miss_counts` | Int32 `[B,M]`, `[B,M]` and `[B]` request miss plans. |
| `hbm_block_table`, `dram_block_table` | Int32 page tables with one row per request. |
| `dram_kv_dev_ptr`, `dram_rope_dev_ptr` | Registered NPU-accessible addresses for CKV and KPE in host memory. |
| `scale_value` | Attention scale. |
| `full_visible`, `resident_refresh_tail` | Full-visible mode and optional overlap for a LIM refresh suffix. Both default to `False`. |
| `current`, `positions`, `valid`, `request_slots`, `req_to_token`, `resident_end`, `host_token_capacity` | Optional full-visible resident staging group. See the native contract for shapes and bounds. |

The operator returns `None` and mutates the HBM caches, `attention_out` and the
optional `resident_end`. See the [native contract](../../../../../csrc/attention/fused_scatter_copy_sparse_flash_attention/README.md)
and [independent attention and replay tests](../../../../../tests/python/operators/attention/test_copy_sfa_split_hbm_resident.py).

## ascendc backend

Uses the wheel’s native split Copy-SFA binding. The correctness suite allocates registered host buffers and compares the output with independent PyTorch attention.
The reference computes expected KV updates and attention for verification. The public `auto` and `ascendc` implementations use registered host addresses and update resident buffers in place.

## Runnable example

```python
import ctypes
import math
import mmap
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.operators.attention import fused_scatter_copy_sparse_flash_attention_split


torch.npu.set_device(0)
acl = ctypes.CDLL("libascendcl.so")
acl.aclrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
acl.aclrtHostRegister.restype = ctypes.c_int
acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
acl.aclrtHostUnregister.restype = ctypes.c_int
registered = []

def mapped_host(shape, dtype):
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    memory = mmap.mmap(-1, size)
    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    device_ptr = ctypes.c_void_p()
    status = acl.aclrtHostRegister(address, size, 0, ctypes.byref(device_ptr))
    if status:
        memory.close()
        raise RuntimeError(f"aclrtHostRegister failed: {status}")
    registered.append((address, memory))
    return torch.frombuffer(memory, dtype=dtype).view(shape), device_ptr.value

device, dtype = "npu:0", torch.bfloat16
npu = lambda value: value.to(device)
try:
    host_k, host_ptr = mapped_host((2048, 512), dtype)
    host_rope, rope_ptr = mapped_host((2048, 64), dtype)
    host_k.fill_(1)
    host_rope.fill_(1)
    hbm_k = torch.zeros((16, 128, 1, 512), dtype=dtype, device=device)
    hbm_rope = torch.zeros((16, 128, 1, 64), dtype=dtype, device=device)
    query = torch.zeros((1, 8, 512), dtype=dtype, device=device)
    query_rope = torch.zeros((1, 8, 64), dtype=dtype, device=device)
    output = torch.empty_like(query)
    zero = npu(torch.zeros(1, dtype=torch.int32))
    src = npu(torch.full((1, 2048), -1, dtype=torch.int32))
    dst_cpu = torch.full((1, 2048), -1, dtype=torch.int32)
    dst_cpu[0, :130] = torch.arange(130, dtype=torch.int32)
    dst = npu(dst_cpu)
    misses = npu(torch.full((1, 32768), -1, dtype=torch.int32))
    req_to_token = npu(torch.arange(2048, dtype=torch.int32).repeat(2, 1))
    resident_end = npu(torch.tensor([-1, -1], dtype=torch.int32))
    fused_scatter_copy_sparse_flash_attention_split(
        query, query_rope, hbm_k, hbm_rope, npu(torch.tensor([1], dtype=torch.int32)),
        npu(torch.tensor([130], dtype=torch.int32)), zero, src, dst, zero,
        misses, misses, zero, npu(torch.arange(16, dtype=torch.int32).view(1, -1)),
        npu(torch.zeros((1, 16), dtype=torch.int32)), host_ptr, output, 1 / math.sqrt(576),
        full_visible=True, current=torch.ones((1, 1, 576), dtype=dtype, device=device),
        positions=npu(torch.tensor([129], dtype=torch.int64)), valid=npu(torch.tensor([True])),
        request_slots=npu(torch.tensor([1], dtype=torch.int32)), req_to_token=req_to_token,
        resident_end=resident_end, host_token_capacity=2048, dram_rope_dev_ptr=rope_ptr,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.ones_like(output.cpu()), atol=0.003, rtol=0.02)
    assert resident_end.cpu().tolist() == [-1, 130]
finally:
    torch.npu.synchronize()
    for address, memory in registered:
        assert acl.aclrtHostUnregister(address) == 0
```
