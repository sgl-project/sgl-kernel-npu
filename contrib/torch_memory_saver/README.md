# Torch Memory Saver

A PyTorch library that allows tensor memory to be temporarily released and resumed later.

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## Examples and Features

### Basic Example

```python
# 1. For tensors that wants to be paused, create them within `region`
with torch_memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='npu')

# 2. After `pause`, NPU memory is released for those tensors.
# For example, check `npu-smi info`'s memory usage to verify.
torch_memory_saver.pause()

# 3. After `resume`, NPU memory is re-occupied for those tensors.
torch_memory_saver.resume()
```

During the pause, physical memory is released and virtual address is preserved. When resume, virtual address is kept unchanged, while physical memory is re-allocated

### Multiple Tags

Please refer to https://github.com/sgl-project/sglang/issues/7009 for details.

```python
# 1. Create tensors with different tags
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='npu')

with torch_memory_saver.region(tag="type2"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='npu')

# 2. Pause and resume with different tags selectively
torch_memory_saver.pause("type1")
torch_memory_saver.pause("type2")

torch_memory_saver.resume("type2")
torch_memory_saver.resume("type1")

torch_memory_saver.pause("type1")
torch_memory_saver.resume("type1")
```

### CPU Backup

By default, in order to save time, the content is thrown away. This is useful for, for example, KV cache that are to be staled, or model weights that are to be updated.

If you want the tensor content to be kept unchanged, use `enable_cpu_backup`.

```python
with torch_memory_saver.region(enable_cpu_backup=True):
    tensor1 = torch.full((5_000_000_000,), 42, dtype=torch.uint8, device='npu')

torch_memory_saver.pause()
torch_memory_saver.resume()

assert tensor1[0] == 42, "content is kept unchanged"
```

### Hook Modes

There are two hook modes:

* **preload**: Use `LD_PRELOAD` to hook CANN's malloc and free API to change allocation behavior.
* **torch**: Use torch's custom allocator API to change allocation behavior.

The mode can be chosen by:

```python
torch_memory_saver.hook_mode = "torch"
```

## Development

1. Before executing the engineering build script build.sh, modify `_ASCEND_INSTALL_PATH` on line 73 of build.sh according to the CANN installation path.

```bash
bash build.sh  -a memory-saver
```
2. Pip install the `.whl` file into your Python environment

```bash
pip install output/deep_ep*.whl
```
## Test
You can use this command for local testing:

### Basic Functions
```bash
python contrib/torch_memory_saver/test/simple.py  torch
```

### CPU Backup
```bash
python contrib/torch_memory_saver/test/cpu_backup.py  torch
```

### RL_Example
```bash
python contrib/torch_memory_saver/test/rl_example.py  torch
```

### FAQ
1. Since PTA accesses aclrtMallocAlign32 via dlopen + dlsym, it is not possible to override this interface using LD_PRELOAD. As a result, the preload mode and the feature of releasing activation value memory in graph mode are currently unsupported. This functionality is under development.


