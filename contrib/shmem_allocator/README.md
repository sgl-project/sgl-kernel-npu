# Torch SHMEM Allocator

A PyTorch pluggable allocator built on [ascend shmem library](https://gitee.com/ascend/shmem).

## Build & Install shmem allocator

- source cann<br>
  ```bash
  # Assuming cann is installed in /usr/local/Ascend/ascend-toolkit
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

- install shmem library<br>
  Please refer to shmem's [README.md](https://gitee.com/ascend/shmem/blob/master/README.md#%E4%B8%89%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B)

- source shmem `set_env.sh` file<br>
  ```bash
  # Assuming shmem is installed in /usr/local/Ascend/shmem
  source /usr/local/Ascend/shmem/latest/set_env.sh
  ```
  
- build shmem allocator
  ```bash
  # Firstly, change current dir to sgl-kernel-npu project root
  cd sgl-kernel-npu
  # Then, build shmem allocator by build.sh
  bash build.sh -a shmem-allocator 
  ```

- install shmem allocator
  ```bash
  pip install output/shmem_allocator-*.whl
  ```

## Use shmem allocator in sglang

Shmem allocator provide two python api for users.<br>
- `switch_to_shmem_allocator()`: Switch pytorch's allocator to shmem allocator.

- `init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port)`: Init underlying shmem library.<br>
   **Parameters:**
    * `my_rank`: rank of current process.
    * `n_ranks`: global world size.
    * `local_mem_size`: shmem pool size to be pre-allocated on each NPU.
    * `meta_size`: the portion of `local_mem_size` that guarantees symmetric allocation across all NPUs.
    * `ip_port`: ip:port for inter-PE bootstrap and synchronization.
   

### Usage example
In sglang's `sglang/srt/managers/scheduler.py`, change npu allocator of scheduler process to shmem allocator as soon as scheduler starts.
```diff
...
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
+   from shmem_allocator import switch_to_shmem_allocator
+   switch_to_shmem_allocator()
    # Generate the logger prefix
    prefix = ""
...
```

In sglang's `sglang/srt/model_executor/model_runner.py`, init shmem in scheduler process exactly after `torch.set_device(idx)` is called.
```diff
...
def init_torch_distributed(self):
    logger.info("Init torch distributed begin.")

    try:
        torch.get_device_module(self.device).set_device(self.gpu_id)
+       from shmem_allocator import init_shmem
+       init_shmem(self.tp_rank, self.tp_size, 39 * (1024 ** 3), 0, 'tcp://127.0.0.1:3366')
    except Exception:
...
```
