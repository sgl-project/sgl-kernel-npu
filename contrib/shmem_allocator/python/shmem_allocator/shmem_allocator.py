import ctypes
import os
from pathlib import Path

import torch
import torch_npu


lib_dir = Path(__file__).resolve().parent / "lib"
shmem_allocator_module_path = list(lib_dir.glob("libshmem_allocator.*.so"))[0]


def switch_to_shmem_allocator():
    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(shmem_allocator_module_path, "my_malloc", "my_free")
    # Swap the current allocator
    torch_npu.npu.memory.change_current_allocator(new_alloc)
    myallocator = ctypes.CDLL(shmem_allocator_module_path)

    init_fn = ctypes.cast(getattr(myallocator, "my_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(myallocator, "my_empty_cache"), ctypes.c_void_p).value
    begin_allocate_to_pool_fn = ctypes.cast(getattr(myallocator, "my_begin_allocate_to_pool"), ctypes.c_void_p).value
    end_allocate_to_pool_fn = ctypes.cast(getattr(myallocator, "my_end_allocate_to_pool"), ctypes.c_void_p).value
    release_pool_fn = ctypes.cast(getattr(myallocator, "my_release_pool"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_begin_allocate_to_pool_fn(begin_allocate_to_pool_fn)
    new_alloc.allocator().set_end_allocate_to_pool_fn(end_allocate_to_pool_fn)
    new_alloc.allocator().set_release_pool_fn(release_pool_fn)


def init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port):
    myallocator = ctypes.CDLL(shmem_allocator_module_path)
    # 设置函数原型
    myallocator.init_shmem.argtypes = [
        ctypes.c_int,      # my_rank
        ctypes.c_int,      # n_ranks
        ctypes.c_uint64,   # local_mem_size
        ctypes.c_uint64,   # meta_size
        ctypes.c_char_p    # ip_port
    ]
    myallocator.init_shmem.restype = None

    myallocator.init_shmem(
        ctypes.c_int(my_rank),                    # my_rank
        ctypes.c_int(n_ranks),                    # n_ranks
        ctypes.c_uint64(local_mem_size),          # local_mem_size
        ctypes.c_uint64(meta_size),               # meta_size
        ip_port.encode('utf-8')                   # ip_port
    )
