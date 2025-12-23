import random
import time
import unittest
import os
import sgl_kernel_npu
import torch
import torch_npu

import shmem as ash
import torch.distributed as dist

torch.set_printoptions(threshold=float("inf"))


g_ash_size = 2 * 1024 * 1024 * 1024
g_malloc_size = 1024 * 1024 * 1024
g_shmem_addr = None
g_team_size = 2


def direct_testing(input_a, input_b, input_c, team_id=0, group_list=()):
    global g_shmem_addr

    a, b = input_a, input_b
    # b_nz = torch_npu.npu_format_cast(b, 29)
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    l_world_size = int(world_size / len(group_list)) if len(group_list) > 0 else world_size

    # assert g_shmem_addr is not None and k == b.shape[0]
    # print('rank', rank, ' is ', a.device, b.device)
    for _ in range(1):
        with torch.npu.stream(torch.npu.current_stream()):
            output = torch.ops.npu.catcoc_allgather_matmul(a, b, input_c, g_shmem_addr, team_id)
        torch.npu.synchronize()

    if l_world_size > 1:
        native_a = [torch.empty_like(input_a) for _ in range(l_world_size)]
        if len(group_list) == 0:
            dist.all_gather(native_a, input_a)
        else:
            for dist_group in group_list:
                dist.all_gather(native_a, input_a, group=dist_group)
        native_a = torch.concat(native_a, dim=0)
    else:
        native_a = input_a

    native_c = torch.matmul(native_a, b)

    if rank == 0:
        print(output.shape, output.flatten()[:10], output.flatten()[-10:])
        print(rank, native_c.shape, native_c.flatten()[:10], native_c.flatten()[-10:])
    # time.sleep(rank * 5)
    torch.allclose(output, native_c, rtol=1e-3, atol=1e-3)
    print('rank', rank, ' success')


def shmem_init():
    from shmem import set_conf_store_tls

    set_conf_store_tls(False, "")
    init_attr = ash.InitAttr()
    shmem_addr = "tcp://127.0.0.1:26666"
    ash.shmem_set_attributes(rank, world_size, g_ash_size, shmem_addr, init_attr)
    ret = ash.shmem_init(init_attr)
    assert ret == 0, '[ERROR] aclshmem_init failed'
    global g_shmem_addr
    g_shmem_addr = ash.shmem_malloc(g_malloc_size)


def run_global_test(test_mnk=(1024, 1024, 1024)):
    test_m, test_n, test_k = test_mnk
    print('+++++++++++++++++++++++Testing NO QUANT...')
    a = torch.rand([test_m, test_k]).to(dtype=torch.float16).to(f"npu:{rank}")
    b = torch.rand([test_k, test_n]).to(dtype=torch.float16).to(f"npu:{rank}")
    c = torch.empty([test_m * world_size, test_n]).to(dtype=torch.float16).to(f"npu:{rank}")  # tmp tensor
    bias = None
    scale = None
    pertoken_scale = None

    assert torch.npu.current_device() == rank, f'[ERROR] device:{torch.npu.current_device()} mismatch with rank:{rank}'
    # npu_graph_test_suit = NPUGraphTest(f"npu:{torch.npu.current_device()}", m=test_m, n=test_n, k=test_k,
    #                                    use_npu_graph=True, test_type=test_type)
    # npu_graph_test_suit.graph_testing(a, b, c, bias, pertoken_scale, scale)
    direct_testing(a, b, c)


if __name__ == "__main__":
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)

    try:
        catcoc_ops = torch.ops.npu.catcoc_allgather_matmul
    except Exception as e:
        print(
            "use catcoc ops in sglang-kernel need to set BUILD_KERNELS_MODULE in cmake during compiling"
        )
        raise e

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    shmem_init()

    for mnk_list in ((64, 7168, 2048),):
        run_global_test(test_mnk=mnk_list)

    ash.shmem_free(g_shmem_addr)
    ash.shmem_finialize()
