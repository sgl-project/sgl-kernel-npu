import os
import random
import time
import unittest

import sgl_kernel_npu
import shmem as ash
import torch
import torch.distributed as dist
import torch_npu

torch.set_printoptions(threshold=float("inf"))


g_ash_size = 2 * 1024 * 1024 * 1024
g_malloc_size = 1024 * 1024 * 1024
g_shmem_addr = None
g_team_size = 2


def direct_testing(input_a, input_b, input_c, team_id=0, group_list=(), use_nz=False, run_cnt=1):
    global g_shmem_addr
    # g_shmem_addr = 0

    a, b = input_a, input_b
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    if rank == 0:
        print(f"[py] addr is:{a.data_ptr()} {b.data_ptr()} {input_c.data_ptr()}")

    l_world_size = (
        int(world_size / len(group_list)) if len(group_list) > 0 else world_size
    )

    # assert g_shmem_addr is not None and k == b.shape[0]
    # print('rank', rank, ' is ', a.device, b.device)
    for _ in range(run_cnt):
        if use_nz:
            b_nz = torch_npu.npu_format_cast(input_b, 29)
            torch.ops.npu.catcoc_matmul_allreduce(
                a, b_nz, input_c, g_shmem_addr, team_id, format_mode="NZ"
            )
        else:
            torch.ops.npu.catcoc_matmul_allreduce(a, b, input_c, g_shmem_addr, team_id)
    torch.npu.synchronize()

    native_c = torch.matmul(a, b)
    if world_size > 1:
        dist.all_reduce(native_c, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(input_c.shape, input_c.flatten()[:10], input_c.flatten()[-10:])
        print(rank, native_c.shape, native_c.flatten()[:10], native_c.flatten()[-10:])
    # time.sleep(rank * 5)
    assert torch.allclose(input_c, native_c, rtol=1e-2, atol=1e-2)
    print("rank", rank, " success")


def shmem_init(rank, world_size):
    # original init
    # from shmem import set_conf_store_tls
    #
    # global g_shmem_addr, g_ash_size, g_malloc_size
    # set_conf_store_tls(False, "")
    # shmem_addr = "tcp://127.0.0.1:26666"
    # attributes = ash.InitAttr()
    # attributes.my_rank = rank
    # attributes.n_ranks = world_size
    # attributes.local_mem_size = g_ash_size
    # attributes.ip_port = shmem_addr
    # attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    # ret = ash.shmem_init(attributes)
    # assert ret == 0, '[ERROR] aclshmem_init failed'
    #
    # g_shmem_addr = ash.shmem_malloc(g_malloc_size)

    # uid init(need env SHMEM_UID_SOCK_IFNAM=enp194s0f0::inet4)
    global g_shmem_addr, g_ash_size, g_malloc_size
    # 0. disable TLS
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] disable tls failed.")

    # 1. get unique id
    uid_size = 512
    tensor = torch.zeros(uid_size, dtype=torch.uint8, device=f"npu:{rank}")
    if rank == 0:
        unique_id = ash.shmem_get_unique_id()
        if unique_id is None:
            raise ValueError("[ERROR] get unique id failed")
        uid_list = [0] * uid_size
        uid_list[: len(unique_id)] = unique_id
        tensor = torch.tensor(uid_list, dtype=torch.uint8, device=f"npu:{rank}")
    dist.broadcast(tensor, src=0)
    torch.npu.synchronize()
    if rank != 0:
        unique_id = bytes(tensor.cpu().tolist())
    # 2. init with unique id
    ret = ash.shmem_init_using_unique_id(rank, world_size, g_ash_size, unique_id)
    if ret != 0:
        raise ValueError("[ERROR] shmem_init failed")

    # test malloc
    g_shmem_addr = ash.shmem_malloc(g_malloc_size)
    print(f"rank[{rank}]: shmem_ptr:{g_shmem_addr} with type{type(g_shmem_addr)}")
    if g_shmem_addr is None:
        raise ValueError("[ERROR] shmem_malloc failed")

    # test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f"rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}")
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError("[ERROR] pe/world failed")


def run_global_test(test_mnk=(1024, 1024, 1024), test_nz=True):
    test_m, test_n, test_k = test_mnk
    if rank == 0:
        print("+++++++++++++++++++++++Testing FP16+++++++++++++++++")
    a = (
        torch.rand([test_m, test_k])
        .to(dtype=torch.float16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    b = (
        torch.rand([test_k, test_n])
        .to(dtype=torch.float16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    c = (
        torch.empty([test_m, test_n])
        .to(dtype=torch.float16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    bias = None
    scale = None
    pertoken_scale = None

    assert (
        torch.npu.current_device() == rank
    ), f"[ERROR] device:{torch.npu.current_device()} mismatch with rank:{rank}"
    # npu_graph_test_suit = NPUGraphTest(f"npu:{torch.npu.current_device()}", m=test_m, n=test_n, k=test_k,
    #                                    use_npu_graph=True, test_type=test_type)
    # npu_graph_test_suit.graph_testing(a, b, c, bias, pertoken_scale, scale)
    direct_testing(a, b, c)
    if test_nz:
        if rank == 0:
            print("+++++++++++++++++++++++Testing FP16(WeightNZ)+++++++++++++++++")
        c = (
            torch.empty([test_m, test_n])
            .to(dtype=torch.float16)
            .to(f"npu:{rank}")
            .contiguous()
        )  # fresh tensor
        direct_testing(a, b, c, use_nz=True)

    if rank == 0:
        print("+++++++++++++++++++++++Testing BF16+++++++++++++++++")
    a = (
        torch.rand([test_m, test_k])
        .to(dtype=torch.bfloat16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    b = (
        torch.rand([test_k, test_n])
        .to(dtype=torch.bfloat16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    c = (
        torch.empty([test_m, test_n])
        .to(dtype=torch.bfloat16)
        .to(f"npu:{rank}")
        .contiguous()
    )
    bias = None
    scale = None
    pertoken_scale = None

    assert (
        torch.npu.current_device() == rank
    ), f"[ERROR] device:{torch.npu.current_device()} mismatch with rank:{rank}"
    # npu_graph_test_suit = NPUGraphTest(f"npu:{torch.npu.current_device()}", m=test_m, n=test_n, k=test_k,
    #                                    use_npu_graph=True, test_type=test_type)
    # npu_graph_test_suit.graph_testing(a, b, c, bias, pertoken_scale, scale)
    direct_testing(a, b, c)
    if test_nz:
        if rank == 0:
            print("+++++++++++++++++++++++Testing BF16(WeightNZ)+++++++++++++++++")
        c = (
            torch.empty([test_m, test_n])
            .to(dtype=torch.bfloat16)
            .to(f"npu:{rank}")
            .contiguous()
        )  # fresh tensor
        direct_testing(a, b, c, use_nz=True)


def run_single_test(test_mnk=(1024, 1024, 1024), test_bf16=True, test_nz=True, test_cnt=10):
    test_m, test_n, test_k = test_mnk
    test_dtype = torch.bfloat16 if test_bf16 else torch.float16
    a = (
        torch.rand([test_m, test_k])
        .to(dtype=test_dtype)
        .to(f"npu:{rank}")
        .contiguous()
    )
    b = (
        torch.rand([test_k, test_n])
        .to(dtype=test_dtype)
        .to(f"npu:{rank}")
        .contiguous()
    )
    c = (
        torch.empty([test_m, test_n])
        .to(dtype=test_dtype)
        .to(f"npu:{rank}")
        .contiguous()
    )
    bias = None
    scale = None
    pertoken_scale = None

    assert (
            torch.npu.current_device() == rank
    ), f"[ERROR] device:{torch.npu.current_device()} mismatch with rank:{rank}"
    # npu_graph_test_suit = NPUGraphTest(f"npu:{torch.npu.current_device()}", m=test_m, n=test_n, k=test_k,
    #                                    use_npu_graph=True, test_type=test_type)
    # npu_graph_test_suit.graph_testing(a, b, c, bias, pertoken_scale, scale)
    if not test_nz:
        direct_testing(a, b, c, run_cnt=test_cnt)
    else:
        c = (
            torch.empty([test_m, test_n])
            .to(dtype=test_dtype)
            .to(f"npu:{rank}")
            .contiguous()
        )  # fresh tensor
        direct_testing(a, b, c, use_nz=True, run_cnt=test_cnt)


if __name__ == "__main__":
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)

    try:
        catcoc_ops = torch.ops.npu.catcoc_matmul_allreduce
    except Exception as e:
        print(
            "use catcoc ops in sglang-kernel need to set BUILD_KERNELS_MODULE in cmake during compiling"
        )
        raise e

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    # dist.init_process_group(backend="gloo", init_method="env://")
    dist.init_process_group(backend="hccl", rank=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    shmem_init(rank, world_size)

    for mnk_list in ((64, 7168, 2048),):
        run_global_test(test_mnk=mnk_list, test_nz=True)
    # run_single_test(test_mnk=(10320, 2048, 2048))

    ash.shmem_free(g_shmem_addr)
    ash.shmem_finialize()
