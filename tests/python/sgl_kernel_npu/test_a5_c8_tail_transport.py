"""A3/A5 byte-exact C8 tail checks through changing speculative windows.

These tests execute the production metadata and registered-memory copy path.
They do not execute A5 LIM, A5 CopySFA, or native FP8 arithmetic.
"""

import os

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("capacity", [8192, 16384, 32384])
def test_packed_tail_acceptance_reuse_and_page_crossing(width, graph_replay, capacity):
    from sgl_kernel_npu.sparsity_driven_kv_offload.c8_metadata import (
        stage_mtp_tail,
    )
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        create_fused_shm_tensor as create_shm_tensor,
    )
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        free_fused_shm as free_shm,
    )

    assert torch.npu.is_available()
    torch.npu.set_device(0)
    context, batch, request_rows = capacity + 1024, 3, 4
    generator = torch.Generator().manual_seed(20261031 + capacity + width)
    host, _, pointer = create_shm_tensor((2 * context, 1, 656), torch.int8)
    try:
        host.copy_(
            torch.randint(-128, 128, host.shape, dtype=torch.int8, generator=generator)
        )
        pages = context // 128
        page_ids = torch.randperm(pages, generator=generator)
        physical = (page_ids[:, None] * 128 + torch.arange(128)[None]).flatten()
        mapping_cpu = torch.zeros(request_rows, context, dtype=torch.int64)
        mapping_cpu[1].copy_(physical)
        mapping_cpu[2].copy_(context + physical)
        # Prefix aliases are the same physical rows in host memory. Hot pools
        # and current speculative writes must remain private to each request.
        mapping_cpu[2, :capacity].copy_(mapping_cpu[1, :capacity])
        mapping = mapping_cpu.to("npu")
        requests = torch.tensor([1, 2, 0], dtype=torch.int64, device="npu")
        positions = torch.zeros(batch * width, dtype=torch.int64, device="npu")
        current = torch.empty(batch * width, 1, 656, dtype=torch.int8, device="npu")
        cache = torch.full(
            (request_rows, capacity + 256, 1, 656), -17, dtype=torch.int8, device="npu"
        )
        starts = torch.full((request_rows + 1,), -1, dtype=torch.int32, device="npu")
        ends = torch.full_like(starts, -1)
        hs = torch.empty(batch * 128, dtype=torch.int64, device="npu")
        hd, hv = torch.empty_like(hs), torch.empty_like(hs, dtype=torch.bool)
        cs = torch.arange(batch * width, dtype=torch.int64, device="npu")
        cd, cv = torch.empty_like(cs), torch.empty_like(cs, dtype=torch.bool)

        def launch():
            stage_mtp_tail(
                current=current,
                cache=cache,
                host=host,
                host_address=pointer,
                req_to_token=mapping,
                requests=requests,
                positions=positions,
                starts=starts,
                ends=ends,
                workspace=(hs, hd, hv, cs, cd, cv),
                capacity=capacity,
                context_len=context,
                query_width=width,
            )

        positions.copy_(
            torch.tensor(
                [[capacity + 125 + offset for offset in range(width)]] * batch,
                device="npu",
            ).flatten()
        )
        current.zero_()
        launch()
        torch.npu.synchronize()
        graph = None
        if graph_replay:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                launch()

        rounds = int(os.environ.get("SGLANG_PACKED_TRANSPORT_ROUNDS", "32"))
        firsts = [capacity + 125, capacity + 127]
        previous = None
        for step in range(rounds):
            if previous is not None:
                previous_firsts, previous_current = previous
                for row in range(2):
                    accepted = 1 + (step + row) % width
                    for offset in range(accepted):
                        position = previous_firsts[row] + offset
                        host[mapping_cpu[row + 1, position]].copy_(
                            previous_current[row * width + offset]
                        )
                    firsts[row] = previous_firsts[row] + accepted
            if step % 8 == 0:
                # Reusing a request slot invalidates its cached tail interval.
                starts.fill_(-1)
                ends.fill_(-1)
                firsts = [capacity + 125, capacity + 253]
            if step % 8 == 5:
                # A prefix rollback must restore bytes outside the old range.
                firsts = [capacity + 3, capacity + 129]
            cpu_current = torch.randint(
                -128, 128, current.shape, dtype=torch.int8, generator=generator
            )
            current.copy_(cpu_current)
            cpu_positions = torch.zeros(batch, width, dtype=torch.int64)
            for row, first in enumerate(firsts):
                cpu_positions[row] = first + torch.arange(width)
            positions.copy_(cpu_positions.flatten())
            graph.replay() if graph is not None else launch()
            torch.npu.synchronize()
            result = cache.cpu()
            for row, first in enumerate(firsts):
                prefix = first // 128 * 128
                for position in range(prefix, first):
                    assert torch.equal(
                        result[row + 1, capacity + position % 256],
                        host[mapping_cpu[row + 1, position]],
                    ), (step, row, position)
                for offset in range(width):
                    assert torch.equal(
                        result[row + 1, capacity + (first + offset) % 256],
                        cpu_current[row * width + offset],
                    ), (step, row, offset)
            assert bool((result[0] == -17).all())
            assert bool((result[3] == -17).all())
            assert bool((result[:, :capacity] == -17).all())
            previous = (list(firsts), cpu_current)
    finally:
        torch.npu.synchronize()
        free_shm()
