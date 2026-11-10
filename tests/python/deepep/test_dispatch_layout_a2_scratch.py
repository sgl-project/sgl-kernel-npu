"""Manual two-node A2 regression for repeated layout scratch/output reuse.

Run on each eight-NPU host with the same MASTER_ADDR and MASTER_PORT:
  RUN_A2_MULTINODE_TESTS=1 NODE_RANK=0 python test_dispatch_layout_a2_scratch.py
Use NODE_RANK=1 on the second host. A2 multi-node CI is currently disabled.
The test uses native zero-token inputs, not dummy-token padding.
"""

import os
import unittest
from datetime import timedelta


def _worker(local_rank):
    import deep_ep
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401 - registers the NPU backend

    local_world, world = 8, 16
    hidden, experts, topk = 6144, 256, 8
    node_rank = int(os.environ["NODE_RANK"])
    rank = node_rank * local_world + local_rank
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        "hccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world,
        rank=rank,
        timeout=timedelta(minutes=10),
    )
    group = dist.new_group(list(range(world)))
    configs = (
        deep_ep.Buffer.get_dispatch_config(world),
        deep_ep.Buffer.get_combine_config(world),
    )
    buffer = deep_ep.Buffer(
        group,
        max(c.get_nvl_buffer_size_hint(hidden * 2, world) for c in configs),
        max(c.get_rdma_buffer_size_hint(hidden * 2, world) for c in configs),
        low_latency_mode=False,
        num_qps_per_rank=deep_ep.Buffer.num_sms,
        allow_mnnvl=False,
    )
    dist.barrier()
    # Reuse one Buffer while changing token counts and which DP replica is idle.
    # Small counts also exercise vector cores outside the active-core set.
    for phase, rounds in (("all_real", 128), ("dp1_empty", 512), ("dp0_empty", 512)):
        for step in range(rounds):
            empty = (phase == "dp1_empty" and node_rank == 1) or (
                phase == "dp0_empty" and node_rank == 0
            )
            count = 0 if empty else (1, 3, 20, 128)[step % 4]
            x = torch.full(
                (count, hidden), rank + 1, dtype=torch.bfloat16, device="npu"
            )
            token_ids = torch.arange(count, dtype=torch.int32, device="npu")
            # Distinguish tokens within a rank as well as between ranks.
            x[:, 0] = (token_ids % 128).to(torch.bfloat16)
            x[:, 1] = ((token_ids * 17 + rank * 7) % 128 - 64).to(torch.bfloat16)
            torch.manual_seed(100000 + rank * 1000 + step)
            ids = torch.topk(
                torch.rand((count, experts), device="npu"), topk, dim=-1
            ).indices
            weights = torch.full(
                (count, topk), 1 / topk, dtype=torch.float32, device="npu"
            )
            per_rank, per_rdma_rank, per_expert, in_rank, layout_event = (
                buffer.get_dispatch_layout(
                    ids,
                    experts,
                    previous_event=deep_ep.Buffer.capture(),
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
            )
            recv_x, _, _, _, handle, dispatch_event = buffer.dispatch(
                x,
                topk_idx=ids,
                topk_weights=weights,
                num_tokens_per_rank=per_rank,
                num_tokens_per_rdma_rank=per_rdma_rank,
                num_tokens_per_expert=per_expert,
                is_token_in_rank=in_rank,
                previous_event=layout_event,
                async_finish=True,
                allocate_on_comm_stream=True,
                expert_alignment=1,
            )
            dispatch_event.current_stream_wait()
            combined, _, combine_event = buffer.combine(
                recv_x,
                handle,
                previous_event=deep_ep.Buffer.capture(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            combine_event.current_stream_wait()
            torch.npu.synchronize()
            torch.testing.assert_close(combined, x, atol=0.01, rtol=0.001)
            dist.barrier()
        if rank == 0:
            print(f"PASS phase={phase} rounds={rounds}", flush=True)
    dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("RUN_A2_MULTINODE_TESTS") == "1",
    "requires two eight-NPU Ascend A2 hosts and explicit opt-in",
)
class TestDispatchLayoutA2Scratch(unittest.TestCase):
    def test_repeated_dispatch_combine(self):
        import torch.multiprocessing as mp

        self.assertIn(os.environ.get("NODE_RANK"), ("0", "1"))
        self.assertTrue(os.environ.get("MASTER_ADDR"))
        self.assertTrue(os.environ.get("MASTER_PORT"))
        mp.spawn(_worker, nprocs=8)


if __name__ == "__main__":
    unittest.main()
