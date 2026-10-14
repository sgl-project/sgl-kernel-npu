"""Four-rank A2 correctness/replay regression; not a latency benchmark.

Reference generation runs in a separate process without the operator-only
library override. See csrc/deepep/ops/README.md for setup and commands.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from test_dispatch_ffn_combine import (
    baseline_test,
    init_base_weights,
    init_baseline_weights,
    init_fused2_weights_int8,
)
from utils import init_dist


def worker(local_rank, args):
    torch.set_num_threads(1)
    torch.manual_seed(1100 + local_rank)
    rank, world, group = init_dist(local_rank, 4)
    assert world == 4
    torch_npu.npu.config.allow_internal_format = True
    n, hidden, width, experts, topk = args.tokens, 2048, 1536, 128, 8
    buffer = Buffer(
        group,
        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(
            n * topk * world, hidden, world, experts
        ),
        low_latency_mode=True,
        num_qps_per_rank=32,
    )
    weights = init_base_weights(32, hidden, width)
    x = torch.randn(n, hidden, dtype=torch.bfloat16, device='npu')
    ids = torch.rand(n, experts, device='npu').topk(topk, dim=-1).indices
    if args.sparse_case:
        generator = torch.Generator(device='cpu').manual_seed(812 + rank)
        ids = torch.rand((n, 126), generator=generator, device='cpu').topk(8, dim=1).indices
        counts126 = [4, 4, 1, 6] if args.sparse_case == 'one' else [2, 5, 5, 3]
        ids[:counts126[rank], 0] = 126
        if args.sparse_case == 'one' and rank == 2:
            ids[counts126[rank], 0] = 127
        x *= 0.1
    ids = ids.to(device='npu', dtype=torch.int32)
    gates = torch.full((n, topk), 1 / topk, device='npu', dtype=torch.float32)
    expected = torch.bincount(ids.flatten().long(), minlength=experts)
    dist.all_reduce(expected, group=group)
    expected = expected[rank * 32:(rank + 1) * 32]
    snapshot = args.reference_dir / f'rank_{rank}.pt'
    if args.write_reference:
        assert not snapshot.exists(), f'Refusing to replace {snapshot}'
        refweights = init_baseline_weights(*(w.clone() for w in weights))
        stats = torch.zeros(32, dtype=torch.int32, device='npu')
        output, counts = baseline_test(
            buffer, x, ids.long(), n, experts, stats, False, *refweights, gates
        )
        torch.npu.synchronize()
        assert torch.equal(counts.long(), expected)
        torch.save(dict(output=output.cpu(), x=x.cpu(), ids=ids.cpu(),
                        weights=[w.cpu() for w in weights], gates=gates.cpu()), snapshot)
    else:
        reference = None
        if not args.sparse_case:
            saved = torch.load(snapshot, weights_only=True)
            assert torch.equal(x.cpu(), saved['x'])
            assert torch.equal(ids.cpu(), saved['ids'])
            assert torch.equal(gates.cpu(), saved['gates'])
            assert all(torch.equal(w.cpu(), s) for w, s in zip(weights, saved['weights']))
            reference = saved['output'].to('npu')
        fused = init_fused2_weights_int8(*(w.clone() for w in weights))

        def call():
            return buffer.fused_deep_moe(
                x, ids, gates, *fused, n * topk * world, experts, 1, 2
            )

        first, error = None, None
        for _ in range(args.repeats):
            output, counts = call()
            torch.npu.synchronize()
            assert torch.equal(counts.long(), expected)
            assert torch.isfinite(output).all()
            if reference is not None:
                error = (output.float() - reference.float()).abs().mean().item()
                assert error < 1e-2, (rank, error)
            if first is None:
                first = output.clone()
            else:
                assert torch.equal(output, first), f'Rank {rank}: output changed'
        dist.barrier()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output, counts = call()
        for _ in range(args.graph_replays):
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(counts.long(), expected)
            assert torch.equal(output, first)
        print(json.dumps(dict(rank=rank, tokens=n, sparse_case=args.sparse_case,
                              calls=args.repeats, graph_replays=args.graph_replays,
                              reference_mean_abs_error=error, passed=True)), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--write-reference', action='store_true')
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--graph-replays', type=int, default=100)
    parser.add_argument('--sparse-case', choices=['one', 'empty'])
    args = parser.parse_args()
    if args.tokens < 8 or args.repeats < 2 or args.graph_replays < 1:
        parser.error('tokens >= 8, repeats >= 2 and graph-replays >= 1 required')
    selected = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').split(',')
    if len(selected) != 4 or len(set(selected)) != 4 or not all(s.isdigit() for s in selected):
        parser.error('explicitly select four idle devices in ASCEND_RT_VISIBLE_DEVICES')
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=4)
