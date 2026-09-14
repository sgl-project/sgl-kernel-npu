"""Hardware protocol checks; one Buffer, one shape, changing inputs and graph replay."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=6)
p.add_argument("--out", required=True)
p.add_argument("--expected-module-root")
a = p.parse_args()
torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
torch_npu.npu.set_compile_mode(jit_compile=False)
dist.init_process_group("hccl")
r, W = dist.get_rank(), dist.get_world_size()
T, H, E, K = a.tokens, 2048, 16 * dist.get_world_size(), 8
import deep_ep
from deep_ep import Buffer

if a.expected_module_root:
    assert (
        Path(deep_ep.__file__)
        .resolve()
        .is_relative_to(Path(a.expected_module_root).resolve())
    ), deep_ep.__file__
buf = Buffer(
    dist.group.WORLD,
    num_nvl_bytes=0,
    num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(T, H, W, E),
    low_latency_mode=True,
    num_qps_per_rank=16,
    low_latency_strategy="default",
)
x = torch.empty((T, H), dtype=torch.bfloat16, device="npu")
ids = torch.empty((T, K), dtype=torch.int64, device="npu")
weights = torch.empty((T, K), dtype=torch.float32, device="npu")
rng = np.random.default_rng(789)
cases = []
for case in range(24):
    route = np.empty((W, T, K), dtype=np.int64)
    for src in range(W):
        for t in range(T):
            if case % 6 == 0:
                row = [((src + 1) % W) * 16 + k for k in range(K)]
            elif case % 6 == 1:
                row = [((src + (k % (W - 1)) + 1) % W) * 16 + k for k in range(K)]
            elif case % 6 == 2:
                row = [src * 16 + k for k in range(K)]
            else:
                row = rng.choice(E, size=K, replace=False).tolist()
            route[src, t] = row
    if case % 6 == 3:
        route[rng.random(route.shape) < 0.4] = -1
    if case % 6 == 4:
        route[:, :, 0] = -1
        route[0, :, :] = -1  # masked potential leaders and an empty source
    if case % 6 == 5:
        route[:] = -1  # zero counts and no payloads on every rank
    cases.append(route)


def set_input(case):
    route = cases[case]
    host = torch.arange(1, T + 1, dtype=torch.float32)[:, None].expand(T, H).clone()
    host[:, 0] = r + 1
    host[:, 2] = case + 1
    x.copy_(host.to(torch.bfloat16))
    ids.copy_(torch.from_numpy(route[r]))
    w = np.tile(np.arange(1, 9, dtype=np.float32) / 36, (T, 1))
    w[route[r] < 0] = 0
    weights.copy_(torch.from_numpy(w))
    return route


def dispatch():
    return buf.low_latency_dispatch(x, ids, T, E, use_fp8=False, topk_weights=weights)


def expected_row(src, t, case):
    z = torch.full((H,), float(t + 1))
    z[0] = src + 1
    z[2] = case + 1
    return z.to(torch.bfloat16)


def validate(case, rx, cnt, handle, identity_out=None):
    route = cases[case]
    counts = np.bincount(route[route >= 0], minlength=E)[r * 16 : (r + 1) * 16]
    actual = cnt.cpu().numpy()
    assert np.array_equal(actual, counts), (actual, counts)
    n = int(counts.sum())
    got = rx[:n].cpu()
    triples = handle[0][: n * 3].cpu().reshape(n, 3).numpy()
    out_rows = torch.empty_like(rx)
    off = 0
    validated = 0
    for local, count in enumerate(counts):
        expert = r * 16 + local
        expected = set(map(tuple, np.argwhere(route == expert).tolist()))
        seen = set()
        for j in range(off, off + int(count)):
            src, t, k = map(int, triples[j])
            assert (src, t, k) in expected
            assert (src, t, k) not in seen
            seen.add((src, t, k))
            assert torch.equal(got[j], expected_row(src, t, case)), (
                case,
                r,
                expert,
                j,
                triples[j],
            )
            validated += 1
        assert seen == expected
        if count:
            out_rows[off : off + int(count)] = (
                rx[off : off + int(count)].float() * (1 + (expert % 5) / 4)
                + (expert % 7) / 8
            ).to(torch.bfloat16)
        off += int(count)
    if identity_out is not None:
        gate = weights.cpu().sum(dim=1)[:, None]
        ref = x.cpu().float() * gate
        active = torch.from_numpy((route[r] >= 0).any(axis=1))
        assert torch.allclose(
            identity_out.cpu().float()[active], ref[active], rtol=0.02, atol=0.08
        )
    return validated, out_rows


set_input(0)
rx, cnt, h, _, _ = dispatch()
validate(0, rx, cnt, h)
for _ in range(3):
    rx, cnt, h, _, _ = dispatch()
    buf.low_latency_combine(rx, ids, weights, h)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    grx, gcnt, gh, _, _ = dispatch()
    gout = buf.low_latency_combine(grx, ids, weights, gh)[0]
checks = rows = 0
max_error = 0.0
for case in range(24):
    route = set_input(case)
    rx, cnt, h, _, _ = dispatch()
    n, transformed = validate(case, rx, cnt, h)
    rows += n
    out = buf.low_latency_combine(transformed, ids, weights, h)[0].cpu().float()
    expected = torch.zeros((T, H), dtype=torch.float32)
    for t in range(T):
        for k, e in enumerate(route[r, t]):
            if e >= 0:
                expected[t] += (
                    (expected_row(r, t, case).float() * (1 + (e % 5) / 4) + (e % 7) / 8)
                    .to(torch.bfloat16)
                    .float()
                ) * ((k + 1) / 36)
    # The native combine leaves fully masked token outputs unspecified; callers ignore those rows.
    active = torch.from_numpy((route[r] >= 0).any(axis=1))
    err = (out[active] - expected[active]).abs().max().item() if active.any() else 0.0
    max_error = max(max_error, err)
    assert torch.allclose(out[active], expected[active], rtol=0.02, atol=0.08), (
        case,
        r,
        err,
    )
    # Two replays exercise both alternating window halves with changed IDs and masks.
    for repeat in range(2):
        graph.replay()
        torch.npu.synchronize()
        n, _ = validate(case, grx, gcnt, gh, gout)
        rows += n
    checks += 1
root = Path(a.out)
root.mkdir(parents=True, exist_ok=True)
res = {
    "valid": True,
    "rank": r,
    "world": W,
    "tokens": T,
    "cases": checks,
    "graph_replays": 48,
    "validated_expert_rows": rows,
    "max_weighted_combine_abs_error": max_error,
    "module": deep_ep.__file__,
    "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    "expected_module_root": a.expected_module_root,
    "fully_masked_output_rows_ignored": True,
}
(root / f"rank{r}.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res), flush=True)
dist.barrier()
dist.destroy_process_group()
