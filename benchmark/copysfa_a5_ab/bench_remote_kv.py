"""Time one decode-side A5 C8 CopySFA with local DRAM or prefill HBM.

Start one process per role with the same fixture options.
Rank 0 (prefill) owns the source HBM. Rank 1 (decode) runs both measurements.
Requires the current A5 sgl_kernel_npu wheel, pyACL and MemFabric Hybrid SHM
with a directly MTE-reachable pair of devices.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from fixture import (
    PAGE,
    ROW_BYTES,
    TOPK,
    FixtureSpec,
    _packed_rows,
    build_torch_fixture,
)
from result_contract import timing_stats

MIB = 1024 * 1024
ACL_MEMCPY_HOST_TO_DEVICE = 1
ROLE_IPS = {"prefill": "141.61.45.121", "decode": "141.61.45.122"}
DEFAULT_STORE = f"tcp://{ROLE_IPS['prefill']}:18570"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "role",
        choices=ROLE_IPS,
        help=f"prefill on {ROLE_IPS['prefill']}, decode on {ROLE_IPS['decode']}",
    )
    p.add_argument(
        "--store", default=DEFAULT_STORE, help=f"SHM store (default: {DEFAULT_STORE})"
    )
    p.add_argument("--device", type=int, default=0, help="local logical NPU index")
    p.add_argument("--q", type=int, choices=(1, 4), default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, choices=(8, 16, 32, 64, 128), default=8)
    p.add_argument("--prefix", type=int, default=16384)
    p.add_argument(
        "--miss-count", type=int, default=512, help="union misses per request"
    )
    p.add_argument(
        "--source-copies",
        type=int,
        default=1,
        help="rotate identical KV banks to vary the source working set",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100, help="paired samples per source")
    p.add_argument("--output", type=Path, help="optional decode-side result JSON")
    args = p.parse_args()
    if min(args.miss_count, args.iters, args.source_copies) < 1 or args.warmup < 0:
        p.error("miss-count, iters and source-copies must be positive, and warmup >= 0")
    spec = FixtureSpec(
        query_width=args.q,
        batch=args.batch,
        heads=args.heads,
        index_heads=1,
        capacity=args.q * TOPK,
        prefix=args.prefix,
        context=args.prefix + args.q,
        miss_counts=(args.miss_count,),
        selection="disjoint",
    )
    try:
        spec.validate()
    except ValueError as exc:
        p.error(str(exc))
    if spec.context > 2048 * PAGE or args.device < 0:
        p.error("CopySFA supports at most 2048 source pages and requires device >= 0")
    if args.source_copies > args.iters:
        p.error("iters must cover every source bank")
    return args, spec


def check(rc, operation):
    if rc != 0:
        raise RuntimeError(f"{operation} failed (rc={rc})")


class PeerWindow:
    """Own a MemFabric MTE window with two ranks."""

    @contextmanager
    def open(self, mf, rank, args, size):
        with ExitStack() as cleanup:
            check(mf.initialize(), "MemFabric initialize")
            cleanup.callback(mf.uninitialize)
            shm = mf.shm
            config = shm.ShmConfig()
            config.operation_timeout = 900
            check(
                shm.initialize(args.store, 2, rank, args.device, config),
                "SHM initialize",
            )
            cleanup.callback(shm.uninitialize, 0)
            self.handle = shm.create(0, 2, rank, size, shm.ShmDataOpType.MTE, 0)
            cleanup.callback(lambda: check(self.handle.destroy(), "SHM destroy"))
            gva = int(self.handle.gva)
            if gva <= 0:
                raise RuntimeError("SHM returned a null GVA")
            # Match the kernel-facing GVA in MemFabric's shm_mte_perf.py.
            # Prefill owns rank 0, so its slice starts at the window base.
            yield gva

    def gather(self, message):
        size = 2048
        data = json.dumps(message).encode()
        if len(data) >= size:
            raise ValueError("control message exceeds SHM allgather slot")
        received = self.handle.all_gather(data.ljust(size, b" "))
        try:
            if len(received) != 2 * size:
                raise ValueError("unexpected payload size")
            return [json.loads(received[i : i + size]) for i in (0, size)]
        except ValueError as exc:
            raise RuntimeError("SHM all_gather did not return both ranks") from exc

    @contextmanager
    def phase(self, name):
        # Keep the source mapped until decode has synchronized, including on
        # numerical failures. Relay the failing phase to the other process.
        try:
            yield
        except Exception as exc:
            self.gather(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
            raise
        errors = self.gather(None)
        if any(errors):
            raise RuntimeError(f"peer failed: {errors}")


def run_decode(torch, args, spec, source_ptr, cleanup):
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        create_fused_shm_tensor,
        free_fused_shm,
    )

    fixture = build_torch_fixture(torch, spec)
    cpu = fixture["tensors"]
    packed = cpu["packed"]
    host, _, host_ptr = create_fused_shm_tensor(
        (args.source_copies, *packed.shape), torch.int8, device_id=args.device
    )
    cleanup.callback(free_fused_shm, args.device)
    host.copy_(packed.expand_as(host))
    device = torch.device("npu", args.device)
    initial = cpu["initial_hbm"].to(device)
    hbm = torch.empty_like(initial)
    output = torch.empty_like(cpu["golden"], dtype=torch.bfloat16, device=device)
    # Both calls get the SAME inert descriptor. Only the explicit pointer
    # can yield correct miss bytes. No remote buffer is staged on decode.
    descriptor = torch.zeros_like(packed)
    metadata = [
        cpu[key].to(device)
        for key in (
            "hbm_table",
            "source_table",
            "destination_slots",
            "source_ids",
            "route_misses",
            "query_ends",
            "seq_lens",
            "hbm_lengths",
            "miss_sources",
            "miss_slots",
            "miss_counts",
        )
    ]
    query = cpu["merged_query"].to(device)
    op = torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention.default
    scale = 1 / math.sqrt(query.shape[-1])
    sources = {"decode_dram": host_ptr, "prefill_hbm": source_ptr}

    def reset():
        hbm.copy_(initial)
        output.fill_(float("nan"))

    def invoke(name, bank):
        op(
            query,
            hbm,
            descriptor,
            sources[name] + bank * packed.nbytes,
            *metadata,
            scale,
            output,
        )

    for bank in range(args.source_copies):
        for name in sources:
            reset()
            invoke(name, bank)
            torch.npu.synchronize()
            try:
                torch.testing.assert_close(
                    hbm.cpu(), cpu["expected_hbm"], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    output.cpu().float(), cpu["golden"], atol=0.008, rtol=0.05
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"{name} bank {bank} failed CopySFA validation: {exc}"
                ) from exc

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = {name: [] for name in sources}
    for i in range(-args.warmup, args.iters):
        bank = i % args.source_copies
        for name in tuple(reversed(sources)) if i % 2 else tuple(sources):
            reset()
            start.record()
            invoke(name, bank)
            end.record()
            end.synchronize()
            if i >= 0:
                samples[name].append(float(start.elapsed_time(end)) * 1000)

    timings = {name: timing_stats(values) for name, values in samples.items()}
    dram, peer = (timings[name]["median_us"] for name in sources)
    if min(dram, peer) <= 0:
        raise ValueError("invalid NPU event median")
    return {
        "fixture": fixture["manifest"],
        "fixture_sha256": fixture["tensor_sha256"],
        "route_misses": cpu["route_misses"].tolist(),
        "source_copies": args.source_copies,
        "warmup_per_source": args.warmup,
        "mode": "eager",
        "timer": "npu_event",
        "validation": "all source banks passed attention and byte-exact resident KV",
        "timings": timings,
        "speedup_dram_over_remote_hbm": dram / peer,
    }


def main():
    args, spec = parse_args()
    # The wheel stages its CANN vendor path before torch_npu initialization.
    import sgl_kernel_npu  # isort: skip
    import acl
    import memfabric_hybrid as mf
    import torch

    rank = 0 if args.role == "prefill" else 1
    torch.npu.set_device(args.device)
    if acl.rt.get_device_info(args.device, 601) != (3510, 0):
        raise RuntimeError("this case requires Ascend950")
    rows = math.ceil(spec.context / PAGE) * PAGE
    nbytes = rows * ROW_BYTES
    window_size = math.ceil(nbytes * args.source_copies / (2 * MIB)) * 2 * MIB
    if window_size > 4096 * MIB:
        raise ValueError("source banks exceed this benchmark's 4 GiB per-rank limit")
    window = PeerWindow()
    print(
        f"{args.role} ({ROLE_IPS[args.role]}): joining {args.store} on npu:{args.device}",
        flush=True,
    )
    with window.open(
        mf, rank, args, window_size
    ) as prefill_ptr, ExitStack() as cleanup:
        check(window.handle.barrier(), "SHM create barrier")
        peers = window.gather(
            {
                "hostname": socket.gethostname(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "device": args.device,
                "model": torch.npu.get_device_name(args.device),
                "fixture": spec.fingerprint(),
                "banks": args.source_copies,
            }
        )
        if peers[0]["boot_id"] == peers[1]["boot_id"]:
            raise ValueError(
                "cross-node benchmark requires different prefill and decode hosts"
            )
        if (peers[0]["fixture"], peers[0]["banks"]) != (
            peers[1]["fixture"],
            peers[1]["banks"],
        ):
            raise ValueError(
                "prefill and decode must specify the same fixture and source-copies"
            )
        with window.phase("publish prefill KV"):
            if rank == 0:
                packed = _packed_rows(torch, rows)
                for bank in range(args.source_copies):
                    check(
                        acl.rt.memcpy(
                            prefill_ptr + bank * nbytes,
                            nbytes,
                            packed.data_ptr(),
                            nbytes,
                            ACL_MEMCPY_HOST_TO_DEVICE,
                        ),
                        "prefill publish H2D",
                    )
                torch.npu.synchronize()
                print(
                    f"prefill: published {nbytes * args.source_copies} KV bytes",
                    flush=True,
                )
        check(window.handle.barrier(), "SHM KV publish barrier")
        try:
            with window.phase("decode CopySFA"):
                if rank == 1:
                    try:
                        result = run_decode(torch, args, spec, prefill_ptr, cleanup)
                    finally:
                        torch.npu.synchronize()
            check(window.handle.barrier(), "SHM teardown barrier")
        finally:
            if rank == 0:
                # Give the client time to consume the final store reply, as in
                # MemFabric's ShiftPutGet example. This is outside the timer.
                time.sleep(1)
    if rank == 1:
        result.update(
            peers=peers,
            configured_role_ips=ROLE_IPS,
            store=args.store,
            remote_hbm_mapping="memfabric_shm_mte_gva",
            kernel_module=sgl_kernel_npu.__file__,
            torch_version=str(torch.__version__),
            memfabric_module=mf.__file__,
        )
        if args.output:
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        print(result["validation"])
        for name, stats in result["timings"].items():
            print(
                f"{name:12} p50 {stats['median_us']:.2f} us   p95 {stats['p95_us']:.2f} us"
            )
        print(
            f"speedup (decode DRAM / prefill HBM): {result['speedup_dram_over_remote_hbm']:.3f}x"
        )


if __name__ == "__main__":
    main()
