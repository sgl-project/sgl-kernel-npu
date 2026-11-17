#!/usr/bin/env python3
"""Trace native A5 C8 LI/SFA versus the current framework HiSparse chain.

Runs one synthetic single-layer eager case, then repeats it with warm residency.
This is a correctness diagnostic, not a latency benchmark or graph acceptance test.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from bench import _add_prefix, _build_native, _git_sha, _sha256
from fixture import FixtureSpec, _index_key_scales, _packed_rows, unpack_rows
from tensor_trace import TensorTrace, gather_logical, summarize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-prefix", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-width", type=int, choices=(1, 4), default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, choices=(8, 16, 32, 64, 128), default=8)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--context", type=int, default=9344)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or args.batch < 1:
        parser.error("device must be nonnegative and batch must be positive")
    if not args.query_width <= args.context <= 262144:
        parser.error("context must be in [query-width, 262144]")
    if not 8192 <= args.capacity <= 32384 or args.capacity % 128:
        parser.error("capacity must be 128-aligned and in [8192, 32384]")
    if any(
        not math.isfinite(x) or x < 0
        for x in (args.atol, args.rtol, args.max_relative_l2)
    ):
        parser.error("tolerances must be finite and nonnegative")
    # Avoid the admission boundary where a Q4 fast path can have prefix < C.
    prefix = (args.context - args.query_width) // 128 * 128
    if args.context >= args.capacity and prefix < args.capacity:
        parser.error("choose context < C or context >= C + query-width")
    return args


def build_inputs(torch, args):
    """Reserve page zero and separate physical KV/index pages for every request."""
    b, q, h, n = args.batch, args.query_width, args.heads, args.context
    pages = (n + 127) // 128
    source_capacity = pages * 128
    base_key, base_value = unpack_rows(torch, _packed_rows(torch, source_capacity))
    dense_k = [torch.zeros(128, 512, dtype=torch.bfloat16)]
    dense_rope = [torch.zeros(128, 64, dtype=torch.bfloat16)]
    for request in range(b):
        dense_k.append((base_value + request / 16).to(torch.bfloat16))
        dense_rope.append((base_key[:, 512:] + request / 32).to(torch.bfloat16))
    dense_k, dense_rope = torch.cat(dense_k), torch.cat(dense_rope)
    index_key = torch.ones(1 + b * pages, 128, 1, 128).to(torch.float8_e4m3fn)
    # The gap at each causal TopK boundary survives BF16 score rounding.
    if n - q + 1 > 2048:
        spec = FixtureSpec(q, b, h, 32, args.capacity, (n - q) // 128 * 128, n)
        scales = _index_key_scales(torch, spec, source_capacity)
    else:
        scales = torch.linspace(1, 8, source_capacity).view(pages, 128, 1)
        if n > 2048:
            # Q4 can straddle the first TopK boundary at lengths 2049..2051.
            flat = scales.flatten()
            boundary = n - 2048
            flat[:boundary] = 0.25 * 2 ** torch.arange(boundary).float()
            flat[boundary:] = torch.linspace(4, 8, source_capacity - boundary)
    key_scale = torch.cat((torch.ones(1, 128, 1), scales.repeat(b, 1, 1)))
    table = 1 + torch.arange(b)[:, None] * pages + torch.arange(pages)[None]
    positions = torch.arange(n - q, n, dtype=torch.int64).repeat(b)
    reqmap = torch.zeros(b + 1, source_capacity, dtype=torch.int32)
    reqmap[1:] = (
        128
        + torch.arange(b)[:, None] * source_capacity
        + torch.arange(source_capacity)[None]
    )
    locations = reqmap[torch.arange(1, b + 1).repeat_interleave(q), positions].long()
    row = torch.arange(b * q)[:, None, None]
    head = torch.arange(h)[None, :, None]
    feature = torch.arange(576)[None, None]
    merged = (((row * 11 + head * 7 + feature * 3) % 31) - 15).to(torch.bfloat16) / 16
    return {
        "dense_k": dense_k,
        "dense_rope": dense_rope,
        "reqmap": reqmap,
        "locations": locations,
        "positions": positions,
        "q": merged[..., :512].contiguous(),
        "q_rope": merged[..., 512:].contiguous(),
        "merged_query": merged,
        "index_query": torch.ones(b * q, 32, 128).to(torch.float8_e4m3fn),
        "index_weights": torch.full((b * q, 32), 1 / 32, dtype=torch.bfloat16),
        "query_scale": torch.ones(b * q, 32),
        "index_key": index_key,
        "key_scale": key_scale,
        "source_table": table.int(),
        "query_ends": torch.arange(q, b * q + 1, q, dtype=torch.int32),
        "seq_lens": torch.full((b,), n, dtype=torch.int32),
        "canonical_ids": torch.full((b * q, 1, 2048), -1, dtype=torch.int32),
    }


def validate_ids(torch, trace, ids, args):
    ids = ids.detach().cpu().reshape(args.batch * args.query_width, -1).long()
    limits = torch.arange(args.context - args.query_width + 1, args.context + 1).repeat(
        args.batch
    )
    trace.require(ids.shape[1] == 2048, "TopK must contain 2048 entries")
    trace.require(
        bool(((ids == -1) | ((ids >= 0) & (ids < limits[:, None]))).all()),
        "noncausal or out-of-range TopK",
    )
    for row, limit in zip(ids, limits):
        live = row[row >= 0]
        trace.require(
            live.numel() == min(2048, int(limit)), "incorrect valid TopK count"
        )
        trace.require(live.unique().numel() == live.numel(), "duplicate TopK token")
    return ids


def check_pre(torch, trace, manager, args, previous_prefix):
    b, q, n, c = args.batch, args.query_width, args.context, args.capacity
    prefix = (n - q) // 128 * 128
    sparse = prefix >= c
    expected = {
        "safe_slots": torch.arange(1, b + 1, dtype=torch.int32),
        "lengths": torch.full((b,), n, dtype=torch.int32),
        "prefixes": torch.full((b,), prefix, dtype=torch.int32),
        "sparse_rows": torch.full((b,), sparse, dtype=torch.bool),
        "valid_rows": torch.ones(b, dtype=torch.bool),
        "request_states": (
            torch.where(
                (previous_prefix >= 0) & (prefix >= previous_prefix), -1, -2
            ).int()
            if sparse
            else torch.full((b,), -3, dtype=torch.int32)
        ),
        "next_prefixes": torch.full((b,), prefix if sparse else -1, dtype=torch.int32),
        "cache_sizes": torch.full((b,), c if sparse else 0, dtype=torch.int32),
    }
    for name, value in expected.items():
        trace.check("lim_pre." + name, manager.lim_adapter.pre[name][:b], value)


def execute(torch, torch_npu, args, trace):
    import sglang
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import free_fused_shm
    from sglang.srt.hardware_backend.npu import memory_pool_npu
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
        c8 as attention_module,
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
        c8_manager as manager_module,
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import (
        c8_metadata,
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_budget import (
        C8BudgetPlan,
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_mtp import (
        use_native_mtp_verify,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

    root = args.framework_prefix.resolve()
    if not Path(sglang.__file__).resolve().is_relative_to(root):
        raise RuntimeError(
            f"framework resolved outside --framework-prefix: {sglang.__file__}"
        )
    trace.report["metadata"]["framework_module"] = sglang.__file__
    device = torch.device(f"npu:{args.device}")
    trace.stage = "fixture"
    cpu = build_inputs(torch, args)
    pool = SimpleNamespace(
        dsa_kv_cache_store_fp8=True,
        page_size=128,
        indexer_layer_ids=(0,),
        start_layer=0,
        layer_num=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dtype=torch.float8_e4m3fn,
        get_index_k_buffer=lambda _: cpu["index_key"].to(device),
    )
    pool._pack_dsa_fp8_kv_cache = MethodType(
        memory_pool_npu.NPUMLATokenToKVPool._pack_dsa_fp8_kv_cache, pool
    )
    # Produce immutable native KV with the very same packer used by offload_v2.
    cpu["packed"] = (
        pool._pack_dsa_fp8_kv_cache(
            cpu["dense_k"].to(device), cpu["dense_rope"].to(device)
        )
        .view(torch.int8)
        .view(-1, 128, 1, 656)
        .cpu()
    )
    selected = {
        k: v for k, v in cpu.items() if k not in ("dense_k", "dense_rope", "reqmap")
    }
    native = _build_native(torch, torch_npu, {"tensors": selected}, device)
    values = native["values"]
    # Reuse the live index buffer rather than allocate one in every getter call.
    pool.get_index_k_buffer = lambda _: values["index_key"]
    pool.get_index_k_scale_buffer = lambda _: values["key_scale"]
    requests = SimpleNamespace(
        req_to_token=cpu["reqmap"].to(device), device=device, alloc=lambda _: None
    )
    budget = C8BudgetPlan(
        request_rows=args.batch + 1,
        context_len=cpu["reqmap"].shape[1],
        num_layers=1,
        top_k=2048,
        copy_sfa_query_heads=args.heads,
        max_query_width=args.query_width,
        batch_capacity=args.batch,
        cache_tokens=args.capacity,
    )
    manager = None
    try:
        trace.stage = "manager_setup"
        manager = manager_module.C8SparseKVCacheManager(
            requests,
            SimpleNamespace(get_kvcache=lambda: pool),
            2048,
            paged_kv_cache=pool,
            budget_plan=budget,
        )
        manager.device_kv_buffer[0].fill_(-91)
        manager.host_kv_buffer[0].view_as(cpu["packed"]).copy_(cpu["packed"])
        b, q = args.batch, args.query_width
        reqs = torch.arange(1, b + 1, dtype=torch.int32)
        batch = SimpleNamespace(
            batch_size=b,
            forward_mode=ForwardMode.TARGET_VERIFY if q == 4 else ForwardMode.DECODE,
            spec_info=SimpleNamespace(num_tokens_per_req=4) if q == 4 else None,
            req_pool_indices=reqs.to(device),
            req_pool_indices_cpu=reqs,
            seq_lens=values["seq_lens"],
            seq_lens_cpu=cpu["seq_lens"],
            positions=values["positions"],
            out_cache_loc=values["locations"],
        )
        backend = SimpleNamespace(
            graph_mode=False,
            device=device,
            sparse_kv_manager=manager,
            forward_metadata=SimpleNamespace(
                seq_lens_cpu_int=cpu["seq_lens"],
                actual_seq_lengths_q=values["query_ends"],
                sparse_kv_lim_plan=None,
            ),
        )
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=args.heads, tp_k_head_num=1, scaling=576**-0.5
        )
        current_k = cpu["dense_k"][cpu["locations"]].to(device)
        current_rope = cpu["dense_rope"][cpu["locations"]].to(device)
        is_copy = use_native_mtp_verify(backend, batch, manager) or (
            q == 1 and args.context >= args.capacity + 128
        )
        uses_lim = q == 1 or use_native_mtp_verify(backend, batch, manager)
        trace.report["metadata"].update(
            copy_sfa_expected=is_copy, lim_expected=uses_lim
        )
        torch.save(
            {
                **cpu,
                "note": "packed is the immutable native reference; page zero is reserved",
            },
            args.output / "inputs.pt",
        )

        for phase in ("cold", "warm"):
            trace.begin(phase)
            # Exercise actual publication: current Host rows must be overwritten.
            manager.host_kv_buffer[0].view(-1, 656)[cpu["locations"]] = -73
            trace.stage = "native_li_sfa"
            native["chain"]()
            torch.npu.synchronize()
            native_ids = native["holder"]["ids"].clone()
            native_output = native["holder"]["chain_output"].clone()
            native_cpu = validate_ids(torch, trace, native_ids, args)
            trace.save("native.ids", native_ids)
            trace.save("native.output", native_output)
            plan = None
            if uses_lim:
                trace.stage = "lim_select"
                previous_prefix = (
                    manager.lim_adapter.last_prefix[0][batch.req_pool_indices.long()]
                    .cpu()
                    .clone()
                )
                select = (
                    manager.lim_adapter.select_mtp
                    if q == 4
                    else manager.lim_adapter.select
                )
                plan = select(
                    manager,
                    query=values["index_query"],
                    weights=values["index_weights"],
                    key=values["index_key"],
                    block_table=values["source_table"],
                    forward_batch=batch,
                    layer_id=0,
                    query_scale=values["query_scale"],
                    key_scale=values["key_scale"],
                )
                torch.npu.synchronize()
                check_pre(torch, trace, manager, args, previous_prefix)
                ids = plan.source_ids
                lim_cpu = validate_ids(torch, trace, ids, args)
                trace.check(
                    "lim.topk_set", lim_cpu.sort(-1).values, native_cpu.sort(-1).values
                )
                trace.save("lim.source_ids", ids)
                trace.save("lim.device_slots", plan.device_slots)
                trace.save("lim.slot_map", manager.device_slot_map[0])
                if is_copy:
                    trace.observe("miss_counts", plan.miss_counts.cpu().tolist())
                    trace.observe("topk_misses", plan.topk_misses.cpu().tolist())
                    trace.save("lim.miss_sources", plan.miss_sources)
                    trace.save("lim.miss_slots", plan.miss_slots)
                    trace.require(
                        bool(
                            (
                                (plan.miss_counts >= 0)
                                & (plan.miss_counts <= args.capacity)
                            ).all()
                        ),
                        "invalid union miss count",
                    )
                    trace.require(
                        bool(
                            ((plan.topk_misses >= 0) & (plan.topk_misses <= 2048)).all()
                        ),
                        "invalid route miss count",
                    )
                    for request, count in enumerate(plan.miss_counts.cpu().tolist()):
                        sources = plan.miss_sources[request, :count]
                        slots = plan.miss_slots[request, :count]
                        trace.require(
                            bool(
                                (
                                    (sources >= 0)
                                    & (sources < (args.context - q) // 128 * 128)
                                ).all()
                            ),
                            "invalid union miss source",
                        )
                        trace.require(
                            bool(((slots >= 0) & (slots < args.capacity)).all()),
                            "invalid union miss destination",
                        )
                    trace.require(
                        bool(
                            (
                                (plan.device_slots >= 0)
                                & (
                                    plan.device_slots
                                    < args.capacity
                                    + args.context
                                    - (args.context - q) // 128 * 128
                                )
                            ).all()
                        ),
                        "invalid CopySFA device slot",
                    )
            else:
                # Match dsa_npu_indexer: short Q4 bypasses LIM and uses native LI.
                ids = native_ids.reshape(b * q, 2048)
                trace.observe(
                    "lim_bypassed", "short Q4 uses native quantized LI in the framework"
                )
            backend.forward_metadata.sparse_kv_lim_plan = plan
            hooks = {"publication": 0, "tail": 0, "metadata": 0, "fallback": 0}
            publication = manager_module.unidex_copy_inplace
            tail = c8_metadata.stage_mtp_tail
            metadata = manager.lim_adapter.copy_sfa_metadata
            vendor_attention = torch_npu.npu_kv_quant_sparse_flash_attention

            def publish_hook(*a, **kw):
                trace.stage = "host_publication"
                result = publication(*a, **kw)
                torch.npu.synchronize()
                hooks["publication"] += 1
                for name, actual, expected in (
                    ("src_index", a[2], torch.arange(b * q, dtype=torch.int64)),
                    ("dst_index", a[3], cpu["locations"]),
                    ("valid", a[4], torch.ones(b * q, dtype=torch.bool)),
                ):
                    trace.check(
                        "publication." + name, actual, expected.to(actual.dtype)
                    )
                trace.check(
                    "publication.packed_current",
                    a[0].view(-1, 656),
                    cpu["packed"].view(-1, 656)[cpu["locations"]],
                )
                trace.check(
                    "publication.host_current",
                    manager.host_kv_buffer[0].view(-1, 656)[cpu["locations"]],
                    cpu["packed"].view(-1, 656)[cpu["locations"]],
                )
                return result

            def tail_hook(**kw):
                trace.stage = "ring_tail"
                result = tail(**kw)
                torch.npu.synchronize()
                hooks["tail"] += 1
                prefix = (args.context - q) // 128 * 128
                positions = torch.arange(prefix, args.context).repeat(b, 1)
                actual = manager.device_kv_buffer[0].cpu()[
                    reqs.long()[:, None], args.capacity + positions % 256, 0
                ]
                expected = gather_logical(
                    torch, cpu["packed"], cpu["source_table"], positions, 1
                )
                trace.check("tail.packed_kv", actual, expected)
                trace.check(
                    "tail.starts",
                    manager.tail_starts[0][batch.req_pool_indices.long()],
                    torch.full((b,), prefix, dtype=torch.int32),
                )
                trace.check(
                    "tail.ends",
                    manager.tail_ends[0][batch.req_pool_indices.long()],
                    torch.full((b,), args.context, dtype=torch.int32),
                )
                for i, name in enumerate(
                    (
                        "history_src",
                        "history_dst",
                        "history_valid",
                        "current_src",
                        "current_dst",
                        "current_valid",
                    )
                ):
                    trace.save("tail." + name, manager.tail_workspace[i])
                return result

            def metadata_hook(owner, selection):
                trace.stage = "copy_metadata"
                result = metadata(owner, selection)
                torch.npu.synchronize()
                hooks["metadata"] += 1
                prefix = (args.context - q) // 128 * 128
                offsets = torch.arange((args.capacity + 256) // 128)
                cp = args.capacity // 128
                pages = torch.where(
                    offsets < cp, offsets, cp + (prefix // 128 + offsets - cp) % 2
                )
                expected = (reqs[:, None] * offsets.numel() + pages[None]).int()
                trace.check(
                    "copy_metadata.hbm_table",
                    owner.planner_hbm_block_table[:b],
                    expected,
                )
                trace.check(
                    "copy_metadata.host_table",
                    owner.planner_dram_block_table[:b],
                    cpu["source_table"],
                )
                trace.check(
                    "copy_metadata.seq_lens",
                    owner.planner_actual_seq_lengths_kv[:b],
                    cpu["seq_lens"],
                )
                trace.check(
                    "copy_metadata.cache_lens",
                    owner.planner_num_cache_tokens[:b],
                    torch.full(
                        (b,), args.capacity + args.context - prefix, dtype=torch.int32
                    ),
                )
                return result

            def fallback_hook(**kw):
                trace.stage = "fallback_gather"
                hooks["fallback"] += 1
                logical = torch.arange(args.context).repeat(b, 1)
                gathered = gather_logical(
                    torch, kw["key"].view(torch.int8), kw["block_table"], logical, 1
                )
                expected = gather_logical(
                    torch, cpu["packed"], cpu["source_table"], logical, 1
                )
                trace.check("fallback.gathered_kv", gathered, expected)
                trace.check("fallback.query", kw["query"], cpu["merged_query"])
                trace.check(
                    "fallback.topk",
                    kw["sparse_indices"]
                    .reshape(b * q, 2048)
                    .cpu()
                    .long()
                    .sort(-1)
                    .values,
                    native_cpu.sort(-1).values,
                )
                return vendor_attention(**kw)

            trace.stage = "framework_forward"
            with (
                patch.object(manager_module, "unidex_copy_inplace", publish_hook),
                patch.object(c8_metadata, "stage_mtp_tail", tail_hook),
                patch.object(manager.lim_adapter, "copy_sfa_metadata", metadata_hook),
                patch.object(
                    torch_npu, "npu_kv_quant_sparse_flash_attention", fallback_hook
                ),
            ):
                output = attention_module.forward_c8_offload(
                    backend,
                    values["q"],
                    current_k,
                    layer,
                    batch,
                    True,
                    values["q_rope"],
                    current_rope,
                    ids,
                )
                torch.npu.synchronize()
            expected_hooks = {
                "publication": 1,
                "tail": 1,
                "metadata": int(is_copy),
                "fallback": int(not is_copy),
            }
            trace.observe("hooks", hooks)
            trace.require(
                hooks == expected_hooks,
                f"incomplete path coverage: {hooks}; expected {expected_hooks}",
            )
            trace.observe(
                "path", manager.last_decode_path if is_copy else "short_vendor_qsfa"
            )
            if is_copy:
                trace.check(
                    "copy.query_packing",
                    manager.c8_cache.query[: b * q],
                    cpu["merged_query"],
                )
                actual = gather_logical(
                    torch,
                    manager.device_kv_buffer[0],
                    manager.planner_hbm_block_table[:b],
                    plan.device_slots,
                    q,
                )
                expected = gather_logical(
                    torch, cpu["packed"], cpu["source_table"], plan.source_ids, q
                )
                trace.check("copy.selected_packed_kv", actual, expected)
                # Keep decoded payloads as well as bytes for interactive inspection.
                ak, av = unpack_rows(torch, actual.reshape(-1, 656))
                ek, ev = unpack_rows(torch, expected.reshape(-1, 656))
                trace.check("copy.selected_key", ak, ek)
                trace.check("copy.selected_value", av, ev)
            tolerance = dict(
                exact=False, atol=args.atol, rtol=args.rtol, l2=args.max_relative_l2
            )
            output = output.reshape(b * q, args.heads, 512)
            trace.check(
                "attention.independent_topk_output",
                output,
                native_output.reshape_as(output),
                **tolerance,
            )
            # Isolate attention arithmetic from any indexer selection difference.
            values["canonical_ids"].copy_(ids.reshape(b * q, 1, 2048))
            trace.stage = "native_fixed_topk_sfa"
            native["attention"]()
            torch.npu.synchronize()
            trace.check(
                "attention.same_topk_output",
                output,
                native["holder"]["standalone_output"].reshape_as(output),
                **tolerance,
            )
            trace.flush()
        trace.report["complete"] = True
    finally:
        trace.flush()
        # This is an isolated process; free only its own registered Host storage.
        if manager is not None:
            free_fused_shm(args.device)


def run(args):
    trace = None
    try:
        # Match the current baseline deliberately; do not test optional v2 here.
        flags = {
            "SGLANG_NPU_A5_C8_SPLIT_QUERY": "0",
            "SGLANG_NPU_A5_C8_FUSED_RING_TAIL": "0",
            "SGLANG_NPU_A5_PD_MF": "0",
            "SGLANG_NPU_A5_C8_RECORD_HITS": "0",
            "SGLANG_NPU_FUSED_LIM_PRE_METADATA": "1",
            "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA": "1",
            "SGL_KERNEL_A5_C8_BACKEND": "original",
        }
        previous = {k: os.environ.get(k) for k in flags}
        os.environ.update(flags)
        _add_prefix(args.framework_prefix)
        # Load the wheel before runtime initialization so original OPP is visible.
        # isort: off
        import sgl_kernel_npu
        import torch
        import torch_npu
        import acl

        # isort: on

        torch.npu.set_device(args.device)
        if acl.rt.get_device_info(args.device, 601) != (3510, 0):
            raise RuntimeError("This diagnostic requires an Ascend950 A5 device")
        library = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
        metadata = {
            "scope": "synthetic one-layer A5 packed-C8 eager, cold then identical warm replay; no model weights or graph",
            "framework_head": _git_sha(args.framework_prefix),
            "kernel_source_head": _git_sha(Path(__file__).resolve().parents[2]),
            "kernel_module": sgl_kernel_npu.__file__,
            "native_library": str(library),
            "native_sha256": _sha256(library),
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(torch_npu.__version__),
            "flags": flags,
            "previous_flags": previous,
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        }
        trace = TensorTrace(torch, args.output, metadata)
        execute(torch, torch_npu, args, trace)
        trace.write()
        return (0 if trace.report["passed"] else 1), trace.report
    except Exception:
        error = traceback.format_exc()
        if trace is not None:
            trace.report["error"] = {"stage": trace.stage, "traceback": error}
            trace.flush()
        else:
            (args.output / "setup_error.txt").write_text(error, encoding="utf-8")
        print(error, file=sys.stderr)
        report = (
            trace.report
            if trace is not None
            else {"passed": False, "error": {"stage": "setup", "traceback": error}}
        )
        return 2, report


@contextmanager
def capture_log(path):
    """Capture Python and native-library output without flooding the terminal."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with path.open("w", encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                with redirect_stdout(log), redirect_stderr(log):
                    yield
            finally:
                log.flush()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def main():
    args = parse_args()
    if args.dry_run:
        print(
            f"A5 C8 alignment: Q{args.query_width}, batch={args.batch}, context={args.context}, C={args.capacity}, device={args.device}; dry run only"
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    with capture_log(args.output / "run.log"):
        code, report = run(args)
    text = summarize(report) + f"Details and tensors: {args.output.resolve()}\n"
    (args.output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
