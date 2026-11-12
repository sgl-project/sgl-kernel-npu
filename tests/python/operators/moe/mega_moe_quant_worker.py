"""Standalone four-rank quantized MegaMoE cases driven by mega_moe_worker."""

from contextlib import ExitStack
from itertools import product

QUANT_MODES = {
    "arch22": ("w4a8", "w8a8"),
    "arch35": ("mx_fp8_e4m3", "mx_fp8_e5m2", "mx_fp4_e2m1", "mx_fp8_e4m3_w4"),
}


def make_int_layer(torch, torch_npu, rows, cols, count, device, int4):
    weights, scales, biases = [], [], []
    lower, upper = (-8, 8) if int4 else (-127, 128)
    for _ in range(count):
        raw = torch.randint(lower, upper, (rows, cols), dtype=torch.int8)
        # Powers-of-two multiples remain exact in the packed scale field.
        scale = torch.randint(1, 5, (cols,)).float() * (2**-6 if int4 else 2**-11)
        scale_bits = scale.view(torch.int32).to(torch.int64)
        if int4:
            # Signed INT4 values occupy consecutive low and high nibbles.
            packed = (raw[:, ::2] & 15) | ((raw[:, 1::2] & 15) << 4)
            weight = torch_npu.npu_format_cast(packed.to(device), 29).view(torch.int32)
            biases.append((8 * raw.float().sum(dim=0) * scale).to(device))
            scale_bits |= 1 << 46
        else:
            weight = torch_npu.npu_format_cast(raw.to(device), 29)
        weights.append(weight)
        scales.append(scale_bits.to(device).view(torch.uint64))
    return weights, scales, biases or None


def make_mx_layer(
    torch, torch_npu, rows, cols, count, device, dtype, fp4, c0_32, layout, storage
):
    prefix = (count,) if layout == "stacked" else ()
    weight_shape = (*prefix, rows, cols // 2 if fp4 else cols)
    scale_shape = (*prefix, rows, (cols + 63) // 64, 2)
    weights, scales = [], []
    for _ in range(1 if prefix else count):
        if fp4:
            weight = torch.randint(
                0, 256, weight_shape, dtype=torch.uint8, device=device
            )
        else:
            weight = (torch.randn(weight_shape, device=device) * 0.125).to(dtype)
        if c0_32:
            weight = torch_npu.npu_format_cast(
                weight,
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
        elif storage == "NZ":
            weight = torch_npu.npu_format_cast(weight, 29)
        weights.append(weight)
        # E8M0 scales are stored in pairs of 32-element K blocks.
        exponents = torch.randint(
            125, 130, scale_shape, dtype=torch.uint8, device=device
        )
        scales.append(exponents.view(torch.float8_e8m0fnu))
    return weights, scales


def run_quantized_case(
    torch, dist, torch_npu, group, args, device, generate_data, measure_latency
):
    from cann_ops_transformer.ops import (
        get_symm_buffer_for_mega_moe as create_reference,
    )
    from cann_ops_transformer.ops import mega_moe as reference
    from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe, mega_moe

    mode = args.quant_mode
    mx = mode in QUANT_MODES["arch35"]
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local_experts = args.num_experts // world_size
    hidden, intermediate = args.hidden, args.intermediate
    tokens = args.tokens + (rank if mx else 0)
    x, ids, routing = generate_data(torch, args, rank, tokens)
    counts = torch.bincount(ids.flatten(), minlength=args.num_experts)
    counts = counts.to(device=device, dtype=torch.int32)
    dist.all_reduce(counts, group=group)
    expected_counts = counts[rank * local_experts : (rank + 1) * local_experts].cpu()

    active = None
    if not mx:
        pad = args.capacity - tokens
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        ids = torch.nn.functional.pad(ids, (0, 0, 0, pad))
        routing = torch.nn.functional.pad(routing, (0, 0, 0, pad))
        active = (torch.arange(args.capacity, device=device) < tokens).to(torch.int8)
    x = x.to(device)
    ids = ids.to(device=device, dtype=torch.int32)
    routing = routing.to(device)
    torch.manual_seed(4200 + rank)

    if mx:
        fp4 = mode in ("mx_fp4_e2m1", "mx_fp8_e4m3_w4")
        weight_type = torch_npu.float4_e2m1fn_x2 if fp4 else None
        dtype = torch.float8_e5m2 if mode == "mx_fp8_e5m2" else torch.float8_e4m3fn
        dispatch_dtype = weight_type if mode == "mx_fp4_e2m1" else dtype
        formats = {
            "mx_fp8_e4m3": ("ND", "NZ"),
            "mx_fp8_e5m2": ("ND",),
            "mx_fp4_e2m1": ("NZ",),
            "mx_fp8_e4m3_w4": ("NZ",),
        }[mode]
        scenes = product(("per-expert", "stacked"), formats)
    else:
        dispatch_dtype = torch.int8
        scenes = [("per-expert", "NZ")]

    buffer_options = dict(
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.capacity,
        num_topk=args.topk,
        hidden=hidden,
        intermediate_hidden=intermediate,
        max_recv_token_num=0,
        dispatch_quant_mode=4 if mx else 2,
        dispatch_quant_out_dtype=dispatch_dtype,
    )
    # Each implementation owns its own symmetric buffer and communicator.
    reference_group = dist.new_group(backend="hccl")
    try:
        # The provider looks up an existing HCCL handle. Unlike the test group
        # above, the new reference group has not performed any collectives yet.
        reference_group._get_backend(device).get_hccl_comm_name(rank)
        with ExitStack() as cleanup:
            ref_buffer = create_reference(reference_group, **buffer_options)
            cleanup.callback(ref_buffer.destroy)
            buffer = get_symm_buffer_for_mega_moe(
                group, impl=args.impl, **buffer_options
            )
            cleanup.callback(buffer.destroy)
            cleanup.callback(torch.npu.synchronize)

            for scene_index, (layout, storage) in enumerate(scenes):
                inputs = dict(x=x, topk_ids=ids, x_active_mask=active)
                if mx:
                    mx_options = dict(
                        dtype=dtype, fp4=fp4, layout=layout, storage=storage
                    )
                    w1, s1 = make_mx_layer(
                        torch,
                        torch_npu,
                        2 * intermediate,
                        hidden,
                        local_experts,
                        device,
                        c0_32=mode == "mx_fp8_e4m3_w4",
                        **mx_options,
                    )
                    w2, s2 = make_mx_layer(
                        torch,
                        torch_npu,
                        hidden,
                        intermediate,
                        local_experts,
                        device,
                        c0_32=fp4,
                        **mx_options,
                    )
                    inputs.update(weight1_type=weight_type, weight2_type=weight_type)
                    b1 = b2 = None
                    inputs["topk_weights"] = (
                        routing if layout == "stacked" else routing.to(torch.bfloat16)
                    )
                else:
                    w1, s1, b1 = make_int_layer(
                        torch,
                        torch_npu,
                        hidden,
                        2 * intermediate,
                        local_experts,
                        device,
                        mode == "w4a8",
                    )
                    w2, s2, b2 = make_int_layer(
                        torch,
                        torch_npu,
                        intermediate,
                        hidden,
                        local_experts,
                        device,
                        mode == "w4a8",
                    )
                    inputs["topk_weights"] = routing
                inputs.update(
                    l1_weights=w1,
                    l2_weights=w2,
                    l1_weights_sf=s1,
                    l2_weights_sf=s2,
                    l1_bias=b1,
                    l2_bias=b2,
                )
                activations = ("situglu",) if mx else ("swiglu", "situglu")
                for activation in activations:
                    inputs["activation"] = activation
                    inputs["activation_params"] = (
                        {"beta": 4.0, "linear_beta": 25.0}
                        if activation == "situglu"
                        else None
                    )
                    expected, ref_counts = reference(sym_buffer=ref_buffer, **inputs)
                    actual, actual_counts = mega_moe(
                        sym_buffer=buffer, impl=args.impl, **inputs
                    )
                    torch.npu.synchronize()
                    assert actual.shape == expected.shape == x.shape
                    actual_active = actual[:tokens].cpu()
                    assert torch.isfinite(actual_active).all()
                    torch.testing.assert_close(
                        actual_active, expected[:tokens].cpu(), rtol=1e-3, atol=1e-2
                    )
                    for expert_counts in (ref_counts, actual_counts):
                        torch.testing.assert_close(
                            expert_counts.reshape(-1).cpu().to(torch.int32),
                            expected_counts,
                            rtol=0,
                            atol=0,
                        )
                    # A failing rank must prevent the others from entering timing.
                    dist.barrier(group=group)
                    if rank == 0:
                        print(
                            f"{mode} {layout} {storage} {activation} impl={args.impl} PASS",
                            flush=True,
                        )
                    if args.perf_iters and scene_index == 0:
                        measure_latency(
                            torch,
                            dist,
                            group,
                            lambda: mega_moe(
                                sym_buffer=buffer, impl=args.impl, **inputs
                            ),
                            device,
                            args.perf_iters,
                        )
    finally:
        dist.destroy_process_group(reference_group)
