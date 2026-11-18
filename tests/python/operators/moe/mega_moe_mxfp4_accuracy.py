"""Independent CPU reference for four-rank A5 A8W4-FP accuracy.

The reference decodes FP4 E2M1/E8M0 and computes dense FP32 expert FFNs.
MXFP8 E4M3 uses 32-element blocks, floor(max exponent)-8 scales, nearest
even conversion and finite saturation. SiTU output is rounded to BF16
before the second MX quantization, matching the public operator dataflow.
"""

from contextlib import ExitStack


def decode_weight(torch, packed, scales):
    low, high = packed & 15, packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    factors = torch.exp2(scales.reshape(scales.shape[0], -1).float() - 127)
    return values * factors.repeat_interleave(32, dim=-1)


def round_mxfp8(torch, values):
    blocks = values.float().reshape(values.shape[0], -1, 32)
    largest = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.floor(torch.log2(largest.clamp_min(2**-126))) - 8
    scale = torch.exp2(exponent.clamp(-127, 127))
    rounded = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (rounded * scale).reshape(values.shape)


def make_expert(torch, expert, hidden, intermediate):
    generator = torch.Generator(device="cpu").manual_seed(8300 + expert)
    layers = []
    for rows, cols in ((2 * intermediate, hidden), (hidden, intermediate)):
        packed = torch.randint(
            0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8
        )
        scales = torch.randint(
            118, 123, (rows, cols // 64, 2), generator=generator, dtype=torch.uint8
        )
        layers.append((packed, scales))
    return layers


def reference_moe(torch, x, ids, routing, hidden, intermediate):
    quantized_x = round_mxfp8(torch, x)
    result = torch.zeros(x.shape, dtype=torch.float32)
    for expert in ids.unique().tolist():
        tokens, slots = torch.where(ids == expert)
        first, second = make_expert(torch, expert, hidden, intermediate)
        w1 = decode_weight(torch, *first)
        w2 = decode_weight(torch, *second)
        gate, up = (
            (quantized_x[tokens] @ w1.T).to(torch.bfloat16).float().chunk(2, dim=-1)
        )
        activated = (
            4.0
            * torch.tanh(gate / 4.0)
            * torch.sigmoid(gate)
            * (25.0 * torch.tanh(up / 25.0))
        ).to(torch.bfloat16)
        projected = (round_mxfp8(torch, activated) @ w2.T).to(torch.bfloat16).float()
        result.index_add_(0, tokens, projected * routing[tokens, slots, None])
    return result


def run_accuracy(torch, dist, torch_npu, group, args, device, generate_data):
    from sgl_kernel_npu.operators.moe import get_symm_buffer_for_mega_moe, mega_moe

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    tokens = args.tokens + rank
    x, ids, routing = generate_data(torch, args, rank, tokens)
    expected = reference_moe(torch, x, ids, routing, args.hidden, args.intermediate)
    local_experts = args.num_experts // world_size
    counts = torch.bincount(ids.flatten(), minlength=args.num_experts).to(device)
    dist.all_reduce(counts, group=group)
    expected_counts = counts[rank * local_experts : (rank + 1) * local_experts].cpu()

    weights, scales = [[], []], [[], []]
    for expert in range(rank * local_experts, (rank + 1) * local_experts):
        for layer, (packed, exponent_bytes) in enumerate(
            make_expert(torch, expert, args.hidden, args.intermediate)
        ):
            weights[layer].append(
                torch_npu.npu_format_cast(
                    packed.to(device),
                    29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
            )
            scales[layer].append(exponent_bytes.to(device).view(torch.float8_e8m0fnu))

    group._get_backend(device).get_hccl_comm_name(rank)
    with ExitStack() as cleanup:
        buffer = get_symm_buffer_for_mega_moe(
            group,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.capacity,
            num_topk=args.topk,
            hidden=args.hidden,
            intermediate_hidden=args.intermediate,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=torch.float8_e4m3fn,
            impl=args.impl,
        )
        cleanup.callback(buffer.destroy)
        cleanup.callback(torch.npu.synchronize)
        output, actual_counts = mega_moe(
            x=x.to(device),
            topk_ids=ids.to(device=device, dtype=torch.int32),
            topk_weights=routing.to(device),
            l1_weights=weights[0],
            l2_weights=weights[1],
            l1_weights_sf=scales[0],
            l2_weights_sf=scales[1],
            sym_buffer=buffer,
            weight1_type=torch_npu.float4_e2m1fn_x2,
            weight2_type=torch_npu.float4_e2m1fn_x2,
            activation="situglu",
            activation_params={"beta": 4.0, "linear_beta": 25.0},
            impl=args.impl,
        )
        torch.npu.synchronize()
        assert output.shape == x.shape and output.dtype == torch.bfloat16
        actual = output.float().cpu()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual_counts.reshape(-1).cpu().long(),
            expected_counts.long(),
            rtol=0,
            atol=0,
        )
        signal = expected.square().mean().sqrt()
        assert signal.item() > 1e-6, "Reference must produce a nonzero signal"
        relative_rms = (actual - expected).square().mean().sqrt() / signal
        # MX rounding boundaries and SiTU approximation can affect individual
        # entries. Bound both the global error and each element relative to RMS.
        torch.testing.assert_close(
            actual, expected, rtol=0.1, atol=0.05 * signal.item()
        )
        assert relative_rms.item() < 0.035, relative_rms.item()
        print(
            f"rank={rank} independent A5 MXFP4 reference "
            f"relative_rms={relative_rms.item():.6f} PASS",
            flush=True,
        )
