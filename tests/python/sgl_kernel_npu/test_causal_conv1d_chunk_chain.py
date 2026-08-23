"""Production-length chunked-prefill regression for the native causal conv1d.

This mirrors the Qwen3.6 TP=2 path: a four-tap convolution over 4096 local
channels with the state deliberately reused across chunks.  It covers both
the radix-aligned 16384/12672/128 chain and the no-radix 16384/12800 chain.
"""

import torch

import sgl_kernel_npu  # noqa: F401 - registers torch.ops.npu.causal_conv1d
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu


@torch.no_grad()
def test_qwen36_production_chunk_chain_exact():
    device = torch.device("npu")
    dtype = torch.bfloat16
    dim = 4096
    width = 4
    cache_slot = 3
    num_cache_lines = 8

    chunk_chains = {
        "radix": [(16384, 16384), (12672, 12672), (128, 67)],
        "no_radix": [(16384, 16384), (12800, 12739)],
    }

    for chain_name, chunks in chunk_chains.items():
        torch.manual_seed(20260823)
        weight_native = (
            torch.randn((width, dim), device=device, dtype=dtype) * 0.02
        )
        bias = torch.randn((dim,), device=device, dtype=dtype) * 0.02
        native_state = torch.zeros(
            (num_cache_lines, width - 1, dim), device=device, dtype=dtype
        )
        reference_state = native_state.transpose(1, 2).contiguous()
        cache_indices = torch.tensor(
            [cache_slot], device=device, dtype=torch.int32
        )

        for chunk_id, (physical_len, valid_len) in enumerate(chunks):
            x = (
                torch.randn((physical_len, dim), device=device, dtype=dtype)
                * 0.02
            )
            query_start_loc = torch.tensor(
                [0, valid_len], device=device, dtype=torch.int32
            )
            has_initial_state = torch.tensor(
                [chunk_id != 0], device=device, dtype=torch.bool
            )

            native_out = torch.ops.npu.causal_conv1d(
                x,
                weight_native,
                conv_states=native_state,
                bias=bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
            reference_out = causal_conv1d_fn_npu(
                x.transpose(0, 1).contiguous(),
                weight_native.transpose(0, 1).contiguous(),
                bias=bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                conv_states=reference_state,
                activation="silu",
                pad_slot_id=-1,
            ).transpose(0, 1)
            torch.npu.synchronize()

            assert torch.equal(native_out[:valid_len], reference_out[:valid_len]), (
                f"{chain_name} chunk {chunk_id} valid output differs: max_abs="
                f"{(native_out[:valid_len].float() - reference_out[:valid_len].float()).abs().max().item()}"
            )
            assert torch.equal(
                native_state[cache_slot],
                reference_state[cache_slot].transpose(0, 1),
            ), f"{chain_name} chunk {chunk_id} state differs"
            assert torch.count_nonzero(native_out[valid_len:]).item() == 0, (
                f"{chain_name} chunk {chunk_id} physical tail was not zeroed"
            )
