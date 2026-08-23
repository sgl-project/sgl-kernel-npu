import argparse

import torch
import torch.nn.functional as F

import sgl_kernel_npu  # noqa: F401 - registers torch.ops.npu.causal_conv1d
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_v2


def _device_int64(values):
    out = torch.empty(len(values), device="npu", dtype=torch.int64)
    for index, value in enumerate(values):
        out[index] = value
    return out


def _device_bool(values):
    out = torch.empty(len(values), device="npu", dtype=torch.bool)
    for index, value in enumerate(values):
        out[index] = value
    return out


def _reference_prefill(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
):
    """PR651/PyTorch semantics: BF16 conv result, then BF16 SiLU."""
    width, dim = weight.shape
    history = width - 1
    outputs = []
    for sequence in range(cache_indices.numel()):
        start = int(query_start_loc[sequence].item())
        end = int(query_start_loc[sequence + 1].item())
        cache_index = int(cache_indices[sequence].item())
        if bool(has_initial_state[sequence].item()):
            prefix = conv_states[cache_index, :history]
        else:
            prefix = torch.zeros(
                (history, dim), device=x.device, dtype=x.dtype
            )
        segment = x[start:end]
        full = torch.cat((prefix, segment), dim=0)
        convolution = F.conv1d(
            full.transpose(0, 1).unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(1),
            bias=bias,
            groups=dim,
        )
        outputs.append(F.silu(convolution).squeeze(0).transpose(0, 1))
        conv_states[cache_index, :history] = full[-history:]
    return torch.cat(outputs, dim=0)


def _assert_exact(label, actual, expected):
    torch.npu.synchronize()
    equal = actual == expected
    exact = int(equal.sum().item())
    total = equal.numel()
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    print(f"{label}: exact={exact}/{total}, max_abs={max_abs}")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _reference_update(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    num_accepted_tokens=None,
):
    width, dim = weight.shape
    history = width - 1
    outputs = []
    for sequence in range(cache_indices.numel()):
        start = int(query_start_loc[sequence].item())
        end = int(query_start_loc[sequence + 1].item())
        cache_index = int(cache_indices[sequence].item())
        segment = x[start:end]
        if num_accepted_tokens is None:
            offset = 0
        else:
            offset = int(num_accepted_tokens[sequence].item()) - 1
            offset = max(0, min(offset, conv_states.shape[1] - history))
        prefix = conv_states[cache_index, offset : offset + history].clone()
        full = torch.cat((prefix, segment), dim=0)
        convolution = F.conv1d(
            full.transpose(0, 1).unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(1),
            bias=bias,
            groups=dim,
        )
        outputs.append(F.silu(convolution).squeeze(0).transpose(0, 1))

        if num_accepted_tokens is None:
            conv_states[cache_index, :history] = full[-history:]
        else:
            # Width four speculative state layout: retain the two history
            # values after the accepted-token offset, then append all draft
            # inputs.  This mirrors WriteBackStateSpec.
            conv_states[cache_index, :2] = prefix[1:3]
            conv_states[cache_index, 2 : 2 + segment.shape[0]] = segment
    return torch.cat(outputs, dim=0)


def run_prefill_chain(dim, lengths, use_bias):
    torch.manual_seed(20260823)
    dtype = torch.bfloat16
    width = 4
    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = (
        torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    )
    native_state = torch.randn((8, width - 1, dim), device="npu", dtype=dtype)
    reference_state = native_state.clone()
    cache_indices = _device_int64([3])

    for chunk_index, length in enumerate(lengths):
        x = torch.randn((length, dim), device="npu", dtype=dtype)
        query_start_loc = _device_int64([0, length])
        has_initial_state = _device_bool([chunk_index != 0])

        expected = _reference_prefill(
            x,
            weight,
            bias,
            reference_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
        )
        actual = torch.ops.npu.causal_conv1d(
            x,
            weight,
            native_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )
        _assert_exact(f"prefill-output-{chunk_index}-{length}", actual, expected)
        _assert_exact(
            f"prefill-state-{chunk_index}-{length}",
            native_state,
            reference_state,
        )


def run_padded_prefill_tail(dim, padded_length, valid_length, use_bias):
    """Varlen padding must be deterministic zeros, like PR651.

    Scheduler/page alignment can pass 128 physical rows while query_start_loc
    marks only 67 real tokens.  Downstream GDN code retains the physical shape,
    so leaving the remaining rows from empty_like() uninitialized injects stale
    allocator data into the recurrent path.
    """
    assert 0 < valid_length < padded_length
    torch.manual_seed(20260828)
    dtype = torch.bfloat16
    width = 4
    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    native_state = torch.randn(
        (8, width - 1, dim), device="npu", dtype=dtype
    )
    reference_state = native_state.clone()
    x = torch.randn((padded_length, dim), device="npu", dtype=dtype)
    query_start_loc = _device_int64([0, valid_length])
    cache_indices = _device_int64([3])
    has_initial_state = _device_bool([False])

    expected_valid = _reference_prefill(
        x[:valid_length],
        weight,
        bias,
        reference_state,
        _device_int64([0, valid_length]),
        cache_indices,
        has_initial_state,
    )

    # Encourage empty_like() to reuse a visibly non-zero allocator block.
    poison = torch.full_like(x, 17.0)
    del poison
    actual = torch.ops.npu.causal_conv1d(
        x,
        weight,
        native_state,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )
    torch.npu.synchronize()

    _assert_exact("padded-prefill-valid", actual[:valid_length], expected_valid)
    _assert_exact("padded-prefill-state", native_state, reference_state)
    expected_padding = torch.zeros_like(actual[valid_length:])
    _assert_exact("padded-prefill-zero-tail", actual[valid_length:], expected_padding)


def run_decode_chain(dim, steps, use_bias):
    torch.manual_seed(20260824)
    dtype = torch.bfloat16
    width = 4
    batch = 2
    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = (
        torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    )
    native_state = torch.randn((8, width - 1, dim), device="npu", dtype=dtype)
    reference_state = native_state.clone()
    query_start_loc = _device_int64([0, 1, 2])
    cache_indices = _device_int64([2, 5])

    for step in range(steps):
        x = torch.randn((batch, dim), device="npu", dtype=dtype)
        expected = _reference_update(
            x,
            weight,
            bias,
            reference_state,
            query_start_loc,
            cache_indices,
        )
        actual = torch.ops.npu.causal_conv1d(
            x,
            weight,
            native_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(native_state, reference_state, rtol=0, atol=0)
    _assert_exact(f"decode-output-after-{steps}", actual, expected)
    _assert_exact(f"decode-state-after-{steps}", native_state, reference_state)


def run_speculative_update(dim, use_bias):
    torch.manual_seed(20260825)
    dtype = torch.bfloat16
    width = 4
    batch = 2
    draft_tokens = 4
    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = (
        torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    )
    native_state = torch.randn((8, 120, dim), device="npu", dtype=dtype)
    reference_state = native_state.clone()
    x = torch.randn((batch * draft_tokens, dim), device="npu", dtype=dtype)
    query_start_loc = _device_int64([0, draft_tokens, 2 * draft_tokens])
    cache_indices = _device_int64([1, 6])
    num_accepted_tokens = torch.empty(batch, device="npu", dtype=torch.int32)
    num_accepted_tokens[0] = 4
    num_accepted_tokens[1] = 2

    expected = _reference_update(
        x,
        weight,
        bias,
        reference_state,
        query_start_loc,
        cache_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    actual = torch.ops.npu.causal_conv1d(
        x,
        weight,
        native_state,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    _assert_exact("speculative-output", actual, expected)
    _assert_exact("speculative-state", native_state, reference_state)


def run_speculative_graph_replay(dim, steps, use_bias):
    """Keep update outputs/state bit-exact across repeated NPU graph replays."""
    torch.manual_seed(20260826)
    dtype = torch.bfloat16
    width = 4
    batch = 1
    draft_tokens = 4
    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = (
        torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    )
    state_len = width - 1 + draft_tokens - 1
    initial_state = torch.randn(
        (8, state_len, dim), device="npu", dtype=dtype
    )
    native_state = initial_state.clone()
    reference_state = initial_state.clone()
    staging_x = torch.empty(
        (batch * draft_tokens, dim), device="npu", dtype=dtype
    )
    query_start_loc = _device_int64([0, draft_tokens])
    cache_indices = _device_int64([3])
    num_accepted_tokens = torch.empty(batch, device="npu", dtype=torch.int32)
    num_accepted_tokens[0] = draft_tokens

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
        graph_output = torch.ops.npu.causal_conv1d(
            staging_x,
            weight,
            native_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
    torch.npu.synchronize()

    # Capture executes the graph once.  Restore the fixed-address state buffer
    # so the replay chain and the eager reference start from identical bytes.
    native_state.copy_(initial_state)
    torch.npu.synchronize()

    for step in range(steps):
        x = torch.randn((batch * draft_tokens, dim), device="npu", dtype=dtype)
        staging_x.copy_(x)
        expected = causal_conv1d_update_v2(
            x=x.view(1, draft_tokens, dim).contiguous(),
            conv_state=reference_state,
            weight=weight,
            bias=bias,
            activation="silu",
            conv_state_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            pad_slot_id=-1,
            validate_data=False,
        ).view(draft_tokens, dim)

        # Exercise allocator reuse around the address of the temporary tiling
        # Tensor that existed only while the graph was captured.
        allocator_pressure = [
            torch.empty(4096, device="npu", dtype=torch.uint8) for _ in range(32)
        ]
        del allocator_pressure
        graph.replay()
        torch.npu.synchronize()

        torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(native_state, reference_state, rtol=0, atol=0)

        accepted_step = step % draft_tokens
        _rollback_speculative_state(
            native_state, 3, accepted_step, draft_tokens
        )
        _rollback_speculative_state(
            reference_state, 3, accepted_step, draft_tokens
        )

    _assert_exact(f"graph-output-after-{steps}", graph_output, expected)
    _assert_exact(f"graph-state-after-{steps}", native_state, reference_state)


def _rollback_speculative_state(state, cache_index, accepted_step, draft_tokens):
    """Mirror conv_state_rollback after target verification."""
    shift = (draft_tokens - 1) - accepted_step
    if shift > 0:
        state[cache_index, shift:] = state[cache_index, :-shift].clone()


def run_speculative_against_triton(dim, steps, use_bias):
    """Compare the AscendC update directly with the known-good PR651 path.

    Production target verification always passes draft_tokens as the accepted
    count to causal_conv1d, then rolls the six-slot conv window back to the
    actually accepted step.  Cycling that rollback step exposes state errors
    which a single fixed-accept invocation cannot see.
    """
    torch.manual_seed(20260827)
    dtype = torch.bfloat16
    width = 4
    draft_tokens = 4
    state_len = width - 1 + draft_tokens - 1
    cache_index = 3

    weight = torch.randn((width, dim), device="npu", dtype=dtype)
    bias = torch.randn((dim,), device="npu", dtype=dtype) if use_bias else None
    initial_state = torch.randn(
        (8, state_len, dim), device="npu", dtype=dtype
    )
    native_state = initial_state.clone()
    triton_state = initial_state.clone()
    query_start_loc = _device_int64([0, draft_tokens])
    cache_indices = _device_int64([cache_index])
    num_accepted_tokens = torch.full(
        (1,), draft_tokens, device="npu", dtype=torch.int32
    )

    for step in range(steps):
        x = torch.randn((draft_tokens, dim), device="npu", dtype=dtype)
        native_output = torch.ops.npu.causal_conv1d(
            x,
            weight,
            native_state,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        triton_output = causal_conv1d_update_v2(
            x=x.view(1, draft_tokens, dim).contiguous(),
            conv_state=triton_state,
            weight=weight,
            bias=bias,
            activation="silu",
            conv_state_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            pad_slot_id=-1,
            validate_data=False,
        ).view(draft_tokens, dim)

        torch.npu.synchronize()
        try:
            torch.testing.assert_close(
                native_output, triton_output, rtol=0, atol=0
            )
            torch.testing.assert_close(native_state, triton_state, rtol=0, atol=0)
        except AssertionError:
            _assert_exact(f"triton-output-first-mismatch-{step}", native_output, triton_output)
            _assert_exact(f"triton-state-first-mismatch-{step}", native_state, triton_state)
            raise

        accepted_step = step % draft_tokens
        _rollback_speculative_state(
            native_state, cache_index, accepted_step, draft_tokens
        )
        _rollback_speculative_state(
            triton_state, cache_index, accepted_step, draft_tokens
        )

    _assert_exact(f"triton-output-after-{steps}", native_output, triton_output)
    _assert_exact(f"triton-state-after-{steps}", native_state, triton_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[257, 129, 67]
    )
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--graph-steps", type=int, default=64)
    parser.add_argument("--triton-steps", type=int, default=256)
    parser.add_argument("--only-triton", action="store_true")
    parser.add_argument("--only-padded-tail", action="store_true")
    parser.add_argument("--padded-length", type=int, default=128)
    parser.add_argument("--valid-length", type=int, default=67)
    args = parser.parse_args()
    torch.npu.set_device(f"npu:{args.device}")
    use_bias = not args.no_bias
    if args.only_padded_tail:
        run_padded_prefill_tail(
            args.dim,
            args.padded_length,
            args.valid_length,
            use_bias=use_bias,
        )
        return
    if args.only_triton:
        run_speculative_against_triton(
            args.dim, args.triton_steps, use_bias=use_bias
        )
        return
    run_prefill_chain(args.dim, args.lengths, use_bias=use_bias)
    run_decode_chain(args.dim, args.decode_steps, use_bias=use_bias)
    run_speculative_update(args.dim, use_bias=use_bias)
    run_speculative_graph_replay(args.dim, args.graph_steps, use_bias=use_bias)
    run_speculative_against_triton(
        args.dim, args.triton_steps, use_bias=use_bias
    )


if __name__ == "__main__":
    main()
