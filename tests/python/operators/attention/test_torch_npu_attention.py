"""Device accuracy cases for TorchNPU attention through the public API."""

import pytest

torch = pytest.importorskip("torch")


def _generate_fia_data(heads, q_len, kv_len):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return tuple(
        torch.randn((1, heads, seq, 128), generator=generator, device="cpu").to(
            torch.float16
        )
        for seq in (q_len, kv_len, kv_len)
    )


def _fia_reference(query, key, value, scale):
    logits = (query.float() @ key.float().transpose(-2, -1)) * scale
    attention = (logits.softmax(dim=-1) @ value.float()).to(query.dtype)
    return attention, logits.logsumexp(dim=-1, keepdim=True)


def _assert_fia_result(result, expected):
    torch.testing.assert_close(result[0].cpu(), expected[0], atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(result[1].cpu(), expected[1], atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("version", [1, 2], ids=["fia", "fia-v2"])
@pytest.mark.parametrize(
    "heads,q_len,kv_len,replay",
    [
        # Batch/head layout from OpPlugin's BNSD decode example, with shorter KV.
        pytest.param(32, 1, 512, False, id="bnsd-decode"),
        pytest.param(4, 17, 128, False, id="bnsd-prefill-query-tail"),
        pytest.param(
            32,
            1,
            512,
            True,
            id="bnsd-decode-graph-replay",
            marks=pytest.mark.extra_checks,
        ),
    ],
)
def test_fused_infer_attention_score_and_out(version, heads, q_len, kv_len, replay):
    from sgl_kernel_npu.operators import attention

    query, key, value = _generate_fia_data(heads, q_len, kv_len)
    scale = query.size(-1) ** -0.5
    expected = _fia_reference(query, key, value, scale)
    args = tuple(t.to("npu") for t in (query, key, value))
    if version == 1:
        op = attention.fused_infer_attention_score
        op_out = attention.fused_infer_attention_score_out
        get_workspace = attention.fused_infer_attention_score_get_max_workspace
        kwargs = dict(num_heads=heads, scale=scale, softmax_lse_flag=True)
    else:
        op = attention.fused_infer_attention_score_v2
        op_out = attention.fused_infer_attention_score_v2_out
        get_workspace = attention.fused_infer_attention_score_v2_get_max_workspace
        kwargs = dict(
            num_query_heads=heads, softmax_scale=scale, return_softmax_lse=True
        )

    kwargs["input_layout"] = "BNSD"
    result = op(*args, **kwargs)
    _assert_fia_result(result, expected)

    workspace = get_workspace(*args, **kwargs)
    buffers = [torch.full_like(t, float("nan")) for t in result]

    def run_out():
        op_out(*args, out=buffers, workspace=workspace, **kwargs, impl="torch_npu")

    run_out()
    _assert_fia_result(buffers, expected)

    if replay:
        updated_query = -query
        updated_query_npu = updated_query.npu()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                run_out()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            run_out()
        torch.npu.synchronize()
        args[0].copy_(updated_query_npu)
        for buffer in buffers:
            buffer.fill_(float("nan"))
        torch.npu.synchronize()
        # Signal the FIA replay event while keeping the captured arguments.
        graph.update(cpu_update_input=[{}])
        graph.replay()
        torch.npu.synchronize()
        _assert_fia_result(buffers, _fia_reference(updated_query, key, value, scale))


def _generate_attention_update_data(rows, head_dim):
    generator = torch.Generator(device="cpu").manual_seed(42)
    lse = torch.randn(2, rows, generator=generator, device="cpu")
    partial = torch.randn(2, rows, head_dim, generator=generator, device="cpu")
    return lse, partial


@pytest.mark.parametrize(
    "rows,head_dim",
    [
        # DSA DCP merges flattened token/head rows of 512-value MLA latents.
        pytest.param(64, 512, id="dsa-dcp-latent-width"),
        pytest.param(4, 32, id="small-partition-merge"),
    ],
)
@pytest.mark.parametrize("update_type", [0, 1], ids=["out-only", "out-and-lse"])
def test_attention_update(rows, head_dim, update_type):
    from sgl_kernel_npu.operators.attention import attention_update

    lse, partial = _generate_attention_update_data(rows, head_dim)
    expected = (lse.softmax(dim=0).unsqueeze(-1) * partial).sum(dim=0)
    result, merged_lse = attention_update(
        tuple(lse.to("npu").unbind()),
        tuple(partial.to("npu").unbind()),
        update_type,
        impl="torch_npu",
    )
    torch.testing.assert_close(result.cpu(), expected, atol=1e-4, rtol=1e-4)
    if update_type == 1:
        torch.testing.assert_close(
            merged_lse.cpu(), lse.logsumexp(dim=0), atol=1e-4, rtol=1e-4
        )
