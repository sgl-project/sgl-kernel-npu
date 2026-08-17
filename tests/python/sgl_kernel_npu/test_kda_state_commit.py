import torch

from sgl_kernel_npu.mamba.kda_state_commit import (
    commit_kda_extended_conv_state,
    move_kda_temporal_snapshot,
    scatter_kda_conv_snapshot,
)


def test_temporal_production_layout_matches_under_graph_replay():
    device = torch.device("npu")
    layers, pool_size, steps, heads, dim_v, dim_k = 69, 3, 2, 6, 128, 128
    src = torch.randn(
        layers,
        1,
        steps,
        heads,
        dim_v,
        dim_k,
        dtype=torch.float32,
        device=device,
    )
    dst_storage = torch.full(
        (layers, pool_size, heads, dim_k, dim_v),
        -17.0,
        dtype=torch.float32,
        device=device,
    )
    dst = dst_storage.transpose(-1, -2)
    dst_indices = torch.tensor([2], dtype=torch.int32, device=device)
    src_indices = torch.tensor([0], dtype=torch.int32, device=device)
    step_indices = torch.tensor([0], dtype=torch.int32, device=device)

    assert move_kda_temporal_snapshot(
        dst, src, dst_indices, src_indices, step_indices
    )
    torch.npu.synchronize()
    torch.testing.assert_close(dst[:, 2], src[:, 0, 0], rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = move_kda_temporal_snapshot(
            dst, src, dst_indices, src_indices, step_indices
        )
    assert captured

    dst.fill_(-17.0)
    step_indices.fill_(1)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(dst[:, 2], src[:, 0, 1], rtol=0, atol=0)
    assert torch.all(dst[:, :2] == -17.0).item()

    dst.fill_(-17.0)
    step_indices.fill_(-1)
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(dst == -17.0).item()


def test_conv_production_layout_matches_under_graph_replay():
    device = torch.device("npu")
    layers, pool_size, steps, channels, window = 69, 3, 8, 2304, 3
    src = torch.randn(
        layers,
        1,
        steps,
        channels,
        window,
        dtype=torch.bfloat16,
        device=device,
    )
    dst = torch.full(
        (layers, pool_size, channels, window),
        -17.0,
        dtype=torch.bfloat16,
        device=device,
    )
    dst_indices = torch.tensor([2], dtype=torch.int32, device=device)
    src_indices = torch.tensor([0], dtype=torch.int32, device=device)
    step_indices = torch.tensor([1], dtype=torch.int32, device=device)

    assert scatter_kda_conv_snapshot(
        dst, src, dst_indices, src_indices, step_indices
    )
    torch.npu.synchronize()
    torch.testing.assert_close(dst[:, 2], src[:, 0, 1], rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = scatter_kda_conv_snapshot(
            dst, src, dst_indices, src_indices, step_indices
        )
    assert captured

    dst.fill_(-17.0)
    step_indices.fill_(5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(dst[:, 2], src[:, 0, 5], rtol=0, atol=0)
    assert torch.all(dst[:, :2] == -17.0).item()

    dst.fill_(-17.0)
    step_indices.fill_(-1)
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(dst == -17.0).item()


def test_extended_conv_state_view_matches_under_graph_replay():
    device = torch.device("npu")
    layers, pool_size, steps, channels, window = 69, 4, 8, 2304, 3
    extended = torch.randn(
        layers,
        pool_size,
        steps + window - 1,
        channels,
        dtype=torch.bfloat16,
        device=device,
    )
    original = extended.clone()
    destination = extended[:, :, -window:, :].transpose(-1, -2)
    primary_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    primary_steps = torch.tensor([2, 5], dtype=torch.int32, device=device)
    track_dst_indices = torch.tensor([2, 3], dtype=torch.int32, device=device)
    track_src_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    track_steps = torch.tensor([4, -1], dtype=torch.int32, device=device)

    # Match the framework order: tracking slots read the unmodified primary
    # windows, then primary slots are committed in place.
    assert commit_kda_extended_conv_state(
        extended, track_dst_indices, track_src_indices, track_steps, steps
    )
    assert commit_kda_extended_conv_state(
        extended, primary_indices, primary_indices, primary_steps, steps
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        destination[:, 0], original[:, 0, 2:5].transpose(-1, -2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        destination[:, 1], original[:, 1, 5:8].transpose(-1, -2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        destination[:, 2], original[:, 0, 4:7].transpose(-1, -2), rtol=0, atol=0
    )
    torch.testing.assert_close(extended[:, 3], original[:, 3], rtol=0, atol=0)
    ordinary = extended.clone()

    extended.copy_(original)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_track = commit_kda_extended_conv_state(
            extended, track_dst_indices, track_src_indices, track_steps, steps
        )
        captured_primary = commit_kda_extended_conv_state(
            extended, primary_indices, primary_indices, primary_steps, steps
        )
    assert captured_track and captured_primary

    extended.copy_(original)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(extended, ordinary, rtol=0, atol=0)

    # Runtime indices and steps must remain live under graph replay.
    extended.copy_(original)
    primary_steps.copy_(torch.tensor([7, 1], dtype=torch.int32, device=device))
    track_steps.copy_(torch.tensor([-1, 3], dtype=torch.int32, device=device))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        destination[:, 0], original[:, 0, 7:10].transpose(-1, -2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        destination[:, 1], original[:, 1, 1:4].transpose(-1, -2), rtol=0, atol=0
    )
    torch.testing.assert_close(extended[:, 2], original[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(
        destination[:, 3], original[:, 1, 3:6].transpose(-1, -2), rtol=0, atol=0
    )
