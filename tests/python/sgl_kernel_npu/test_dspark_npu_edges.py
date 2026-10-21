"""NPU-specific cases in addition to the unchanged CUDA parity inventory."""

import unittest
from types import SimpleNamespace

import torch
from dspark_reference.metadata import BuildDsparkSwaRingIndices as ReferenceRing
from test_dspark_kernel_parity import HAS_NPU

if HAS_NPU:
    from sgl_kernel_npu.attention.dspark_commit_swa import committed_swa_locations
    from sgl_kernel_npu.attention.dspark_kv_write import fused_kv_norm_rope_write
    from sgl_kernel_npu.attention.dspark_ring import BuildDsparkSwaRingIndices
    from sgl_kernel_npu.sample.dspark_accept import AcceptGreedy, SoftmaxTemp
    from sgl_kernel_npu.sample.dspark_draft import MarkovGreedyStep
    from sgl_kernel_npu.sample.dspark_greedy import row_argmax
    from sgl_kernel_npu.sample.dspark_kv_proj import (
        CommitKvProj,
        _dequant_linear_weight,
    )
    from sgl_kernel_npu.sample.dspark_verify_window import BuildOutTokens


@unittest.skipUnless(HAS_NPU, "requires Ascend NPU and torch_npu")
class TestDsparkNpuEdges(unittest.TestCase):
    def test_fp8_cast_and_block_scale_boundaries(self):
        # Exercise every FP8 bit pattern, including NaNs and both signs, and
        # cross both output/input 128-element scale boundaries.
        bits = (torch.arange(129 * 257) % 256).to(torch.uint8)
        weight = bits.view(torch.float8_e4m3fn).view(129, 257)
        scale = torch.tensor([[0.25, 0.5, 0.75], [1.25, 1.5, 1.75]])
        got = _dequant_linear_weight(
            SimpleNamespace(weight=weight.to("npu"), weight_scale_inv=scale.to("npu"))
        )
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:129, :257]
        ref = (weight.float() * expanded).bfloat16()
        torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0, equal_nan=True)

    def test_commit_projection_dtypes_and_partial_tiles(self):
        torch.manual_seed(60)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = (torch.randn(17, 65, device="npu") * 0.1).to(dtype)
            weights = [
                (torch.randn(33, 65, device="npu") * 0.1).to(dtype) for _ in range(2)
            ]
            got = CommitKvProj.execute(
                main_x=x, wkv_linears=[SimpleNamespace(weight=w) for w in weights]
            )
            for actual, weight in zip(got, weights):
                ref = torch.nn.functional.linear(
                    x.cpu().float(), weight.cpu().float()
                ).to(dtype)
                torch.testing.assert_close(actual.cpu(), ref, rtol=2e-2, atol=2e-3)

    def test_argmax_nan_inf_ties_and_strides(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.full((7, 2051), -float("inf"), dtype=dtype, device="npu")
            x[1, 1024:1026] = 1
            x[2, 8] = float("nan")
            x[2, 2049] = float("nan")
            x[3, 2049:] = float("inf")
            x[4] = 0
            x[5, 2001] = 2
            x[6, 0] = float("nan")
            for view in (x, x[:, ::2], x[::2]):
                torch.testing.assert_close(
                    row_argmax(view), view.argmax(-1), rtol=0, atol=0
                )

    def test_greedy_all_prefix_lengths(self):
        for bs in (1, 7, 65):
            width, vocab = 6, 19
            target = torch.arange(bs * width, device="npu").view(bs, width) % vocab
            candidates = torch.zeros_like(target)
            candidates[:, 1:] = target[:, :-1]
            lengths = torch.arange(bs, device="npu") % width
            for i in range(bs):
                if int(lengths[i]) < width - 1:
                    j = int(lengths[i]) + 1
                    candidates[i, j] = (candidates[i, j] + 1) % vocab
            logits = torch.full((bs * width, vocab), -10.0, device="npu")
            logits.scatter_(1, target.reshape(-1, 1), 10.0)
            got, bonus, trim = AcceptGreedy.execute(
                candidates=candidates,
                target_logits=logits,
                verify_num_draft_tokens=width,
            )
            torch.testing.assert_close(got, lengths.to(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(
                bonus, target[torch.arange(bs, device="npu"), lengths], rtol=0, atol=0
            )
            self.assertTrue(bool((trim == 0).all()))

    def test_ring_window_floor_and_padding(self):
        for block in (1, 5, 7):
            prefix = torch.tensor(
                [0, 3, 127, 128, 258], device="npu", dtype=torch.int32
            )
            seq = (prefix[:, None] + torch.arange(1, block + 1, device="npu")).reshape(
                -1
            )
            req = torch.arange(5, device="npu").repeat_interleave(block)
            kwargs = dict(
                seq_lens_casual=seq,
                req_pool_indices_repeated=req,
                valid_from=torch.tensor([0, 2, 7, 64, 255], device="npu"),
                ring_size=256,
                block_size=block,
                swa_window=128,
                page_index_aligned_size=96,
            )
            got = BuildDsparkSwaRingIndices.execute(**kwargs)
            ref = ReferenceRing.torch(**kwargs)
            for actual, expected in zip(got, ref):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_out_tokens_preserve_large_int64_ids(self):
        draft = torch.tensor(
            [[2**40 + 1, 2**40 + 3], [2**40 + 5, 2**40 + 7]], device="npu"
        )
        lengths = torch.tensor([0, 2], device="npu", dtype=torch.int32)
        bonus = torch.tensor([2**40 + 9, 2**40 + 11], device="npu")
        got = BuildOutTokens.execute(
            draft_tokens=draft,
            correct_len=lengths,
            bonus=bonus,
            verify_num_draft_tokens=3,
            gamma=2,
        )
        ref = torch.cat(
            (draft, torch.zeros((2, 1), dtype=torch.int64, device="npu")), 1
        )
        ref.scatter_(1, lengths.long()[:, None], bonus[:, None])
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_markov_greedy_non_power_of_two(self):
        torch.manual_seed(61)
        for rows, vocab, rank in ((1, 257, 33), (7, 2051, 65)):
            base = torch.randn(rows, vocab, device="npu")
            embeds = torch.randn(rows, rank, device="npu")
            weight = torch.randn(vocab, rank, device="npu")
            got = MarkovGreedyStep.execute(
                base_logits=base, prev_embeds=embeds, w2_weight=weight
            )
            ref = (base + embeds @ weight.T).argmax(-1)
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_committed_swa_masks_invalid_tail_locations(self):
        loc = torch.tensor([0, -1, 2**40, 3, 4, 5], device="npu")
        mapping = torch.arange(17, device="npu", dtype=torch.int32) * 3
        lengths = torch.tensor([2, 0], device="npu", dtype=torch.int32)
        got = committed_swa_locations(loc, mapping, lengths, 3)
        ref = torch.tensor([0, 48, -1, -1, -1, -1], device="npu", dtype=torch.int32)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_graph_replay_reads_new_values(self):
        x = torch.randn(6, 1025, device="npu")
        temps = torch.ones(2, device="npu")
        kwargs = dict(logits=x, temperatures=temps, rows_per_request=3)
        SoftmaxTemp.execute(**kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = SoftmaxTemp.execute(**kwargs)
        x.copy_(torch.randn_like(x))
        temps.copy_(torch.tensor([0.4, 1.7], device="npu"))
        graph.replay()
        torch.npu.synchronize()
        ref = torch.softmax(x / temps.repeat_interleave(3)[:, None], -1)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6)

    def test_kv_norm_rope_write_mask_and_styles(self):
        torch.manual_seed(72)
        for neox in (True, False):
            rows, layers, heads, dim, capacity = 11, 2, 2, 64, 20
            kvsize = heads * dim
            kv = torch.randn(
                rows, layers * 2 * kvsize, dtype=torch.bfloat16, device="npu"
            )
            weights = torch.randn(layers, dim, device="npu")
            angles = torch.randn(rows + 1, dim // 2, device="npu")
            cache = torch.cat((angles.cos(), angles.sin()), 1)
            positions = torch.arange(rows, device="npu")
            locs = torch.arange(rows, device="npu")
            locs[2] = -1
            lens = torch.tensor([rows - 2], device="npu", dtype=torch.int32)
            keys = [
                torch.full(
                    (capacity, heads, dim), -7, dtype=torch.bfloat16, device="npu"
                )
                for _ in range(layers)
            ]
            values = [torch.full_like(k, -9) for k in keys]
            metadata = torch.tensor(
                [
                    [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                    for k, v in zip(keys, values)
                ],
                dtype=torch.int64,
                device="npu",
            )
            expected_k = [k.cpu().clone() for k in keys]
            expected_v = [v.cpu().clone() for v in values]
            cpu = kv.cpu().float().view(rows, layers, 2, heads, dim)
            cos, sin = cache.cpu()[:, : dim // 2], cache.cpu()[:, dim // 2 :]
            for t in range(rows - 2):
                if t == 2:
                    continue
                for layer in range(layers):
                    k = cpu[t, layer, 0]
                    norm = (
                        (
                            k
                            * torch.rsqrt((k * k).mean(-1, keepdim=True) + 1e-6)
                            * weights[layer].cpu()
                        )
                        .bfloat16()
                        .float()
                    )
                    first = norm[:, : dim // 2] if neox else norm[:, 0::2]
                    second = norm[:, dim // 2 :] if neox else norm[:, 1::2]
                    a = first * cos[t] - second * sin[t]
                    b = second * cos[t] + first * sin[t]
                    rotated = (
                        torch.cat((a, b), -1)
                        if neox
                        else torch.stack((a, b), -1).flatten(-2)
                    )
                    expected_k[layer][t] = rotated.bfloat16()
                    expected_v[layer][t] = cpu[t, layer, 1].bfloat16()
            fused_kv_norm_rope_write(
                kv,
                metadata,
                weights,
                cache,
                positions,
                locs,
                layers,
                kvsize,
                dim,
                1e-6,
                commit_lens=lens,
                locs_row_width=rows,
                is_neox_style=neox,
            )
            torch.npu.synchronize()
            for i in range(layers):
                torch.testing.assert_close(
                    keys[i].cpu(), expected_k[i], rtol=1e-2, atol=1e-2
                )
                torch.testing.assert_close(
                    values[i].cpu(), expected_v[i], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
