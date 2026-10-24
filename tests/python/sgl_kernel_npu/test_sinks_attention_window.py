import math
import unittest

import torch


def make_inputs(lengths, query_lengths, page_size=128, dtype=torch.bfloat16, seed=13):
    generator = torch.Generator().manual_seed(seed)
    q_heads, kv_heads, dim = 64, 8, 64
    pages = max(1, math.ceil(max(lengths) / page_size))
    num_blocks = len(lengths) * pages
    table = torch.randperm(num_blocks, generator=generator).reshape(len(lengths), pages)
    shape = (num_blocks, page_size, kv_heads, dim)
    return {
        "query": torch.randn(sum(query_lengths), q_heads * dim, generator=generator).to(
            dtype
        ),
        "k_cache": torch.randn(shape, generator=generator).to(dtype),
        "v_cache": torch.randn(shape, generator=generator).to(dtype),
        "sinks": torch.randn(q_heads, generator=generator).to(dtype),
        "block_tables": table.to(torch.int32),
        "context_lens": torch.tensor(lengths, dtype=torch.int32),
        "seq_lens": torch.tensor(query_lengths, dtype=torch.int32),
        "scale": dim**-0.5,
        "q_head_num": q_heads,
        "k_head_num": kv_heads,
    }


def reference(inputs, left_extent):
    """Dense CPU reference: causal key distance is at most left_extent."""
    query = inputs["query"].double()
    q_heads, kv_heads = inputs["q_head_num"], inputs["k_head_num"]
    dim = query.shape[-1] // q_heads
    outputs = []
    offset = 0
    for batch, (length, q_len) in enumerate(
        zip(inputs["context_lens"].tolist(), inputs["seq_lens"].tolist())
    ):
        q = query[offset : offset + q_len].reshape(q_len, q_heads, dim)
        offset += q_len
        pages = math.ceil(length / inputs["k_cache"].shape[1])
        ids = inputs["block_tables"][batch, :pages].long()
        k = inputs["k_cache"][ids].reshape(-1, kv_heads, dim)[:length].double()
        v = inputs["v_cache"][ids].reshape(-1, kv_heads, dim)[:length].double()
        k = k.repeat_interleave(q_heads // kv_heads, dim=1)
        v = v.repeat_interleave(q_heads // kv_heads, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q, k) * inputs["scale"]
        query_positions = torch.arange(length - q_len, length)
        key_positions = torch.arange(length)
        distance = query_positions[:, None] - key_positions[None, :]
        visible = distance >= 0
        if left_extent >= 0:
            visible &= distance <= left_extent
        logits.masked_fill_(~visible[None, :, :], -torch.inf)
        sink = inputs["sinks"].double()[:, None, None].expand(-1, q_len, 1)
        probabilities = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)
        output = torch.einsum("hqk,khd->qhd", probabilities[..., :-1], v)
        outputs.append(output.reshape(q_len, q_heads * dim))
    return torch.cat(outputs)


def assert_attention_output(test_case, output, expected, head_dim=64):
    """Check each token/head independently without an absolute small-value escape."""
    actual = output.detach().cpu().double()
    expected = expected.detach().cpu().double()
    test_case.assertEqual(actual.shape, expected.shape)
    test_case.assertTrue(bool(torch.isfinite(actual).all()))
    test_case.assertTrue(bool(torch.isfinite(expected).all()))
    actual_heads = actual.reshape(-1, head_dim)
    expected_heads = expected.reshape(-1, head_dim)
    zero_heads = torch.all(expected_heads == 0, dim=-1)
    test_case.assertEqual(torch.count_nonzero(actual_heads[zero_heads]).item(), 0)
    reference_norm = torch.linalg.vector_norm(expected_heads, dim=-1)
    error_norm = torch.linalg.vector_norm(actual_heads - expected_heads, dim=-1)
    # Half the BF16 subnormal quantum per component; this is not a 1e-2
    # absolute tolerance. The test's FP16 outputs must meet the same bound.
    tolerance = 0.01 * reference_norm + math.sqrt(head_dim) * 2**-134
    nonzero_heads = ~zero_heads
    max_relative_error = (
        (error_norm[nonzero_heads] / reference_norm[nonzero_heads]).max().item()
        if bool(nonzero_heads.any())
        else 0.0
    )
    test_case.assertTrue(
        bool(torch.all(error_norm <= tolerance)),
        f"per-token/head relative L2 exceeds 1%: maximum {max_relative_error:.6g}",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def boundary_inputs(query_length=1, dtype=torch.bfloat16):
    inputs = make_inputs([257], [query_length], dtype=dtype)
    inputs["query"].zero_()
    inputs["k_cache"].zero_()
    inputs["v_cache"].zero_()
    inputs["sinks"].fill_(-80)
    first_query_position = 257 - query_length
    # A 128-token window includes distance 127 and excludes distance 128.
    for position, value in (
        (first_query_position - 127, 64),
        (first_query_position - 128, 128),
    ):
        page, index = divmod(position, 128)
        block = int(inputs["block_tables"][0, page])
        inputs["v_cache"][block, index].fill_(value)
    return inputs


class TestSinksWindowReference(unittest.TestCase):
    def test_boundary_signal(self):
        for q_len in (1, 17):
            with self.subTest(query_length=q_len):
                expected = reference(boundary_inputs(q_len), 127)
                torch.testing.assert_close(
                    expected[0], torch.full_like(expected[0], 0.5)
                )
                self.assertEqual(torch.count_nonzero(expected[1:]).item(), 0)

    def test_current_token_and_full_window(self):
        inputs = make_inputs([3], [1])
        inputs["query"].zero_()
        inputs["k_cache"].zero_()
        inputs["v_cache"].zero_()
        inputs["sinks"].zero_()
        block = int(inputs["block_tables"][0, 0])
        inputs["v_cache"][block, :3] = torch.tensor([2, 4, 6])[:, None, None]
        # The virtual sink has a zero value and the same logit as every key.
        for left_extent, value in ((0, 3), (1, 10 / 3), (-1, 3)):
            with self.subTest(left_extent=left_extent):
                expected = reference(inputs, left_extent)
                torch.testing.assert_close(expected, torch.full_like(expected, value))

    def test_small_and_zero_outputs_cannot_use_absolute_tolerance(self):
        expected = torch.full((2, 128), 1e-6, dtype=torch.float64)
        actual = expected.clone()
        # One small head has a 2% error even though the absolute difference is
        # far below 1e-2 and other heads are correct.
        actual[1, 64:] *= 1.02
        with self.assertRaises(AssertionError):
            assert_attention_output(self, actual, expected)
        expected.zero_()
        actual.zero_()
        actual[0, 0] = 2**-134
        with self.assertRaises(AssertionError):
            assert_attention_output(self, actual, expected)
        actual.zero_()
        assert_attention_output(self, actual, expected)


class TestSinksAttentionWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_npu
        except ImportError:
            raise unittest.SkipTest("an Ascend NPU is required") from None
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        from sgl_kernel_npu.attention.sinks_attention import (
            attention_sinks_prefill_triton,
            attention_sinks_triton,
        )

        cls.decode = staticmethod(attention_sinks_triton)
        cls.prefill = staticmethod(attention_sinks_prefill_triton)
        cls.torch_npu = torch_npu
        cls.device = torch.device("npu", torch_npu.npu.current_device())

    def to_device(self, inputs):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def run_operator(self, inputs, left_extent, prefill=False):
        args = dict(inputs, sliding_window_size=left_extent)
        if prefill:
            return self.prefill(**args)
        args.pop("seq_lens")
        return self.decode(**args)

    def assert_output(self, output, expected):
        assert_attention_output(self, output, expected)

    def test_boundary_signal(self):
        for dtype in (torch.float16, torch.bfloat16):
            for q_len in (1, 17):
                with self.subTest(dtype=dtype, query_length=q_len):
                    inputs = boundary_inputs(q_len, dtype)
                    output = self.run_operator(self.to_device(inputs), 127, q_len > 1)
                    self.torch_npu.npu.synchronize()
                    self.assert_output(output, reference(inputs, 127))
                    torch.testing.assert_close(
                        output[0].float().cpu(),
                        torch.full_like(output[0].float().cpu(), 0.5),
                        rtol=0,
                        atol=0,
                    )

    def test_decode_windows_and_page_tails(self):
        for dtype in (torch.float16, torch.bfloat16):
            for page_size in (128, 256):
                inputs = make_inputs(
                    [0, 1, 127, 128, 129, 255, 256, 257, 513],
                    [1] * 9,
                    page_size,
                    dtype,
                )
                device_inputs = self.to_device(inputs)
                for left_extent in (-1, 0, 1, 127, 255):
                    with self.subTest(dtype=dtype, page=page_size, left=left_extent):
                        output = self.run_operator(device_inputs, left_extent)
                        self.torch_npu.npu.synchronize()
                        self.assert_output(output, reference(inputs, left_extent))

    def test_prefill_windows_and_prefixes(self):
        for dtype in (torch.float16, torch.bfloat16):
            inputs = make_inputs([127, 128, 129, 257], [7, 16, 17, 33], dtype=dtype)
            device_inputs = self.to_device(inputs)
            for left_extent in (-1, 0, 1, 127, 255):
                with self.subTest(dtype=dtype, left=left_extent):
                    output = self.run_operator(device_inputs, left_extent, prefill=True)
                    self.torch_npu.npu.synchronize()
                    self.assert_output(output, reference(inputs, left_extent))

    def test_decode_graph_reads_current_lengths_and_pages(self):
        inputs = make_inputs([257, 513], [1, 1])
        device_inputs = self.to_device(inputs)
        self.run_operator(device_inputs, 127)
        self.torch_npu.npu.synchronize()
        graph = self.torch_npu.npu.NPUGraph()
        with self.torch_npu.npu.graph(graph):
            output = self.run_operator(device_inputs, 127)
        for seed, lengths in ((21, [128, 513]), (22, [0, 511])):
            changed = make_inputs([257, 513], [1, 1], seed=seed)
            changed["context_lens"] = torch.tensor(lengths, dtype=torch.int32)
            for key, value in changed.items():
                if isinstance(value, torch.Tensor):
                    device_inputs[key].copy_(value)
            graph.replay()
            self.torch_npu.npu.synchronize()
            self.assert_output(output, reference(changed, 127))


if __name__ == "__main__":
    unittest.main()
