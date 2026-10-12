import gc
import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch.nn.functional as F
import torch_npu

torch_npu.npu.config.allow_internal_format = True

HC_MULT = 4
DSV4_FLASH_HIDDEN_SIZE = 4096
HIDDEN_SIZE = DSV4_FLASH_HIDDEN_SIZE
EXTENDED_HIDDEN_SIZE = 7168
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def _make_hc_pre_inputs(shape: tuple[int, ...]):
    torch.manual_seed(1024)
    hidden_size = shape[-1]
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = (
        torch.rand(
            MIX_HC,
            fan_in,
            dtype=torch.float32,
        )
        / fan_in
    )
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)

    # HcPre performs this matmul in HF32 round-toward-zero mode.
    mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split(
        [HC_MULT, HC_MULT, HC_MULT * HC_MULT],
        dim=-1,
    )
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)

    mix_for_y = pre_mix.float() if pre_mix is not None else pre
    y = (mix_for_y.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb_frag, pre


def _assert_close_with_pass_rate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_threshold: float,
    required_pass_rate: float,
):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny)
        <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert (
        pass_rate >= required_pass_rate
    ), f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; max absolute difference: {max_abs_diff}"


def _compare_hc_pre_with_cpu(shape: tuple[int, ...]):
    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    expected_y, expected_post, expected_comb_frag, _ = _hc_pre_cpu(
        x,
        hc_fn,
        hc_scale,
        hc_base,
    )
    y, post, comb_frag = torch.ops.npu.npu_hc_pre_v2(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        HC_MULT,
        HC_SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
    )

    batch_shape = shape[:-2]
    assert y.shape == (*batch_shape, shape[-1])
    assert post.shape == (*batch_shape, HC_MULT)
    assert comb_frag.shape == (*batch_shape, HC_MULT, HC_MULT)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    _assert_close_with_pass_rate(
        y,
        expected_y,
        diff_threshold=Y_DIFF_THRESHOLD,
        required_pass_rate=Y_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        post,
        expected_post,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        comb_frag,
        expected_comb_frag,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )


class TestHcPre(unittest.TestCase):
    @torch.inference_mode()
    def test_npu_hc_pre_v2_bf16_3d_input(self):
        _compare_hc_pre_with_cpu((2, HC_MULT, HIDDEN_SIZE))
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_npu_hc_pre_v2_bf16_4d_input(self):
        _compare_hc_pre_with_cpu((1, 2, HC_MULT, HIDDEN_SIZE))
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_npu_hc_pre_v2_bf16_dsv4_flash_hidden_size(self):
        _compare_hc_pre_with_cpu((4, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_npu_hc_pre_v2_bf16_extended_hidden_size(self):
        _compare_hc_pre_with_cpu((2, HC_MULT, EXTENDED_HIDDEN_SIZE))
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_npu_hc_pre_v3_uses_external_pre_mix(self):
        shape = (2, HC_MULT, HIDDEN_SIZE)
        x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
        pre_mix = torch.rand(shape[:-1], dtype=torch.float32)
        expected_y, expected_post, expected_comb_frag, expected_pre = _hc_pre_cpu(
            x, hc_fn, hc_scale, hc_base, pre_mix
        )
        y, post, comb_frag, pre = torch.ops.npu.npu_hc_pre_v3(
            x.npu(),
            hc_fn.npu(),
            hc_scale.npu(),
            hc_base.npu(),
            pre_mix.npu(),
            hc_mult=HC_MULT,
            hc_sinkhorn_iters=HC_SINKHORN_ITERS,
            norm_eps=NORM_EPS,
            hc_eps=HC_EPS,
        )
        assert y.shape == (shape[0], shape[-1])
        assert post.shape == (shape[0], HC_MULT)
        assert comb_frag.shape == (shape[0], HC_MULT, HC_MULT)
        assert pre.shape == (shape[0], HC_MULT)
        assert y.dtype == torch.bfloat16
        assert post.dtype == torch.float32
        assert comb_frag.dtype == torch.float32
        assert pre.dtype == torch.float32
        for actual, expected, diff_threshold, required_pass_rate in (
            (y, expected_y, Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
            (post, expected_post, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
            (comb_frag, expected_comb_frag, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
            (pre, expected_pre, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        ):
            _assert_close_with_pass_rate(
                actual,
                expected,
                diff_threshold=diff_threshold,
                required_pass_rate=required_pass_rate,
            )
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_npu_graph_replay(self):
        for shape in ((2, HC_MULT, HIDDEN_SIZE), (1, 2, HC_MULT, HIDDEN_SIZE)):
            for version, external_pre in ((2, False), (3, False), (3, True)):
                with self.subTest(
                    shape=shape, version=version, external_pre=external_pre
                ):
                    cpu_inputs = _make_hc_pre_inputs(shape)
                    inputs = [value.npu() for value in cpu_inputs]
                    pre_mix = (
                        torch.rand(shape[:-1], dtype=torch.float32)
                        if external_pre
                        else None
                    )
                    npu_pre_mix = pre_mix.npu() if pre_mix is not None else None

                    def call():
                        if version == 2:
                            return torch.ops.npu.npu_hc_pre_v2(
                                *inputs, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
                            )
                        return torch.ops.npu.npu_hc_pre_v3(*inputs, npu_pre_mix)

                    call()  # Initialize the persistent tiling cache before capture.
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    stream = torch.npu.Stream()
                    with torch.npu.graph(
                        graph, stream=stream, auto_dispatch_capture=True
                    ):
                        captured = call()
                    torch.npu.synchronize()
                    for factor in (0.5, 1.5):
                        changed_x = (cpu_inputs[0].float() * factor).to(torch.bfloat16)
                        inputs[0].copy_(changed_x)
                        changed_pre = pre_mix * factor if pre_mix is not None else None
                        if npu_pre_mix is not None:
                            npu_pre_mix.copy_(changed_pre)
                        graph.replay()
                        torch.npu.synchronize()
                        expected = _hc_pre_cpu(changed_x, *cpu_inputs[1:], changed_pre)
                        for index, actual in enumerate(captured):
                            _assert_close_with_pass_rate(
                                actual,
                                expected[index],
                                diff_threshold=(
                                    Y_DIFF_THRESHOLD
                                    if index == 0
                                    else AUX_DIFF_THRESHOLD
                                ),
                                required_pass_rate=(
                                    Y_REQUIRED_PASS_RATE
                                    if index == 0
                                    else AUX_REQUIRED_PASS_RATE
                                ),
                            )
                    gc.collect()
                    torch.npu.empty_cache()

    def test_meta_output_shapes_and_dtypes(self):
        for shape in ((2, HC_MULT, HIDDEN_SIZE), (1, 2, HC_MULT, HIDDEN_SIZE)):
            with self.subTest(shape=shape):
                inputs = [value.to("meta") for value in _make_hc_pre_inputs(shape)]
                batch = shape[:-2]
                expected_shapes = (
                    (*batch, shape[-1]),
                    (*batch, HC_MULT),
                    (*batch, HC_MULT, HC_MULT),
                    (*batch, HC_MULT),
                )
                outputs_v2 = torch.ops.npu.npu_hc_pre_v2(
                    *inputs, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
                )
                outputs_v3 = torch.ops.npu.npu_hc_pre_v3(*inputs)
                self.assertEqual(len(outputs_v2), 3)
                self.assertEqual(len(outputs_v3), 4)
                for outputs in (outputs_v2, outputs_v3):
                    for index, output in enumerate(outputs):
                        self.assertEqual(tuple(output.shape), expected_shapes[index])
                        self.assertEqual(output.device.type, "meta")
                        self.assertEqual(
                            output.dtype,
                            torch.bfloat16 if index == 0 else torch.float32,
                        )


if __name__ == "__main__":
    unittest.main()
