"""A5-only validation for the AscendC SiTU + MXFP8 quant operator."""

import time

import torch
import torch_npu

from sgl_kernel_npu.activation.situ import situ
from sgl_kernel_npu.activation.situ_mxfp8_quant import situ_mxfp8_quant


def _situ_reference(x):
    gate, up = x.float().chunk(2, dim=-1)
    value = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    value = value * (25.0 * torch.tanh(up / 25.0))
    return value.to(torch.bfloat16)


def _dequant(payload, scales):
    scale = scales.contiguous().view(torch.uint8).reshape(scales.shape[0], -1).float()
    scale = torch.pow(2.0, scale - 127.0)
    return payload.float() * scale.repeat_interleave(32, dim=-1)


def _valid_rows(group_list, group_list_type):
    if group_list_type == 0:
        return int(group_list[-1].item())
    return int(group_list.sum().item())


def _run_case(group_list_type, dtype):
    torch.manual_seed(7)
    capacity = 128
    counts = torch.tensor([0, 7, 13, 0, 9, 3, 0, 5], dtype=dtype)
    group_list = counts.cumsum(0) if group_list_type == 0 else counts
    valid = _valid_rows(group_list, group_list_type)
    x = (torch.randn(capacity, 6144, dtype=torch.bfloat16) * 2).npu()
    group_list = group_list.npu()

    payload, scales = situ_mxfp8_quant(x, group_list, group_list_type)
    reference_bf16 = _situ_reference(x[:valid])
    ref_payload, ref_scales = torch_npu.npu_dynamic_mx_quant(
        reference_bf16, dst_type=torch.float8_e4m3fn
    )
    torch.npu.synchronize()

    assert payload.shape == (capacity, 3072)
    assert payload.dtype == torch.float8_e4m3fn
    assert scales.shape == (capacity, 48, 2)
    assert scales.dtype == torch.float8_e8m0fnu

    actual = _dequant(payload[:valid], scales[:valid])
    expected = _dequant(ref_payload, ref_scales)
    # E4M3FN reserves its all-ones magnitude encoding for NaN. The vendor
    # quantizer may emit that encoding at the rounding boundary, so compare
    # the NaN locations first and then treat matching NaNs as equal.
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08, equal_nan=True)


def test_count_int64():
    _run_case(1, torch.int64)


def test_count_int32():
    _run_case(1, torch.int32)


def test_cumulative_int64():
    _run_case(0, torch.int64)


def benchmark():
    torch.manual_seed(11)
    capacity = 32768
    counts = torch.tensor([4, 8, 3, 0, 6, 2, 5, 4], dtype=torch.int64).npu()
    x = torch.randn(capacity, 6144, dtype=torch.bfloat16).npu()

    def baseline():
        value, _ = situ(x, counts, 1, need_quant=False)
        return torch_npu.npu_dynamic_mx_quant(value, dst_type=torch.float8_e4m3fn)

    def fused():
        return situ_mxfp8_quant(x, counts, 1)

    for fn in (baseline, fused):
        for _ in range(5):
            fn()
        torch.npu.synchronize()

    for name, fn in (("baseline", baseline), ("fused", fused)):
        samples = []
        for _ in range(50):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000.0)
        samples.sort()
        print(f"{name}: p50={samples[len(samples)//2]:.3f} us avg={sum(samples)/len(samples):.3f} us")


if __name__ == "__main__":
    test_count_int64()
    test_count_int32()
    test_cumulative_int64()
    print("precision: PASS")
    benchmark()
