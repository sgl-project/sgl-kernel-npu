"""57-case NPU acceptance for the installed REFACTOR wheel."""

import importlib.metadata
from pathlib import Path


def main():
    import sgl_kernel_npu
    import torch
    import torch_npu

    print("PACKAGE:", sgl_kernel_npu.__file__)
    print("PACKAGE_VERSION:", importlib.metadata.version("sgl_kernel_npu"))
    print("TORCH:", torch.__version__, "TORCH_NPU:", torch_npu.__version__)
    if not (Path(sgl_kernel_npu.__file__).parent / "_cann").is_dir():
        raise RuntimeError("Wheel test imported a legacy or source-tree package")
    print(
        "CANN_VENDOR:",
        Path(sgl_kernel_npu.__file__).parent / "_cann/vendors/sgl_kernel_npu",
    )
    new = torch.ops.npu.apply_token_bitmask

    generator = torch.Generator().manual_seed(20261022)
    modes = ("all_masked", "all_unmasked", "half_masked", "random")
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    count = 0

    def check(actual, expected):
        actual = actual.cpu()
        assert torch.equal(
            torch.isneginf(actual), torch.isneginf(expected)
        ), "-inf positions differ"
        finite = ~torch.isneginf(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], atol=1e-2, rtol=1e-2
        )

    def case(shape, dtype, mode, indices=None, label="none"):
        nonlocal count
        batch, vocab = shape
        logits = torch.randn(shape, generator=generator).to(dtype)
        width = (vocab + 31) // 32
        if mode == "random":
            mask = torch.randint(
                -(2**31), 2**31, (batch, width), generator=generator, dtype=torch.int32
            )
        else:
            value = {"all_masked": 0, "all_unmasked": -1, "half_masked": 0x55555555}[
                mode
            ]
            mask = torch.full((batch, width), value, dtype=torch.int32)
        column = torch.arange(vocab)
        allowed = ((mask[:, column // 32].to(torch.int64) >> (column % 32)) & 1).bool()
        expected = logits.clone()
        # Preserve the legacy implementation: omitted AND empty indices mean all rows.
        rows = (
            torch.arange(batch)
            if indices is None or indices.numel() == 0
            else indices.long()
        )
        expected[rows] = expected[rows].masked_fill(~allowed[rows], float("-inf"))
        outputs = []
        for implementation in [new]:
            values = logits.npu()
            bits = mask.npu()
            selected = indices.npu() if indices is not None else None
            original = values.data_ptr()
            result = implementation(values, bits, selected)
            assert (
                result.data_ptr() == original
            ), "Operator did not return the original logits storage"
            check(result, expected)
            check(values, expected)
            assert torch.equal(bits.cpu(), mask), "bitmask was modified"
            if selected is not None:
                assert torch.equal(selected.cpu(), indices), "indices was modified"
            outputs.append(result.cpu())
        count += 1
        print(
            f"PASS: shape={shape} dtype={dtype} mask={mode} indices={label}", flush=True
        )

    for shape in ((1, 64), (4, 320), (64, 32000), (2, 128256)):
        for dtype in dtypes:
            for mode in modes:
                case(shape, dtype, mode)
    for dtype in dtypes:
        case(
            (8, 320),
            dtype,
            "random",
            torch.tensor([0, 2, 5, 7], dtype=torch.int32),
            "subset",
        )
        case((4, 320), dtype, "random", torch.arange(4, dtype=torch.int32), "all")
        case(
            (4, 320),
            dtype,
            "random",
            torch.empty(0, dtype=torch.int32),
            "empty-means-all",
        )
    torch.npu.synchronize()
    kind = "installed REFACTOR wheel"
    print(f"PASS: {count}/57 {kind} cases")


if __name__ == "__main__":
    main()
