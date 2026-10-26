"""Public-interface coverage, preserving the existing 57 acceptance cases."""
import pytest

torch = pytest.importorskip("torch")
from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
CASES = [(shape, mode, "none")
         for shape in [(1, 64), (4, 320), (64, 32000), (2, 128256)]
         for mode in ["all_masked", "all_unmasked", "half_masked", "random"]]
CASES += [((8, 320), "random", "subset"), ((4, 320), "random", "all"),
          ((4, 320), "random", "empty")]
# Additional shapes from the unchanged LEGACY functional suite.
LEGACY_SHAPES = [(1, 320), (4, 320), (1, 64), (1, 32000), (8, 32000), (2, 128256),
                 (16, 32000), (2, 1), (2, 31), (2, 32), (2, 33), (4, 100),
                 (32, 32000), (8, 32100), (1, 128)]
STANDARD = {str(dtype): AccuracyStandard(atol=1e-2, rtol=1e-2) for dtype in DTYPES}


def validate_case(shape, dtype, mode, selection):
    from sgl_kernel_npu.operators.sampler import apply_token_bitmask
    from tests.python.utils.common import get_test_impls

    generator = torch.Generator(device="cpu").manual_seed(20261022)
    logits = torch.randn(shape, generator=generator, device="cpu").to(dtype).npu()
    width = (shape[1] + 31) // 32
    if mode == "random":
        bitmask = torch.randint(-(2**31), 2**31, (shape[0], width),
                                generator=generator, dtype=torch.int32, device="cpu").npu()
    else:
        value = {"all_masked": 0, "all_unmasked": -1, "half_masked": 0x55555555}[mode]
        bitmask = torch.full((shape[0], width), value, dtype=torch.int32, device=logits.device)
    choices = {"none": None, "subset": [0, 2, 5, 7], "all": list(range(shape[0])), "empty": []}
    indices = None if selection == "none" else torch.tensor(choices[selection], dtype=torch.int32, device=logits.device)

    def fresh():
        return (logits.clone(), bitmask.clone(), None if indices is None else indices.clone()), {}

    executor = check_accuracy(apply_token_bitmask, input_factory=fresh, standard=STANDARD)
    impl, _ = get_test_impls()
    (values, bits, rows), _ = fresh()
    result = apply_token_bitmask(values, bits, rows, impl=impl)
    assert result.data_ptr() == values.data_ptr()
    torch.testing.assert_close(result[~torch.isneginf(result)], logits[~torch.isneginf(result)], atol=0, rtol=0)
    assert torch.equal(bits, bitmask)
    if rows is not None:
        assert torch.equal(rows, indices)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,mode,selection", CASES)
def test_apply_token_bitmask(shape, dtype, mode, selection):
    validate_case(shape, dtype, mode, selection)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", LEGACY_SHAPES)
def test_legacy_shapes(shape, dtype):
    validate_case(shape, dtype, "random", "none")
