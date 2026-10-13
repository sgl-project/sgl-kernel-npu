"""CPU execution of Triton function bodies; does not validate NPU compilation."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

SAMPLE = (
    Path(__file__).resolve().parents[3] / "python/sgl_kernel_npu/sgl_kernel_npu/sample"
)


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Pointer:
    def __init__(self, array, offset=0):
        self.array = array.reshape(-1)
        self.offset = np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset)


class Language:
    float32, int64 = np.float32, np.int64

    def __init__(self):
        self.program = 0
        self.programs = 3

    def program_id(self, axis):
        return tensor(self.program, np.int32)

    def num_programs(self, axis):
        return self.programs

    def range(self, start, end, step):
        for i in range(int(start), int(end), int(step)):
            yield tensor(i, np.int32)

    def arange(self, start, end):
        return tensor(np.arange(start, end, dtype=np.int32))

    def load(self, pointer, mask=True, other=0):
        offsets, active = np.broadcast_arrays(pointer.offset, mask)
        assert ((offsets[active] >= 0) & (offsets[active] < pointer.array.size)).all()
        return tensor(
            np.where(active, pointer.array[np.where(active, offsets, 0)], other),
            pointer.array.dtype,
        )

    def store(self, pointer, value, mask=True):
        offsets, values, active = np.broadcast_arrays(pointer.offset, value, mask)
        indices = offsets[active]
        assert ((indices >= 0) & (indices < pointer.array.size)).all()
        assert np.unique(indices).size == indices.size
        pointer.array[indices] = values[active]

    def where(self, condition, left, right):
        return tensor(np.where(condition, left, right))

    def div_rn(self, left, right):
        return tensor(np.divide(left, right, dtype=np.float32))


def _kernels(tl):
    path = SAMPLE / "probability_triton.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_kernel"):
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)
    scope = {"tl": tl}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def _check(probs, top_p, scalar):
    rows, vocab = probs.shape
    values, indices = probs.sort(-1, descending=True)
    cumulative = values.cumsum(-1)
    ps = torch.tensor([0.0, 0.25, 0.5, 0.95, 1.0])
    ks = torch.tensor([1, 2, 3, max(1, vocab - 1), vocab]).clamp(1, vocab)
    thresholds = ps if top_p else ks
    if scalar:
        thresholds = thresholds[3:4]
    expected_values = values.clone()
    if top_p:
        expected_values.masked_fill_(cumulative - values > thresholds[:, None], 0)
    else:
        expected_values.masked_fill_(torch.arange(vocab) >= thresholds[:, None], 0)
    expected_keys = indices.float() * 2 + (expected_values != 0).float()
    expected_values /= expected_values.sum(-1, keepdim=True).clamp_min_(1e-20)
    expected = torch.zeros_like(probs).scatter_(-1, indices, expected_values)
    work = values.numpy().copy()
    keys = np.full_like(work, np.nan)
    tl = Language()
    kernels = _kernels(tl)
    blocks = (vocab + 2047) // 2048
    for tl.program in range(tl.programs):
        kernels["_filter_encode_kernel"](
            Pointer(work),
            Pointer(indices.numpy()),
            Pointer(cumulative.numpy()),
            Pointer(thresholds.numpy()),
            Pointer(keys),
            rows,
            vocab,
            blocks,
            0 if scalar else 1,
            top_p,
            2048,
        )
    np.testing.assert_array_equal(keys, expected_keys.numpy())
    denominator = torch.from_numpy(work).sum(-1, keepdim=True).clamp_min_(1e-20).numpy()
    ordered = np.sort(keys, axis=-1)
    output = np.full_like(work, np.nan)
    original = probs.numpy().copy()
    for tl.program in range(tl.programs):
        kernels["_restore_kernel"](
            Pointer(original),
            Pointer(ordered),
            Pointer(denominator),
            Pointer(output),
            rows,
            vocab,
            blocks,
            2048,
        )
    np.testing.assert_array_equal(output, expected.numpy())
    np.testing.assert_array_equal(original, probs.numpy())


@pytest.mark.parametrize("vocab", [1, 4, 97, 2051])
@pytest.mark.parametrize("top_p", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("case", ["random", "edges", "tiny"])
def test_kernel_bodies(vocab, top_p, scalar, case):
    gen = torch.Generator().manual_seed(23)
    probs = torch.softmax(torch.randn(5, vocab, generator=gen), -1)
    if case == "edges":
        probs[0].fill_(1.0 / vocab)
        probs[1].zero_()
        probs[2].zero_()
        probs[2, -1] = 1
    elif case == "tiny":
        probs *= 1e-25
    _check(probs, top_p, scalar)


@pytest.mark.parametrize("vocab", [154880, 248320])
@pytest.mark.parametrize("top_p", [False, True])
def test_production_vocabulary(vocab, top_p):
    gen = torch.Generator().manual_seed(24)
    probs = torch.softmax(torch.randn(5, vocab, generator=gen), -1)
    _check(probs, top_p, False)


def test_key_precision_boundary():
    indices = np.array([0, 1, 2**23 - 2, 2**23 - 1], np.int64)
    for keep in (0, 1):
        keys = indices.astype(np.float32) * np.float32(2) + np.float32(keep)
        np.testing.assert_array_equal(keys.astype(np.int64), indices * 2 + keep)
