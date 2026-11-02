"""CPU execution of the actual Triton kernel body through a NumPy TL facade.

This checks vector address arithmetic and masks, not Triton compilation, cache
visibility, or the A5 runtime. Real NPU coverage is in tests/python/operators/mem_cache/test_kv_transfer.py.
"""

import importlib.util
import inspect
from itertools import product
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]


class Value(np.ndarray if np is not None else object):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to(self, dtype):
        return self.astype(dtype).view(Value)


class Pointer:
    def __init__(self, array, offset=0):
        self.array, self.offset = array, np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset)


@dataclass(frozen=True)
class Device:
    type: str
    index: int = 0


NPU, CPU = Device("npu"), Device("cpu")


class Tensor:
    def __init__(self, shape, dtype, device, base=None, strides=None, values=None):
        self.array = (
            np.zeros(shape, dtype=dtype)
            if values is None
            else np.asarray(values, dtype=dtype)
        )
        self.shape, self.ndim, self.dtype = (
            self.array.shape,
            self.array.ndim,
            self.array.dtype,
        )
        self.device = device
        self.base = self.array.ctypes.data if base is None else base
        self.strides = (
            tuple(s // self.array.itemsize for s in self.array.strides)
            if strides is None
            else strides
        )

    def numel(self):
        return self.array.size

    def stride(self, axis):
        return self.strides[axis]

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def element_size(self):
        return self.array.itemsize

    def data_ptr(self):
        return self.base


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Language:
    constexpr = object
    pids = (0, 0)

    @staticmethod
    def program_id(axis):
        return Language.pids[axis]

    static_range = staticmethod(range)

    @staticmethod
    def arange(start, stop):
        return Value(np.arange(start, stop))

    @staticmethod
    def load(ptr, mask=True, other=0):
        offsets, valid = np.broadcast_arrays(ptr.offset, np.asarray(mask))
        result = np.full(offsets.shape, other, dtype=ptr.array.dtype)
        result[valid] = ptr.array[offsets[valid].astype(np.int64)]
        return result.view(Value)

    @staticmethod
    def store(ptr, values, mask=True):
        offsets, valid, values = np.broadcast_arrays(
            ptr.offset, np.asarray(mask), np.asarray(values)
        )
        ptr.array[offsets[valid].astype(np.int64)] = values[valid].astype(
            ptr.array.dtype
        )

    @staticmethod
    def where(condition, a, b):
        return Value(np.where(condition, a, b))


class Kernel:
    def __init__(self, fn, do_not_specialize=()):
        self.fn, self.signature = fn, inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.launches = []
        self.runtime_arguments = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append(grid)

            def runtime_value(value):
                if isinstance(value, Tensor):
                    return Pointer(value.array)
                if isinstance(value, tuple):
                    raise TypeError("legacy Triton cannot bind runtime tuple arguments")
                return Value(np.int64(value))

            bound = self.signature.bind(*args, **kwargs)
            self.runtime_arguments.append(
                {
                    name: value
                    for name, value in bound.arguments.items()
                    if self.signature.parameters[name].annotation
                    is inspect.Parameter.empty
                }
            )
            for name, value in bound.arguments.items():
                if (
                    self.signature.parameters[name].annotation
                    is inspect.Parameter.empty
                ):
                    bound.arguments[name] = runtime_value(value)
            for pids in product(*(range(size) for size in grid)):
                Language.pids = pids
                self.fn(**bound.arguments)

        return run


@unittest.skipIf(np is None, "optional NumPy required for the CPU TL facade")
class TestTritonKVMetadata(unittest.TestCase):
    def setUp(self):
        Language.int64, Language.uint64 = np.int64, np.uint64
        triton = SimpleNamespace(
            jit=lambda **options: lambda fn: Kernel(fn, **options),
            language=Language,
            cdiv=lambda x, y: (x + y - 1) // y,
        )
        torch = SimpleNamespace(
            int64=np.dtype(np.int64),
            int32=np.dtype(np.int32),
            iinfo=np.iinfo,
            empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
            empty_like=lambda t: Tensor(t.shape, t.dtype, t.device),
            npu=SimpleNamespace(device=lambda _: Context()),
        )
        with patch.dict(
            sys.modules, {"torch": torch, "triton": triton, "triton.language": Language}
        ):
            path = (
                ROOT
                / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/memfabric/mem_cache/kv_transfer.py"
            )
            spec = importlib.util.spec_from_file_location("_cpu_kv_transfer", path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def indices(self, pages, page_size):
        values = [
            page * page_size + token for page in pages for token in range(page_size)
        ]
        return Tensor((len(values),), np.int64, NPU, values=values)

    def components(
        self, layers=(3, 3, 2), widths=(5, 2, 1), dtypes=None, pages=10, ps=4
    ):
        dtypes = (np.uint8, np.uint16, np.float32) if dtypes is None else dtypes
        devices, hosts = [], []
        for i, (nlayer, width, dtype) in enumerate(zip(layers, widths, dtypes)):
            item = np.dtype(dtype).itemsize
            device = Tensor(
                (nlayer, pages, ps, 1, width),
                dtype,
                NPU,
                base=(1 << 44) + i * (1 << 36),
            )
            # A layer slice: the page stride retains two extra physical layers.
            host = Tensor(
                (pages + 2, nlayer, ps, 1, width),
                dtype,
                CPU,
                base=(1 << 45) + i * (1 << 36) + ps * width * item,
                strides=((nlayer + 2) * ps * width, ps * width, width, width, 1),
            )
            devices.append(device)
            hosts.append(host)
        return devices, hosts

    def expected(self, devices, hosts, dp, hp):
        rows = []
        for dev, host in zip(devices, hosts):
            for dpage, hpage in zip(dp, hp):
                for layer in range(dev.shape[0]):
                    da = (
                        dev.base
                        + (layer * dev.stride(0) + dpage * dev.stride(1))
                        * dev.element_size()
                    )
                    ha = (
                        host.base
                        + (hpage * host.stride(0) + layer * host.stride(1))
                        * host.element_size()
                    )
                    size = int(np.prod(dev.shape[2:])) * dev.element_size()
                    rows.append((da, ha, size))
        return rows

    def assert_metadata(self, devices, hosts, dp, hp, ps, direction):
        kernel = self.module._build_kv_sparse_metadata_kernel
        previous_launches = len(kernel.launches)
        outputs = self.module.build_kv_sparse_metadata(
            tuple(zip(devices, hosts)),
            self.indices(dp, ps),
            self.indices(hp, ps),
            ps,
            direction,
        )
        self.assertEqual(
            kernel.launches[previous_launches:],
            [((len(dp) * dev.shape[0] + 127) // 128,) for dev in devices],
        )
        reference = self.expected(devices, hosts, dp, hp)
        src, dst, lengths, count = (t.array for t in outputs)
        self.assertEqual(
            src.tolist(), [row[1 if direction == 1 else 0] for row in reference]
        )
        self.assertEqual(
            dst.tolist(), [row[0 if direction == 1 else 1] for row in reference]
        )
        self.assertEqual(lengths.tolist(), [row[2] for row in reference])
        total = len(reference)
        self.assertEqual(count.tolist(), [total])

    def test_mixed_components_and_strided_host(self):
        devices, hosts = self.components()
        for direction in (1, 2):
            self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, direction)

    def test_reallocated_buffers_use_unspecialized_runtime_scalars(self):
        devices, hosts = self.components()
        kernel = self.module._build_kv_sparse_metadata_kernel
        self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, 1)
        first = kernel.runtime_arguments[-1]
        for dev, host in zip(devices, hosts):
            dev.base += 1 << 35
            host.base += 1 << 36
            dev.shape = (dev.shape[0], 11, *dev.shape[2:])
            host.shape = (13, *host.shape[1:])
        self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, 1)
        second = kernel.runtime_arguments[-1]
        changed = {
            name
            for name in first
            if isinstance(first[name], int) and first[name] != second[name]
        }
        self.assertTrue(changed)
        self.assertTrue(changed.issubset(kernel.do_not_specialize))
        self.assertIn("entries", kernel.do_not_specialize)
        self.assertIn("total_entries", kernel.do_not_specialize)

    def test_four_components_with_independent_layer_counts_and_tails(self):
        devices, hosts = self.components(
            layers=(70, 70, 21, 21),
            widths=(5, 2, 3, 1),
            dtypes=(np.uint16, np.uint16, np.uint8, np.float32),
        )
        for direction in (1, 2):
            self.assert_metadata(devices, hosts, [6, 2, 4], [1, 8, 3], 4, direction)

    def test_per_layer_metadata_with_four_components(self):
        devices, hosts = self.components(
            layers=(1, 1, 1, 1),
            widths=(5, 2, 3, 1),
            dtypes=(np.uint16, np.uint16, np.uint8, np.float32),
        )
        for direction in (1, 2):
            self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, direction)

    def test_layer_first_host_strides(self):
        devices, hosts = self.components()
        for host in hosts:
            _, _, ps, _, width = host.shape
            host.strides = (ps * width, 14 * ps * width, width, width, 1)
        for direction in (1, 2):
            self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, direction)

    def test_aicpu_record_limit_65536(self):
        devices, hosts = self.components(
            layers=(1,), widths=(1,), dtypes=(np.uint8,), pages=65538, ps=1
        )
        self.assert_metadata(
            devices, hosts, list(range(65536)), list(range(65536)), 1, 1
        )
        with self.assertRaisesRegex(ValueError, "single-batch.*got 65537"):
            self.module.build_kv_sparse_metadata(
                tuple(zip(devices, hosts)),
                self.indices(list(range(65537)), 1),
                self.indices(list(range(65537)), 1),
                1,
                1,
            )

    def test_address_offsets_above_32_bits(self):
        devices, hosts = self.components(layers=(3,), widths=(1,), dtypes=(np.uint8,))
        devices[0].strides = ((1 << 33) + 80, 4, 1, 1, 1)
        hosts[0].strides = ((1 << 34) + 20, 4, 1, 1, 1)
        self.assert_metadata(devices, hosts, [6, 2], [1, 8], 4, 1)

    def test_invalid_page_starts_emit_rejected_lengths_and_safe_bases(self):
        devices, hosts = self.components(layers=(3,), widths=(1,), dtypes=(np.uint8,))
        for dpage in (-1, 10):
            src, dst, lengths, _ = self.module.build_kv_sparse_metadata(
                tuple(zip(devices, hosts)),
                self.indices([dpage], 4),
                self.indices([1], 4),
                4,
                2,
            )
            self.assertEqual(src.array.tolist(), [devices[0].base] * 3)
            self.assertEqual(dst.array.tolist(), [hosts[0].base] * 3)
            self.assertEqual(lengths.array.tolist(), [-1] * 3)
        di = self.indices([0], 4)
        di.array[0] = 1
        _, _, lengths, _ = self.module.build_kv_sparse_metadata(
            tuple(zip(devices, hosts)), di, self.indices([1], 4), 4, 2
        )
        self.assertEqual(lengths.array.tolist(), [-1] * 3)

    def test_host_pool_pointer_preserves_layer_and_page_offsets(self):
        devices, hosts = self.components(layers=(3,), widths=(1,), dtypes=(np.uint8,))
        expected = [row[1] for row in self.expected(devices, hosts, [6, 2], [1, 8])]
        for direction in (1, 2):
            src, dst, _, _ = self.module.build_kv_sparse_metadata(
                tuple(zip(devices, hosts)),
                self.indices([6, 2], 4),
                self.indices([1, 8], 4),
                4,
                direction,
            )
            host_table = src if direction == 1 else dst
            self.assertEqual(host_table.array.tolist(), expected)

    def test_validation_rejects_bad_layout_and_length_overflow(self):
        devices, hosts = self.components(layers=(3,), widths=(1,), dtypes=(np.uint8,))
        pairs = tuple(zip(devices, hosts))
        devices[0].strides = (80, 4, 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.module._check_component(*pairs[0], 4, NPU)
        # Simulate a large page using metadata only; do not allocate its payload.
        for tensor in (devices[0], hosts[0]):
            tensor.shape = (*tensor.shape[:2], 4, 1, 1 << 30)
            tensor.strides = (1 << 36, 1 << 34, 1 << 30, 1 << 30, 1)
        with self.assertRaisesRegex(ValueError, "positive int32"):
            self.module._check_component(*pairs[0], 4, NPU)


if __name__ == "__main__":
    unittest.main()
