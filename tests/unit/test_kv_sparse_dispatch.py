"""CPU checks of the real wrapper's submission, lifetime and stream ordering.

Only the NPU/Memfabric boundary is mocked; no PyTorch/CANN install is needed.
"""

import gc
import importlib.util
from functools import partial
import sys
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Device:
    type: str
    index: int = 0


NPU = Device("npu")
CPU = Device("cpu")


class Tensor:
    def __init__(self, shape, device=NPU, dtype="int64"):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.device = device
        self.dtype = dtype
        self.recorded = []
        self.pinned = None

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def contiguous(self):
        return self

    def pin_memory(self):
        self.pinned = Tensor(self.shape, CPU, self.dtype)
        return self.pinned

    def to(self, device, non_blocking=False):
        assert non_blocking
        return Tensor(self.shape, device, self.dtype)

    def record_stream(self, stream):
        self.recorded.append(stream)

    def unsqueeze(self, dim):
        return Tensor((*self.shape, 1), self.device, self.dtype)

    def element_size(self):
        return {"bf16": 2, "fp8": 1, "fp32": 4}.get(self.dtype, 8)

    def data_ptr(self):
        return id(self)

    def stride(self, axis):
        result = 1
        for dim in self.shape[axis + 1:]:
            result *= dim
        return result

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Tensor(shape, self.device, self.dtype)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        shape = list(self.shape)
        for axis, item in enumerate(key):
            start, stop, step = item.indices(shape[axis])
            shape[axis] = len(range(start, stop, step))
        return self if tuple(shape) == self.shape else Tensor(shape, self.device, self.dtype)


class Event:
    def __init__(self):
        self.done = False
        self.stream = None

    def query(self):
        return self.done

    def record(self, stream):
        self.stream = stream


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class TestSparseDispatch(unittest.TestCase):
    def setUp(self):
        self.stream = SimpleNamespace(wait_event=Mock())

        def build(components, di, hi, page_size, direction):
            total = di.numel() // page_size * sum(d.shape[0] for d, _ in components)
            self.tables = [
                Tensor((total,)),
                Tensor((total,)),
                Tensor((total,), dtype="int32"),
                Tensor((1,), dtype="int32"),
            ]
            return self.tables

        self.builder = Mock(side_effect=build)
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            int64="int64",
            int32="int32",
            iinfo=lambda _: SimpleNamespace(max=0x7fffffff),
            npu=SimpleNamespace(
                device=lambda _: Context(),
                current_stream=lambda: self.stream,
                Event=Event,
            ),
        )
        self.offload = SimpleNamespace(
            _get_scene=lambda: 2,
            Scene=SimpleNamespace(LOCAL=1, LOCAL_URMA=2),
            get_dva=Mock(return_value=1 << 44),
            CopyDirection=SimpleNamespace(H2D=1, D2H=2),
            sparse_copy=Mock(return_value=0),
        )
        language = SimpleNamespace(constexpr=object)
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "triton": SimpleNamespace(jit=lambda **_: lambda fn: fn, language=language),
                "triton.language": language,
                "memfabric_hybrid": SimpleNamespace(offload=self.offload),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        module_name = "sgl_kernel_npu.implementation.memfabric.mem_cache.kv_transfer"
        path = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/memfabric/mem_cache/kv_transfer.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.real_builder = self.module.build_kv_sparse_metadata
        self.module.build_kv_sparse_metadata = self.builder
        self.call = partial(self.module.transfer_kv, host_layout="page_first",
                            kv_layers=(0, 0, None), index_layers=(0, 0, None),
                            backend="aicpu", direction=1)

    def transfer(self, layers=3, pages=2, direction=1, cpu_indices=False):
        dev = Tensor((layers, pages + 3, 4, 1, 5), dtype="bf16")
        host = Tensor((pages + 5, layers, 4, 1, 5), CPU, "bf16")
        di = Tensor((pages * 4,), CPU if cpu_indices else NPU)
        hi = Tensor((pages * 4,), CPU if cpu_indices else NPU)
        self.call(
            di,
            hi,
            dev,
            host,
            Tensor((0,)),
            Tensor((0,), CPU),
            page_size=4,
            direction=direction,
        )
        return dev, host, di, hi

    def test_one_submission_uses_complete_metadata_and_retains_it(self):
        self.transfer()
        self.builder.assert_called_once()
        self.offload.sparse_copy.assert_called_once()
        args = self.offload.sparse_copy.call_args.args
        retained = self.module._inflight[0][1]
        for submitted, table in zip(args[:4], self.tables):
            self.assertIs(submitted, table)
            self.assertIn(table, retained)
        self.assertEqual(args[-1], 1)

    def test_keepalive_protects_inputs_without_record_stream(self):
        dev, host, di, hi = self.transfer()
        for tensor in (dev, di, hi):
            self.assertEqual(tensor.recorded, [])
        refs = [weakref.ref(tensor) for tensor in (dev, host, di, hi)]
        del tensor, dev, host, di, hi
        self.builder.reset_mock()
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        self.module._inflight[0][0].done = True
        self.transfer()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_aicpu_validates_once_and_launches_metadata_once(self):
        self.torch.empty = lambda shape, dtype, device: Tensor(shape, device, dtype)
        self.torch.empty_like = lambda tensor: Tensor(
            tensor.shape, tensor.device, tensor.dtype
        )
        kernel = MagicMock()
        sys.modules["triton"].cdiv = lambda size, block: (size + block - 1) // block
        with (
            patch.object(self.module, "build_kv_sparse_metadata", self.real_builder),
            patch.object(self.module, "_build_kv_sparse_metadata_kernel", kernel),
            patch.object(
                self.module, "_check_component", wraps=self.module._check_component
            ) as check,
        ):
            self.transfer()
        self.assertEqual(check.call_count, 1)
        kernel.__getitem__.assert_called_once_with((1, 1))
        kernel.__getitem__.return_value.assert_called_once()
        self.offload.sparse_copy.assert_called_once()

    def test_cross_stream_dependency_covers_both_directions(self):
        self.transfer()
        first = self.module._last_sparse_event[NPU]
        second_stream = SimpleNamespace(wait_event=Mock())
        self.stream = second_stream
        self.transfer(direction=2)
        second_stream.wait_event.assert_called_once_with(first)
        self.assertEqual(self.offload.sparse_copy.call_args.args[-1], 2)
        self.assertIs(self.module._last_sparse_event[NPU].stream, second_stream)

    def test_submission_failure_keeps_metadata_and_orders_next_call(self):
        self.offload.sparse_copy.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "code=7"):
            self.transfer()
        failed = self.module._last_sparse_event[NPU]
        self.assertTrue(self.module._inflight)
        self.offload.sparse_copy.return_value = 0
        self.transfer()
        self.stream.wait_event.assert_called_with(failed)

    def test_pinned_cpu_staging_is_retained_and_released_after_completion(self):
        _, _, di, hi = self.transfer(cpu_indices=True)
        retained = self.module._inflight[0][1]
        self.assertIn(di.pinned, retained)
        self.assertIn(hi.pinned, retained)
        self.module._inflight[0][0].done = True
        self.transfer()
        self.assertEqual(len(self.module._inflight), 1)

    def test_wrong_scene_and_bad_index_shape_fail_before_submission(self):
        self.offload._get_scene = lambda: 0
        with self.assertRaisesRegex(RuntimeError, "LOCAL_URMA"):
            self.transfer()
        self.builder.assert_not_called()
        with self.assertRaisesRegex(ValueError, "complete pages"):
            self.call(
                Tensor((3,)),
                Tensor((3,)),
                Tensor((1, 2, 4, 1, 5)),
                Tensor((2, 1, 4, 1, 5), CPU),
                Tensor((0,)),
                Tensor((0,), CPU),
                page_size=4,
            )
        self.offload.sparse_copy.assert_not_called()

    def test_sparse_copy_keeps_host_pointers_in_both_scenes_and_directions(self):
        self.offload.get_dva.side_effect = AssertionError(
            "sparse_copy must not translate Host pool pointers"
        )
        for scene in (1, 2):
            self.offload._get_scene = lambda: scene
            for direction in (1, 2):
                self.offload.sparse_copy.reset_mock()
                _, host, _, _ = self.transfer(direction=direction)
                self.offload.sparse_copy.assert_called_once()
                self.assertIs(self.builder.call_args.args[0][0][1], host)
                self.assertEqual(self.builder.call_args.kwargs, {})
                self.offload.get_dva.assert_not_called()

    def test_empty_transfer_has_no_backend_side_effect(self):
        self.transfer(pages=0)
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()

    def public_wrapper(self):
        return self.call

    def test_public_defaults_and_explicit_overrides(self):
        implementations = {}
        def register_impl(name, impl, **options):
            self.assertNotIn("supported_generations", options)
            self.assertNotIn("excluded_generations", options)
            def register(fn):
                implementations[name, impl] = fn
                return fn
            return register
        package = "sgl_kernel_npu.operators.mem_cache"
        parent = ModuleType(package)
        parent.__path__ = []
        name = package + ".kv_transfer"
        path = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/operators/mem_cache/kv_transfer.py"
        spec = importlib.util.spec_from_file_location(name, path)
        public = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            package: parent,
            "sgl_kernel_npu.operators.utils.registry": SimpleNamespace(
                register_impl=register_impl,
                register_operator=lambda _: lambda fn: fn,
                get_impl_by_name=lambda name, impl: implementations[name, impl]),
        }):
            spec.loader.exec_module(public)
        public._memfabric_fn = Mock()
        args = ("di", "hi", "dk", "hk", "dv", "hv")
        for suffix in ("_pf_lf", ""):
            fn = getattr(public, "transfer_kv_per_layer_kv_split" + suffix)
            for impl, expected in (("auto", "aicpu"), ("aiv", "aiv"), ("aicpu", "aicpu")):
                fn(*args, device_layer_id=2, host_layer_id=4, impl=impl)
                kw = public._memfabric_fn.call_args.kwargs
                self.assertEqual((kw["backend"], kw["direction"]), (expected, 1))
        for suffix in ("_lf_pf", ""):
            fn = getattr(public, "transfer_kv_all_layer_kv_split" + suffix)
            for impl, expected in (("auto", "aiv"), ("aicpu", "aicpu"), ("aiv", "aiv")):
                fn(*args, impl=impl)
                kw = public._memfabric_fn.call_args.kwargs
                self.assertEqual((kw["backend"], kw["direction"]), (expected, 2))

    def test_public_wrapper_rejects_invalid_required_or_empty_buffers(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7))
        host_k = Tensor((9, 3, 4, 1, 7), CPU)
        v = Tensor((3, 8, 4, 1, 2))
        host_v = Tensor((9, 3, 4, 1, 2), CPU)
        args = dict(
            device_indices=Tensor((8,)),
            host_indices=Tensor((8,)),
            device_k=k,
            host_k=host_k,
            device_v=v,
            host_v=host_v,
            page_size=4,
        )
        invalid = (
            {"device_k": None},
            {"host_k": None},
            {"device_k": Tensor((0,))},
            {"host_k": Tensor((0,), CPU)},
            {"device_v": None},
            {"host_v": None},
            {"host_v": Tensor((0,), CPU)},
            {"device_index_k": Tensor((0,)), "host_index_k": host_k},
            {"device_index_k_scale": k, "host_index_k_scale": Tensor((0,), CPU)},
        )
        for kwargs in invalid:
            with self.subTest(overrides=tuple(kwargs)):
                with self.assertRaises(ValueError):
                    call(**dict(args, **kwargs))
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()

    def test_optional_components_skip_either_missing_end_independently(self):
        call = self.public_wrapper()
        k, h = Tensor((3, 8, 4, 1, 7)), Tensor((9, 3, 4, 1, 7), CPU)
        index, host_index = Tensor((2, 8, 4, 1, 3)), Tensor((9, 2, 4, 1, 3), CPU)
        scale, host_scale = Tensor((2, 8, 4, 1)), Tensor((9, 2, 4, 1, 1), CPU)
        for missing in (
            "device_index_k",
            "host_index_k",
            "device_index_k_scale",
            "host_index_k_scale",
        ):
            for direction in (1, 2):
                with self.subTest(missing=missing, direction=direction):
                    optional = dict(
                        device_index_k=index,
                        host_index_k=host_index,
                        device_index_k_scale=scale,
                        host_index_k_scale=host_scale,
                    )
                    optional[missing] = None
                    index.recorded.clear()
                    scale.recorded.clear()
                    call(
                        Tensor((8,)),
                        Tensor((8,)),
                        k,
                        h,
                        Tensor((0,)),
                        Tensor((0,), CPU),
                        page_size=4,
                        direction=direction,
                        **optional,
                    )
                    pairs = self.builder.call_args.args[0]
                    self.assertEqual(len(pairs), 2)
                    if missing.endswith("_scale"):
                        self.assertEqual(pairs, [(k, h), (index, host_index)])
                        self.assertEqual(scale.recorded, [])
                    else:
                        self.assertEqual(pairs[1][0].shape, (2, 8, 4, 1, 1))
                        self.assertIs(pairs[1][1], host_scale)
                        self.assertEqual(index.recorded, [])

    def test_public_wrapper_accepts_empty_device_v_with_host_placeholder(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7), dtype="fp8")
        host_k = Tensor((9, 3, 4, 1, 7), CPU, "fp8")
        host_v = Tensor((9, 3, 4, 1, 2), CPU, "bf16")
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            host_k,
            Tensor((3, 8, 4, 1, 0)),
            host_v,
            page_size=4,
        )
        self.assertEqual(self.builder.call_args.args[0], [(k, host_k)])

    def test_public_wrapper_skips_only_paired_empty_optional_buffers(self):
        call = self.public_wrapper()
        k, h = Tensor((3, 8, 4, 1, 7)), Tensor((9, 3, 4, 1, 7), CPU)
        v, hv = Tensor((3, 8, 4, 1, 2)), Tensor((9, 3, 4, 1, 2), CPU)
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            h,
            v,
            hv,
            device_index_k=Tensor((0,)),
            host_index_k=Tensor((0,), CPU),
            page_size=4,
        )
        self.assertEqual(self.builder.call_args.args[0], [(k, h), (v, hv)])

    def test_public_wrapper_skips_empty_v_and_pads_indexer_scale(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7), dtype="fp8")
        h = Tensor((9, 3, 4, 1, 7), CPU, "fp8")
        scale = Tensor((2, 8, 4, 1), dtype="fp32")
        hs = Tensor((9, 2, 4, 1, 1), CPU, "fp32")
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            h,
            Tensor((0,)),
            Tensor((0,), CPU),
            device_index_k_scale=scale,
            host_index_k_scale=hs,
            page_size=4,
        )
        pairs = self.builder.call_args.args[0]
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1][0].shape, (2, 8, 4, 1, 1))
        self.assertIs(pairs[1][1], hs)


if __name__ == "__main__":
    unittest.main()
