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
from unittest.mock import Mock, create_autospec, patch

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

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def contiguous(self):
        return self

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

    def narrow(self, axis, start, length):
        key = [slice(None)] * self.ndim
        key[axis] = slice(start, start + length)
        return self[tuple(key)]

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        shape = list(self.shape)
        for axis, item in enumerate(key):
            start, stop, step = item.indices(shape[axis])
            shape[axis] = len(range(start, stop, step))
        return self if tuple(shape) == self.shape else Tensor(shape, self.device, self.dtype)


class Stream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.wait_event = Mock()

    def __eq__(self, other):
        return isinstance(other, Stream) and self.stream_id == other.stream_id


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
        self.stream = Stream()

        def build(components, di, hi, page_size, direction):
            total = (
                di.numel()
                // page_size
                * sum(pair[0].shape[0] for pair in components if pair is not None)
            )
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
                is_current_stream_capturing=Mock(return_value=False),
                Event=Event,
            ),
        )
        self.offload = SimpleNamespace(
            get_dva=Mock(return_value=1 << 44),
            sparse_copy=create_autospec(
                lambda src, dst, lengths, count, device: 0, return_value=0
            ),
        )
        language = SimpleNamespace(constexpr=object)
        triton = SimpleNamespace(jit=lambda **_: lambda fn: fn, language=language)
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "memfabric_hybrid": SimpleNamespace(offload=self.offload),
                "triton": triton,
                "triton.language": language,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        module_name = "sgl_kernel_npu.implementation.memfabric.mem_cache.kv_transfer"
        path = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/memfabric/mem_cache/kv_transfer.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
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

    def test_preselected_views_are_reused_without_slicing(self):
        dev = Tensor((1, 8, 4, 1, 5), dtype="bf16")
        host = Tensor((10, 1, 4, 1, 5), CPU, "bf16")
        with patch.object(Tensor, "narrow", side_effect=AssertionError("redundant slice")):
            selected = self.module._layer_views(dev, host, "page_first", (0, 0, 1))
            self.assertIs(selected[0], dev)
            self.assertIs(selected[1], host)

    def test_layer_views_select_only_requested_ranges(self):
        dev = Tensor((4, 8, 4, 1, 5))
        for layout, shape in (
            ("page_first", (10, 6, 4, 1, 5)),
            ("layer_first", (6, 10, 4, 1, 5)),
        ):
            with self.subTest(layout=layout):
                host = Tensor(shape, CPU)
                selected = self.module._layer_views(dev, host, layout, (2, 3, 1))
                self.assertEqual(selected[0].shape, (1, 8, 4, 1, 5))
                self.assertEqual(selected[1].shape, (10, 1, 4, 1, 5))
                with self.assertRaisesRegex(ValueError, "layer selection"):
                    self.module._layer_views(dev, host, layout, (4, 3, 1))

    def test_one_submission_uses_complete_metadata_and_retains_it(self):
        self.transfer()
        self.builder.assert_called_once()
        self.offload.sparse_copy.assert_called_once()
        args = self.offload.sparse_copy.call_args.args
        retained = self.module._inflight[0][1]
        for submitted, table in zip(args[:4], self.tables):
            self.assertIs(submitted, table)
            self.assertIn(table, retained)
        self.assertEqual(len(args), 5)
        self.assertEqual(args[-1], NPU)

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

    def test_aicpu_validates_each_component_and_submits_one_copy(self):
        with patch.object(
            self.module, "_check_component", wraps=self.module._check_component
        ) as check:
            device_k = Tensor((3, 5, 4, 1, 5), dtype="bf16")
            host_k = Tensor((7, 3, 4, 1, 5), CPU, "bf16")
            device_v = Tensor((3, 5, 4, 1, 2), dtype="bf16")
            host_v = Tensor((7, 3, 4, 1, 2), CPU, "bf16")
            self.call(
                Tensor((8,)), Tensor((8,)),
                device_k, host_k, device_v, host_v,
                page_size=4,
            )
        self.assertEqual(check.call_count, 2)
        self.builder.assert_called_once()
        self.offload.sparse_copy.assert_called_once()
        src, dst, lengths, count = self.offload.sparse_copy.call_args.args[:4]
        self.assertEqual(src.shape, (12,))
        self.assertEqual(dst.shape, (12,))
        self.assertEqual(lengths.shape, (12,))
        self.assertEqual(count.shape, (1,))

    def test_capture_skips_eager_event_queries_waits_and_retirement(self):
        self.transfer()
        previous = self.module._last_sparse_submission[NPU]
        previous_done = previous[1]
        previous_done.query = Mock(side_effect=AssertionError("event query during capture"))
        self.torch.npu.Event = Mock(
            side_effect=AssertionError("event creation during capture")
        )
        self.torch.npu.is_current_stream_capturing.return_value = True
        self.stream.wait_event.side_effect = AssertionError("eager wait during capture")
        self.builder.reset_mock()
        self.offload.sparse_copy.reset_mock()
        for direction in (1, 2):
            self.transfer(direction=direction)
        self.assertEqual(self.builder.call_count, 2)
        self.assertEqual(self.offload.sparse_copy.call_count, 2)
        self.assertEqual(len(self.module._inflight), 1)
        self.assertIs(self.module._last_sparse_submission[NPU], previous)
        previous_done.query.assert_not_called()
        self.torch.npu.Event.assert_not_called()
        self.stream.wait_event.assert_not_called()

    def test_capture_without_eager_work_creates_no_retirement_state(self):
        self.torch.npu.is_current_stream_capturing.return_value = True
        self.torch.npu.Event = Mock(
            side_effect=AssertionError("event creation during capture")
        )
        self.transfer()
        self.assertFalse(self.module._inflight)
        self.assertFalse(self.module._last_sparse_submission)
        self.offload.sparse_copy.assert_called_once()

    def test_capture_rejects_cpu_indices_and_unsupported_backend(self):
        self.torch.npu.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(ValueError, "indices must be on the KV buffer's NPU"):
            self.transfer(cpu_indices=True)
        with self.assertRaisesRegex(RuntimeError, "impl='aicpu'"):
            self.call(
                Tensor((8,)),
                Tensor((8,)),
                Tensor((1, 8, 4, 1, 5)),
                Tensor((10, 1, 4, 1, 5), CPU),
                Tensor((0,)),
                Tensor((0,), CPU),
                page_size=4,
                backend="aiv",
            )
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()
        self.assertFalse(self.module._inflight)

    def test_capture_submission_failure_does_not_create_eager_events(self):
        self.torch.npu.is_current_stream_capturing.return_value = True
        self.torch.npu.Event = Mock(
            side_effect=AssertionError("event creation during capture")
        )
        self.offload.sparse_copy.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "code=7"):
            self.transfer()
        self.torch.npu.Event.assert_not_called()
        self.assertFalse(self.module._inflight)
        self.assertFalse(self.module._last_sparse_submission)

    def test_same_stream_skips_wait_and_retains_each_completion_event(self):
        self.transfer()
        first = self.module._last_sparse_submission[NPU][1]
        self.transfer(direction=2)
        second = self.module._last_sparse_submission[NPU][1]
        self.stream.wait_event.assert_not_called()
        self.assertIsNot(first, second)
        self.assertEqual(len(self.module._inflight), 2)
        self.assertIs(first.stream, self.stream)
        self.assertIs(second.stream, self.stream)

    def test_equivalent_stream_wrappers_skip_wait(self):
        self.transfer()
        first_stream = self.stream
        self.stream = Stream(first_stream.stream_id)
        self.assertIsNot(first_stream, self.stream)
        self.transfer()
        self.stream.wait_event.assert_not_called()

    def test_cross_stream_dependency_covers_both_directions(self):
        self.transfer()
        first = self.module._last_sparse_submission[NPU][1]
        second_stream = Stream(1)
        self.stream = second_stream
        self.transfer(direction=2)
        second_stream.wait_event.assert_called_once_with(first)
        self.assertEqual(self.offload.sparse_copy.call_args.args[-1], NPU)
        self.assertEqual(self.builder.call_args.args[-1], 2)
        latest_stream, latest_done = self.module._last_sparse_submission[NPU]
        self.assertIs(latest_stream, second_stream)
        self.assertIs(latest_done.stream, second_stream)

    def test_switching_back_waits_for_the_latest_submission(self):
        self.transfer()
        first_stream = self.stream
        first = self.module._last_sparse_submission[NPU][1]
        second_stream = Stream(1)
        self.stream = second_stream
        self.transfer()
        second = self.module._last_sparse_submission[NPU][1]
        self.stream = first_stream
        self.transfer()
        second_stream.wait_event.assert_called_once_with(first)
        first_stream.wait_event.assert_called_once_with(second)

    def test_submission_failure_keeps_inputs_and_orders_next_call(self):
        self.offload.sparse_copy.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "code=7"):
            self.transfer()
        failed = self.module._last_sparse_submission[NPU][1]
        self.assertTrue(self.module._inflight)
        self.offload.sparse_copy.return_value = 0
        self.stream = Stream(1)
        self.transfer()
        self.stream.wait_event.assert_called_once_with(failed)

    def test_indices_must_share_the_kv_npu_before_submission(self):
        self.torch.npu.Event = Mock(side_effect=AssertionError("unexpected event"))
        for backend in ("aicpu", "aiv"):
            for index_device in (CPU, Device("npu", 1)):
                for side in (0, 1):
                    with self.subTest(backend=backend, device=index_device, side=side):
                        indices = [Tensor((8,)), Tensor((8,))]
                        indices[side] = Tensor((8,), index_device)
                        with self.assertRaisesRegex(ValueError, "indices must be on"):
                            self.call(
                                *indices,
                                Tensor((1, 8, 4, 1, 5)),
                                Tensor((10, 1, 4, 1, 5), CPU),
                                Tensor((0,)),
                                Tensor((0,), CPU),
                                page_size=4,
                                backend=backend,
                            )
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()
        self.torch.npu.Event.assert_not_called()
        self.assertFalse(self.module._inflight)

    def test_aiv_retains_inputs_and_pinned_metadata_staging(self):
        staging = Tensor((44,), CPU)
        submit = Mock(return_value=staging)
        self.torch.ops = SimpleNamespace(npu=SimpleNamespace(kv_exchange=submit))
        with patch.object(self.module, "_resolve_host_dva", return_value=1 << 44):
            for direction in (1, 2):
                di, hi = Tensor((8,)), Tensor((8,))
                dev = Tensor((1, 8, 4, 1, 5))
                host = Tensor((10, 1, 4, 1, 5), CPU)
                self.call(
                    di, hi, dev, host, Tensor((0,)), Tensor((0,), CPU),
                    page_size=4, direction=direction, backend="aiv",
                )
                self.assertEqual(submit.call_args.args[2], [dev])
                self.assertEqual(submit.call_args.args[3], [host])
                done, retained = self.module._inflight[-1]
                self.assertIs(done.stream, self.stream)
                for tensor in (di, hi, dev, host, staging):
                    self.assertIn(tensor, retained)
        self.assertEqual(submit.call_count, 2)
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()
        self.assertFalse(self.module._last_sparse_submission)

    def test_component_selection_preserves_missing_only_when_requested(self):
        k, hk = Tensor((3, 8, 4, 1, 7)), Tensor((9, 3, 4, 1, 7), CPU)
        v, hv = Tensor((3, 8, 4, 1, 2)), Tensor((9, 3, 4, 1, 2), CPU)
        index, hi = Tensor((2, 8, 4, 1, 3)), Tensor((9, 2, 4, 1, 3), CPU)
        scale, hs = Tensor((2, 8, 4, 1, 1)), Tensor((9, 2, 4, 1, 1), CPU)
        for mask in range(8):
            with self.subTest(mask=mask):
                expected = [
                    (k, hk),
                    (v, hv) if mask & 1 else None,
                    (index, hi) if mask & 2 else None,
                    (scale, hs) if mask & 4 else None,
                ]
                args = (
                    k,
                    hk,
                    v if mask & 1 else Tensor((0,)),
                    hv,
                    index if mask & 2 else None,
                    hi,
                    scale if mask & 4 else None,
                    hs,
                    "page_first",
                    (0, 0, None),
                    (0, 0, None),
                )
                self.assertEqual(
                    self.module.select_components(*args),
                    [pair for pair in expected if pair is not None],
                )
                self.assertEqual(
                    self.module.select_components(*args, preserve_missing=True),
                    expected,
                )

    def test_bad_index_shape_fails_before_submission(self):
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

    def test_sparse_copy_keeps_host_pointers_and_infers_both_directions(self):
        self.offload.get_dva.side_effect = AssertionError(
            "sparse_copy must not translate Host pool pointers"
        )
        for direction in (1, 2):
            self.offload.sparse_copy.reset_mock()
            _, host, _, _ = self.transfer(direction=direction)
            self.offload.sparse_copy.assert_called_once()
            self.assertEqual(len(self.offload.sparse_copy.call_args.args), 5)
            self.assertIs(self.builder.call_args.args[0][0][1], host)
            self.assertEqual(self.builder.call_args.args[-1], direction)
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
            {"device_k": Tensor((3, 8, 4, 7))},
            {"host_k": Tensor((9, 3, 4, 7), CPU)},
            {"kv_layers": (0, 0, 0)},
            {"kv_layers": (0, 0, 4)},
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
                    self.assertEqual(len(pairs), 4)
                    self.assertIsNone(pairs[1])
                    if missing.endswith("_scale"):
                        self.assertEqual(
                            pairs, [(k, h), None, (index, host_index), None]
                        )
                        self.assertEqual(scale.recorded, [])
                    else:
                        self.assertIsNone(pairs[2])
                        self.assertEqual(pairs[3][0].shape, (2, 8, 4, 1, 1))
                        self.assertIs(pairs[3][1], host_scale)
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
        self.assertEqual(
            self.builder.call_args.args[0], [(k, host_k), None, None, None]
        )

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
        self.assertEqual(self.builder.call_args.args[0], [(k, h), (v, hv), None, None])

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
        self.assertEqual(len(pairs), 4)
        self.assertIsNone(pairs[1])
        self.assertIsNone(pairs[2])
        self.assertEqual(pairs[3][0].shape, (2, 8, 4, 1, 1))
        self.assertIs(pairs[3][1], hs)


if __name__ == "__main__":
    unittest.main()
