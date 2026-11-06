"""CPU checks of the real wrapper's submission, lifetime and stream ordering.

Only the NPU/Memfabric boundary is mocked; no PyTorch/CANN install is needed.
"""

import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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
        for dim in self.shape[axis + 1 :]:
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
        return (
            self
            if tuple(shape) == self.shape
            else Tensor(shape, self.device, self.dtype)
        )


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

        def submit(
            components,
            di,
            hi,
            page_size,
            direction,
            host_layout,
        ):
            pass

        self.builder = Mock(side_effect=submit)
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            int64="int64",
            int32="int32",
            iinfo=lambda _: SimpleNamespace(max=0x7FFFFFFF),
            npu=SimpleNamespace(
                device=lambda _: Context(),
                current_stream=lambda: self.stream,
                is_current_stream_capturing=Mock(return_value=False),
                Event=Event,
            ),
        )
        self.offload = SimpleNamespace()
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "memfabric_hybrid": SimpleNamespace(offload=self.offload),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        module_name = "sgl_kernel_npu.implementation.memfabric.mem_cache.kv_transfer"
        path = (
            ROOT
            / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/memfabric/mem_cache/kv_transfer.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.real_submit = self.module._submit_aiv
        self.module._submit_aiv = self.builder
        self.call = partial(
            self.module.transfer_kv,
            host_layout="page_first",
            kv_layers=(0, 0, None),
            index_layers=(0, 0, None),
            direction=1,
        )

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
        with patch.object(
            Tensor, "narrow", side_effect=AssertionError("redundant slice")
        ):
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

    def test_submission_needs_no_event_bookkeeping(self):
        # Input lifetime rides on the caching allocator's stream-ordered
        # reuse (the C++ adapter record_stream's the NPU inputs), so the
        # Python path must not create or poll any completion event.
        self.torch.npu.Event = Mock(side_effect=AssertionError("unexpected event"))
        self.transfer()
        self.transfer(direction=2)
        self.torch.npu.Event.assert_not_called()
        self.assertEqual(self.builder.call_count, 2)

    def test_indices_must_share_the_kv_npu_before_submission(self):
        self.torch.npu.Event = Mock(side_effect=AssertionError("unexpected event"))
        for index_device in (CPU, Device("npu", 1)):
            for side in (0, 1):
                with self.subTest(device=index_device, side=side):
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
                        )
        self.builder.assert_not_called()
        self.torch.npu.Event.assert_not_called()

    def test_aiv_submits_component_pairs_without_event_bookkeeping(self):
        submit = Mock(return_value=None)
        self.torch.ops = SimpleNamespace(npu=SimpleNamespace(kv_exchange=submit))
        with patch.object(self.module, "_submit_aiv", self.real_submit), patch.object(
            self.module, "_resolve_host_dva", return_value=1 << 44
        ):
            for direction in (1, 2):
                di, hi = Tensor((8,)), Tensor((8,))
                dev = Tensor((1, 8, 4, 1, 5))
                host = Tensor((10, 1, 4, 1, 5), CPU)
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
                self.assertEqual(submit.call_args.args[2], [dev])
                self.assertEqual(submit.call_args.args[3], [host])
        self.assertEqual(submit.call_count, 2)
        self.builder.assert_not_called()

    def test_aiv_transfer_runs_without_memfabric_hybrid_available(self):
        """Registered-host AIV transfers must not import MemFabric."""
        submit = Mock(return_value=None)
        self.torch.ops = SimpleNamespace(npu=SimpleNamespace(kv_exchange=submit))
        with patch.dict(sys.modules, {"memfabric_hybrid": None}), patch.object(
            self.module, "_submit_aiv", self.real_submit
        ), patch.object(self.module, "_resolve_host_dva", return_value=1 << 44):
            for direction in (1, 2):
                self.call(
                    Tensor((8,)),
                    Tensor((8,)),
                    Tensor((1, 8, 4, 1, 5)),
                    Tensor((10, 1, 4, 1, 5), CPU),
                    Tensor((0,)),
                    Tensor((0,), CPU),
                    page_size=4,
                    direction=direction,
                )
            self.assertEqual(submit.call_count, 2)

    def test_component_selection_omits_missing_components(self):
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
        self.builder.assert_not_called()

    def test_empty_transfer_has_no_backend_side_effect(self):
        self.transfer(pages=0)
        self.builder.assert_not_called()

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
        path = (
            ROOT
            / "python/sgl_kernel_npu/sgl_kernel_npu/operators/mem_cache/kv_transfer.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        public = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                package: parent,
                "sgl_kernel_npu.operators.utils.registry": SimpleNamespace(
                    register_impl=register_impl,
                    register_operator=lambda _: lambda fn: fn,
                    get_impl_by_name=lambda name, impl: implementations[name, impl],
                ),
            },
        ):
            spec.loader.exec_module(public)
        public._memfabric_fn = Mock()
        args = ("di", "hi", "dk", "hk", "dv", "hv")
        for suffix in ("_pf_lf", ""):
            fn = getattr(public, "transfer_kv_per_layer_kv_split" + suffix)
            for impl in ("auto", "aiv"):
                fn(*args, device_layer_id=2, host_layer_id=4, impl=impl)
                kw = public._memfabric_fn.call_args.kwargs
                self.assertEqual(kw["direction"], 1)
        for suffix in ("_lf_pf", ""):
            fn = getattr(public, "transfer_kv_all_layer_kv_split" + suffix)
            for impl in ("auto", "aiv"):
                fn(*args, impl=impl)
                kw = public._memfabric_fn.call_args.kwargs
                self.assertEqual(kw["direction"], 2)

    def test_acl_backend_routes_auto_to_aiv_and_keeps_explicit_impls(self):
        """Auto and explicit AIV transfers work with ACL host memory."""
        implementations = {}

        def register_impl(name, impl, **options):
            def register(fn):
                implementations[name, impl] = fn
                return fn

            return register

        package = "sgl_kernel_npu.operators.mem_cache"
        parent = ModuleType(package)
        parent.__path__ = []
        name = package + ".kv_transfer"
        path = (
            ROOT
            / "python/sgl_kernel_npu/sgl_kernel_npu/operators/mem_cache/kv_transfer.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        public = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                package: parent,
                "sgl_kernel_npu.operators.utils.registry": SimpleNamespace(
                    register_impl=register_impl,
                    register_operator=lambda _: lambda fn: fn,
                    get_impl_by_name=lambda name, impl: implementations[name, impl],
                ),
            },
        ):
            spec.loader.exec_module(public)
        public._memfabric_fn = Mock()
        args = ("di", "hi", "dk", "hk", "dv", "hv")
        with patch.dict(os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": "acl"}):
            for suffix in ("_pf_lf", ""):
                fn = getattr(public, "transfer_kv_per_layer_kv_split" + suffix)
                for impl in ("auto", "aiv"):
                    fn(*args, device_layer_id=2, host_layer_id=4, impl=impl)
                    kw = public._memfabric_fn.call_args.kwargs
                    self.assertEqual(kw["direction"], 1)
            for suffix in ("_lf_pf", ""):
                fn = getattr(public, "transfer_kv_all_layer_kv_split" + suffix)
                for impl in ("auto", "aiv"):
                    fn(*args, impl=impl)
                    kw = public._memfabric_fn.call_args.kwargs
                    self.assertEqual(kw["direction"], 2)
        public._memfabric_fn.reset_mock()
        with patch.dict(os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": ""}):
            getattr(public, "transfer_kv_per_layer_kv_split_pf_lf")(
                *args, device_layer_id=2, host_layer_id=4, impl="auto"
            )
        self.assertIs(
            implementations["mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "auto"],
            implementations["mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "aiv"],
        )

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
                    self.assertEqual(pairs[0], (k, h))
                    if missing.endswith("_scale"):
                        self.assertEqual(pairs[1], (index, host_index))
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
        self.assertEqual(pairs[0], (k, h))
        self.assertEqual(pairs[1][0].shape, (2, 8, 4, 1, 1))
        self.assertIs(pairs[1][1], hs)


if __name__ == "__main__":
    unittest.main()
