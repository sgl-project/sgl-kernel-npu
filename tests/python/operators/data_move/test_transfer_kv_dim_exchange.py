import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize(
    "direction,layout,pattern",
    [
        pytest.param(direction, layout, "varied", id=f"{layout}-{direction.lower()}")
        for direction in ("D2H", "H2D")
        for layout in ("kv", "kv-index", "k-only", "scale")
    ]
    + [pytest.param("D2H", "mla", "varied", id="mla-indexer-sparse-pages")]
    + [
        pytest.param(
            direction, layout, "uniform", id=f"{layout}-{direction.lower()}-zero-one"
        )
        for direction in ("D2H", "H2D")
        for layout in ("kv", "kv-index", "k-only")
    ],
)
def test_transfer_kv_dim_exchange(direction, layout, pattern):
    from sgl_kernel_npu.operators.data_move import (
        TransferDirection,
        transfer_kv_dim_exchange,
    )

    # Qwen3-32B TP=2 per-card shape; MLA uses only K and sparse indexer pages.
    sparse = layout == "mla"
    layers, pages, host_pages = (2, 4, 5) if sparse else (64, 30, 30)
    heads, width = (1, 1) if layout in ("scale", "mla") else (4, 128)
    page_size = 128
    to_host = direction == "D2H"

    def pair(dtype, heads, width, offset=1, guard=False):
        dev = torch.empty(
            layers * (2 if guard else 1),
            pages,
            page_size,
            heads,
            width,
            dtype=dtype,
            device="npu",
        )
        host = torch.empty(
            host_pages * (2 if guard else 1),
            layers,
            page_size,
            heads,
            width,
            dtype=dtype,
            pin_memory=True,
            device="cpu",
        )
        source, destination = (dev, host) if to_host else (host, dev)
        values = torch.arange(
            source.size(0) * source.size(1), device=source.device
        ).reshape(*source.shape[:2], 1, 1, 1)
        if pattern == "uniform":
            # Exercise zero-source overwrites as well as one-source copies.
            source.fill_(1 if to_host else 0)
            destination.fill_(0 if to_host else 1)
        else:
            source.copy_(values % 97 + offset)
            destination.zero_()
            if guard:
                unused = host[host_pages:] if to_host else dev[layers:]
                unused.fill_(7)
        return dev, host

    k_storage, host_storage = pair(
        torch.float32 if heads == 1 else torch.bfloat16,
        heads,
        width,
        guard=layout == "k-only",
    )
    k, host_k = k_storage[:layers], host_storage[:host_pages]
    has_v = layout in ("kv", "kv-index")
    v, host_v = (
        pair(k.dtype, heads, width, offset=2)
        if has_v
        else (torch.empty(0, device="cpu"), torch.empty(0, device="cpu"))
    )
    kwargs = dict(
        device_k=k,
        host_k=host_k,
        device_v=v,
        host_v=host_v,
        page_size=page_size,
        direction=TransferDirection[direction],
    )
    pairs = [("device_k", "host_k")]
    if has_v:
        pairs.append(("device_v", "host_v"))
    if layout in ("kv-index", "mla"):
        kwargs["device_index_k"], kwargs["host_index_k"] = pair(
            torch.int8 if sparse else k.dtype, heads, width, offset=3
        )
        pairs.append(("device_index_k", "host_index_k"))
    if layout in ("scale", "mla"):
        scale, kwargs["host_index_k_scale"] = pair(torch.float32, 1, 1, offset=4)
        kwargs["device_index_k_scale"] = scale.squeeze(-1)
        pairs.append(("device_index_k_scale", "host_index_k_scale"))
    tokens = torch.arange(page_size, device="cpu")
    kwargs["device_indices"] = (
        torch.cat([tokens + 256, tokens])
        if sparse
        else torch.arange(pages * page_size, device="cpu")
    )
    kwargs["host_indices"] = (
        torch.cat([tokens + 128, tokens + 512]) if sparse else kwargs["device_indices"]
    )

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    destinations = [host if to_host else dev for dev, host in pairs]
    if checked:
        require_distinct_impls(transfer_kv_dim_exchange, impl, impl_ref)
        reference = kwargs.copy()
        for key in destinations:
            reference[key] = (
                torch.zeros_like(kwargs[key], pin_memory=True)
                if to_host
                else torch.zeros_like(kwargs[key])
            )
        transfer_kv_dim_exchange(**reference, impl=impl_ref)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert transfer_kv_dim_exchange(**kwargs, impl=impl) is None
    stream.synchronize()
    if checked:
        for key in destinations:
            # One layer/page at a time bounds comparison memory for the full Qwen shape.
            for actual, expected in zip(kwargs[key], reference[key]):
                assert_accuracy(
                    actual.view(torch.uint8), expected.view(torch.uint8), path=key
                )
        # Check the requested mapping independently of the reference backend.
        for dk, hk in pairs:
            device_page = kwargs[dk][:, 2 if sparse else 0].cpu()
            host_page = kwargs[hk][1 if sparse else 0].reshape_as(device_page)
            assert_accuracy(device_page.view(torch.uint8), host_page.view(torch.uint8))
        if layout == "k-only":
            unused = host_storage[host_pages:] if to_host else k_storage[layers:]
            sentinel = (0 if to_host else 1) if pattern == "uniform" else 7
            assert_accuracy(
                unused.view(torch.uint8),
                torch.full_like(unused, sentinel).view(torch.uint8),
            )
    executor = make_executor(
        transfer_kv_dim_exchange,
        kwargs=kwargs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=sparse)


@pytest.mark.parametrize("direction", ["D2H", "H2D"])
@pytest.mark.parametrize("scale_ndim", [4, 5])
@pytest.mark.parametrize("has_v", [False, True], ids=["k-only", "kv"])
def test_transfer_kv_indexer_scale(direction, scale_ndim, has_v):
    from sgl_kernel_npu.operators.data_move import (
        TransferDirection,
        transfer_kv_dim_exchange,
    )

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    layers, pages, host_pages, page_size = 2, 4, 5, 4
    kwargs = dict(page_size=page_size, direction=TransferDirection[direction])
    pairs = []
    specifications = [("k", torch.float16, 2, 3)]
    if has_v:
        specifications.append(("v", torch.float16, 2, 5))
    specifications.extend(
        [("index_k", torch.int8, 1, 3), ("index_k_scale", torch.float32, 1, 1)]
    )
    for component, dtype, heads, width in specifications:
        # Nonzero storage offsets must not change the page mapping.
        device = torch.empty(
            layers + 1, pages, page_size, heads, width, dtype=dtype, device="npu"
        )[1:]
        host = torch.empty(
            host_pages + 1,
            layers,
            page_size,
            heads,
            width,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )[1:]
        source, target = (device, host) if direction == "D2H" else (host, device)
        source.copy_(
            (torch.arange(source.numel(), device=source.device) % 97).reshape_as(source)
        )
        target.fill_(-7)
        if component == "index_k_scale" and scale_ndim == 4:
            device = device.squeeze(-1)
        device_key, host_key = f"device_{component}", f"host_{component}"
        kwargs[device_key], kwargs[host_key] = device, host
        pairs.append((device_key, host_key))
    if not has_v:
        kwargs["device_v"] = torch.empty(0, device="cpu")
        kwargs["host_v"] = torch.empty(0, device="cpu")
    tokens = torch.arange(page_size, device="cpu")
    kwargs["device_indices"] = torch.cat([tokens + 2 * page_size, tokens])
    kwargs["host_indices"] = torch.cat([tokens + page_size, tokens + 4 * page_size])
    targets = [hk if direction == "D2H" else dk for dk, hk in pairs]
    if checked:
        require_distinct_impls(transfer_kv_dim_exchange, impl, impl_ref)
        reference = kwargs.copy()
        expected = {key: kwargs[key].cpu().clone() for key in targets}
        for key in targets:
            reference[key] = (
                torch.empty_like(kwargs[key], pin_memory=True).copy_(kwargs[key])
                if direction == "D2H"
                else kwargs[key].clone()
            )
        for dk, hk in pairs:
            dev, host = kwargs[dk].cpu(), kwargs[hk].cpu()
            for d, h in ((2, 1), (0, 4)):
                for layer in range(layers):
                    if direction == "D2H":
                        block = expected[hk][h, layer]
                        block.copy_(dev[layer, d].reshape_as(block))
                    else:
                        block = expected[dk][layer, d]
                        block.copy_(host[h, layer].reshape_as(block))
        transfer_kv_dim_exchange(**reference, impl=impl_ref)
    transfer_kv_dim_exchange(**kwargs, impl=impl)
    torch.npu.synchronize()
    if checked:
        for key in targets:
            assert_accuracy(
                kwargs[key].cpu().view(torch.uint8),
                expected[key].view(torch.uint8),
                path=key,
            )
            assert_accuracy(
                reference[key].cpu().view(torch.uint8),
                expected[key].view(torch.uint8),
                path=key,
            )
