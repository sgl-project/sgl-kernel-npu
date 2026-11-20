import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

# (layers, slots, payload) exercised by the perf gate.

perf_shape = [(4, 8, (128,))]


@pytest.fixture
def acl_host():
    """Register host pools with the acl backend so the AIV path resolves DVAs."""
    from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
        register_acl_host_memory,
        uninitialize_host_memory,
    )

    def register(tensor):
        register_acl_host_memory(tensor)
        return tensor

    yield register
    torch.npu.synchronize()
    uninitialize_host_memory()


def _indices(values):
    return torch.tensor(values, dtype=torch.int64, device="npu")


@pytest.mark.parametrize(
    "layers,slots,host_slots,payload,dtype,device_ids,host_ids",
    [
        pytest.param(4, 8, 8, (128,), torch.bfloat16, range(8), range(8), id="bf16"),
        pytest.param(
            3, 6, 6, (8, 64), torch.float16, [1, 3, 5], [5, 0, 2], id="subset"
        ),
        pytest.param(
            2,
            10,
            12,
            (32,),
            torch.float32,
            [7, 2, 9],
            [11, 1, 4],
            id="fp32-pools-differ",
        ),
        pytest.param(3, 5, 5, (16,), torch.float32, [2], [3], id="single-slot"),
        # The AIV kernel stages each slot through an 88KB UB buffer and
        # splits larger entries, so cover the boundary from both sides:
        # 45056 * 2B = 88KB exactly, 45057/45055 straddle it by one element.
        pytest.param(
            2,
            4,
            4,
            (45056,),
            torch.float16,
            range(4),
            range(4),
            id="ub-88kb-exact-fp16",
        ),
        pytest.param(
            2, 4, 4, (45057,), torch.float16, [0, 2], [1, 3], id="ub-88kb-over-fp16"
        ),
        pytest.param(
            2,
            4,
            4,
            (45055,),
            torch.float16,
            range(4),
            range(4),
            id="ub-88kb-under-fp16",
        ),
        # Real GDN state shape (~2MB per slot, fp32): several splits per entry.
        pytest.param(
            2,
            4,
            4,
            (32, 128, 128),
            torch.float32,
            range(4),
            range(4),
            id="gdn-shaped-fp32",
        ),
    ],
)
def test_state_exchange_all_layer_d2h(
    acl_host, layers, slots, host_slots, payload, dtype, device_ids, host_ids
):
    from sgl_kernel_npu.operators.data_move import transfer_state_all_layer_direct_lf_pf

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_state_all_layer_direct_lf_pf, impl, impl_ref)
    torch.manual_seed(0)
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(
        torch.randn(host_slots, layers, 1, *payload, dtype=dtype, device="cpu")
    )
    d, h = _indices(device_ids), _indices(host_ids)

    ref_dev = dev.clone()
    ref_host = host.clone()
    if checked:
        transfer_state_all_layer_direct_lf_pf(
            [ref_dev], [ref_host], d, h, impl=impl_ref
        )
    host.zero_()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert (
            transfer_state_all_layer_direct_lf_pf([dev], [host], d, h, impl=impl)
            is None
        )
    stream.synchronize()

    if checked:
        expected = torch.zeros_like(ref_host)
        for device_slot, host_slot in zip(device_ids, host_ids):
            expected[host_slot, :, 0] = ref_dev[:, device_slot].transpose(0, 1)
        assert_accuracy(host.view(torch.uint8), expected.view(torch.uint8))
    executor = make_executor(
        transfer_state_all_layer_direct_lf_pf,
        args=([dev], [host], d, h),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=(layers, slots, payload) in perf_shape)


def test_state_exchange_all_layer_draft_mixed(acl_host):
    # Main layers live in one layer-first tensor while each draft layer is its
    # own single-layer allocation; the host pool is narrowed per component so
    # one submission covers independent layer counts.

    from sgl_kernel_npu.operators.data_move import transfer_state_all_layer_direct_lf_pf

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_state_all_layer_direct_lf_pf, impl, impl_ref)
    layers_main, drafts, slots, host_slots, payload = 3, 2, 8, 9, (48,)
    total_layers = layers_main + drafts
    torch.manual_seed(0)
    dev_main = torch.randn(
        layers_main, slots, *payload, dtype=torch.float32, device="npu"
    )
    dev_drafts = [
        torch.randn(1, slots, *payload, dtype=torch.float32, device="npu")
        for _ in range(drafts)
    ]
    host = acl_host(
        torch.randn(
            host_slots, total_layers, 1, *payload, dtype=torch.float32, device="cpu"
        )
    )
    d, h = _indices([4, 1, 6]), _indices([7, 0, 3])
    device_states = [dev_main, *dev_drafts]
    host_states = [host.narrow(1, 0, layers_main)] + [
        host.narrow(1, layers_main + i, 1) for i in range(drafts)
    ]

    ref_device_states = [dev.clone() for dev in device_states]
    ref_host = host.clone()
    if checked:
        ref_host_states = [ref_host.narrow(1, 0, layers_main)] + [
            ref_host.narrow(1, layers_main + i, 1) for i in range(drafts)
        ]
        transfer_state_all_layer_direct_lf_pf(
            ref_device_states, ref_host_states, d, h, impl=impl_ref
        )
    host.zero_()
    transfer_state_all_layer_direct_lf_pf(device_states, host_states, d, h, impl=impl)
    torch.npu.synchronize()

    if checked:
        expected = torch.zeros_like(ref_host)
        for device_slot, host_slot in zip(d.tolist(), h.tolist()):
            expected[host_slot, :layers_main, 0] = ref_device_states[0][
                :, device_slot
            ].transpose(0, 1)
            for i, draft in enumerate(ref_device_states[1:]):
                expected[host_slot, layers_main + i, 0] = draft[0, device_slot]
        assert_accuracy(host.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "layers,slots,host_slots,payload,dtype,layer_id,host_ids,device_ids",
    [
        pytest.param(4, 8, 8, (128,), torch.bfloat16, 2, range(8), range(8), id="bf16"),
        pytest.param(3, 6, 6, (8, 64), torch.float16, 0, [2, 5], [1, 4], id="subset"),
        pytest.param(
            2, 10, 12, (32,), torch.float32, 1, [9, 3], [5, 0], id="fp32-pools-differ"
        ),
        # UB-staging boundary (88KB per slot) in both directions, plus a
        # multi-split GDN-shaped payload.
        pytest.param(
            2,
            4,
            4,
            (45056,),
            torch.float16,
            1,
            range(4),
            range(4),
            id="ub-88kb-exact-fp16",
        ),
        pytest.param(
            2, 4, 4, (45057,), torch.float16, 0, [1, 3], [0, 2], id="ub-88kb-over-fp16"
        ),
        pytest.param(
            2,
            4,
            4,
            (32, 128, 128),
            torch.float32,
            1,
            range(4),
            range(4),
            id="gdn-shaped-fp32",
        ),
    ],
)
def test_state_exchange_per_layer_h2d(
    acl_host, layers, slots, host_slots, payload, dtype, layer_id, host_ids, device_ids
):
    from sgl_kernel_npu.operators.data_move import transfer_state_per_layer_direct_pf_lf

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(transfer_state_per_layer_direct_pf_lf, impl, impl_ref)
    torch.manual_seed(0)
    dev = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(
        torch.randn(host_slots, layers, 1, *payload, dtype=dtype, device="cpu")
    )
    h, d = _indices(host_ids), _indices(device_ids)

    ref_dev = torch.zeros_like(dev)
    ref_host = host.clone()
    if checked:
        transfer_state_per_layer_direct_pf_lf(
            ref_host, ref_dev[layer_id], h, d, layer_id, impl=impl_ref
        )
    dev.zero_()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert (
            transfer_state_per_layer_direct_pf_lf(
                host, dev[layer_id], h, d, layer_id, impl=impl
            )
            is None
        )
    stream.synchronize()

    if checked:
        assert_accuracy(dev.view(torch.uint8), ref_dev.view(torch.uint8))


def test_state_exchange_roundtrip(acl_host):
    # Backup every layer D2H, restore them per-layer H2D, and compare.

    from sgl_kernel_npu.operators.data_move import (
        transfer_state_all_layer_direct_lf_pf,
        transfer_state_per_layer_direct_pf_lf,
    )

    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    layers, slots, payload, dtype = 4, 8, (64,), torch.float16
    torch.manual_seed(0)
    original = torch.randn(layers, slots, *payload, dtype=dtype, device="npu")
    host = acl_host(torch.zeros(slots, layers, 1, *payload, dtype=dtype, device="cpu"))
    ids = _indices(range(slots))

    transfer_state_all_layer_direct_lf_pf([original], [host], ids, ids, impl=impl)
    restored = torch.zeros_like(original)
    for layer_id in range(layers):
        transfer_state_per_layer_direct_pf_lf(
            host, restored[layer_id], ids, ids, layer_id, impl=impl
        )
    torch.npu.synchronize()

    if checked:
        assert_accuracy(restored.view(torch.uint8), original.view(torch.uint8))


def test_state_exchange_empty_indices_noop(acl_host):
    # Empty indices return before any submission; buffers stay untouched.

    from sgl_kernel_npu.operators.data_move import (
        transfer_state_all_layer_direct_lf_pf,
        transfer_state_per_layer_direct_pf_lf,
    )

    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = acl_host(torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu"))
    before_host, before_dev = host.clone(), dev.clone()
    empty = _indices([])
    transfer_state_all_layer_direct_lf_pf(
        [dev], [host], empty, empty, impl="ascendc_aiv"
    )
    transfer_state_per_layer_direct_pf_lf(
        host, dev[0], empty, empty, 0, impl="ascendc_aiv"
    )
    assert_accuracy(host.view(torch.uint8), before_host.view(torch.uint8))
    assert_accuracy(dev.view(torch.uint8), before_dev.view(torch.uint8))


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,error_type,message",
    [
        ("unregistered", RuntimeError, "not registered"),
        ("too-many", ValueError, "at most 16 components"),
        ("dtype", RuntimeError, "dtypes must match"),
        ("host-layout", RuntimeError, "buffers must be device \\(layers, slots"),
        ("device-layout", RuntimeError, "buffers must be device \\(layers, slots"),
    ],
)
def test_state_exchange_all_layer_rejected(acl_host, case, error_type, message):
    from sgl_kernel_npu.operators.data_move import transfer_state_all_layer_direct_lf_pf

    torch.manual_seed(0)
    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu")
    if case != "unregistered":
        host = acl_host(host)
    if case == "too-many":
        dev_states = [dev] * 17
        host_states = [host] * 17
    else:
        dev_states, host_states = [dev], [host]
        if case == "dtype":
            host_states = [acl_host(host.to(torch.bfloat16))]
        elif case == "host-layout":
            host_states = [host.squeeze(2)]
        elif case == "device-layout":
            dev_states = [dev[0]]
    d, h = _indices([1]), _indices([2])
    with pytest.raises(error_type, match=message):
        transfer_state_all_layer_direct_lf_pf(
            dev_states, host_states, d, h, impl="ascendc_aiv"
        )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,error_type,message",
    [
        ("unregistered", RuntimeError, "not registered"),
        ("layer-id", ValueError, "layer_id is outside"),
        ("host-layout", RuntimeError, "buffers must be device \\(layers, slots"),
    ],
)
def test_state_exchange_per_layer_rejected(acl_host, case, error_type, message):
    from sgl_kernel_npu.operators.data_move import transfer_state_per_layer_direct_pf_lf

    torch.manual_seed(0)
    dev = torch.randn(2, 4, 16, dtype=torch.float32, device="npu")
    host = torch.randn(4, 2, 1, 16, dtype=torch.float32, device="cpu")
    if case != "unregistered":
        host = acl_host(host)
    layer_id = 0
    dst = dev[0]
    if case == "layer-id":
        layer_id = 2
    elif case == "host-layout":
        host = host.squeeze(2)
    h, d = _indices([2]), _indices([1])
    with pytest.raises(error_type, match=message):
        transfer_state_per_layer_direct_pf_lf(
            host, dst, h, d, layer_id, impl="ascendc_aiv"
        )
