"""Resolve official ops lazily; protect their first aclnn symbol lookup."""

import contextlib
import functools
import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)
_VENDOR_LOCK = threading.RLock()

# Vendors can shadow the libopapi_transformer.so symbols on the first
# call; these ops are resolved once with cust-op vendors hidden.
_CANN_OPAPI_OPS = (
    "mixed_quant_sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
    "kv_compress_epilog",
    "compressor",
    "inplace_partial_rotary_mul",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "indexer_quant_cache",
    "mhc_post",
    "mhc_pre_sinkhorn",
    "sparse_flash_mla_metadata",
    "sparse_flash_mla",
)


@contextlib.contextmanager
def _without_cust_opapi_vendors():
    """Temporarily hide cust-op vendors from the aclnn symbol search."""
    with _VENDOR_LOCK:
        saved_custom = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        saved_ld = os.environ.get("LD_LIBRARY_PATH")
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
        if saved_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                d
                for d in saved_ld.split(os.pathsep)
                if not (d and os.path.exists(os.path.join(d, "libcust_opapi.so")))
            )
        try:
            yield
        finally:
            if saved_custom is not None:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = saved_custom
            if saved_ld is not None:
                os.environ["LD_LIBRARY_PATH"] = saved_ld


class _GuardedOpPacket(torch._ops.OpOverloadPacket):
    """Packet-shaped guard; must not degrade to a bare function for dynamo."""

    def __init__(self, packet):
        self.__dict__.update(packet.__dict__)
        self._sglang_cann_opapi_guard = True
        self._guard_done = False
        # A default-device TorchFunctionMode can redispatch this same packet
        # before its first call returns. Permit that same-thread reentry.
        self._guard_lock = threading.RLock()

    def __call__(self, *args, **kwargs):
        if self._guard_done:
            return super().__call__(*args, **kwargs)
        with self._guard_lock:
            if self._guard_done:
                return super().__call__(*args, **kwargs)
            with _without_cust_opapi_vendors():
                result = super().__call__(*args, **kwargs)
            self._guard_done = True
        return result


def _install_cann_opapi_guards() -> None:
    """Keep CANN cust-op vendors from shadowing the transformer ops."""
    try:
        import cann_ops_transformer  # noqa: F401  (registers the namespace)
    except Exception as e:
        raise RuntimeError(
            "Cannot load cann_ops_transformer. Install a matching CANN extension "
            "or register the legacy custom operators and select the legacy backend."
        ) from e
    namespace = torch.ops.cann_ops_transformer
    for name in _CANN_OPAPI_OPS:
        op = getattr(namespace, name, None)
        if op is None or getattr(op, "_sglang_cann_opapi_guard", False):
            continue
        setattr(namespace, name, _GuardedOpPacket(op))


@functools.lru_cache(maxsize=None)
def get_op(name):
    _install_cann_opapi_guards()
    try:
        return getattr(torch.ops.cann_ops_transformer, name)
    except AttributeError as error:
        raise RuntimeError(f"cann_ops_transformer.{name} is required") from error
