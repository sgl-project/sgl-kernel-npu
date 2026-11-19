"""Registered host-memory device-address table for the AIV kv_exchange kernel.

Two registration providers map host DRAM into the device VA space, selected
by SGLANG_HICACHE_HOST_MEM_BACKEND in the SGLang process:

- ``memfabric``: an ``memfabric_hybrid.offload.Offload`` instance owns the
  host pool; blocks allocated with ``pool.empty`` are registered with
  ``pool.host_register`` (publish the instance once via
  :func:`set_memfabric_pool` after ``pool.initialize``); every MF-allocated
  host buffer must go through :meth:`register_host_memory` at allocation
  time.
- ``acl``: plain pageable host tensors (torch.empty) are registered with
  the standard CANN runtime ``aclrtHostRegister(ACL_HOST_REGISTER_MAPPED)``,
  which returns the NPU-accessible device pointer directly; see
  :meth:`register_acl_host_memory`.

Both providers record into the same table so the AIV kv_exchange kernel
resolves interior tensor views of either kind of block by byte offset.
"""

import logging
from typing import List, NamedTuple, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class HostMemoryMapping(NamedTuple):
    """Device-visible mapping of one registered host block."""

    device_ptr: int
    registered_host_ptr: int
    owned: bool
    # "memfabric" (offload.host_register) or "acl" (aclrtHostRegister).
    kind: str = "memfabric"


# Registered blocks: (base_hva, nbytes, HostMemoryMapping). The table also
# resolves interior tensor views to device pointer + byte offset.
_G_MAPPINGS: List[Tuple[int, int, HostMemoryMapping]] = []


def _lookup_mapping(host_ptr: int) -> Tuple[Optional[HostMemoryMapping], int]:
    if host_ptr:
        for reg_ptr, nbytes, mapping in reversed(_G_MAPPINGS):
            if reg_ptr <= host_ptr < reg_ptr + nbytes:
                return mapping, reg_ptr
    return None, 0


def _load_offload():
    from memfabric_hybrid import offload

    return offload


# The Offload instance that owns the registered host pool. The memfabric
# backend's register/get-device-pointer calls live on the instance, so the
# pool owner (SGLang's MemfabricHostMemoryBackend) publishes it here once
# after initialize; None falls back to the legacy module-level API.
_MF_POOL = None


def set_memfabric_pool(pool) -> None:
    """Publish the memfabric Offload instance that owns the host pool."""
    global _MF_POOL
    _MF_POOL = pool


def get_memfabric_pool():
    """Return the initialized Offload instance published by the host pool."""
    return _MF_POOL


def _register_mf_block(host_ptr: int) -> int:
    """Register one MF-allocated block; return 0 on success.

    Prefers the published Offload instance (release/1.3 instance API,
    idempotent on the pre-registered pool); falls back to the legacy
    module-level ``offload.host_register`` when no pool was published.
    """
    if _MF_POOL is not None:
        return int(_MF_POOL.host_register(host_ptr))
    offload = _load_offload()
    legacy = getattr(offload, "host_register", None)
    if legacy is None:
        raise RuntimeError(
            "memfabric host_register is only available on an Offload "
            "instance in this build; call set_memfabric_pool(pool) after "
            "pool.initialize first"
        )
    return int(legacy(host_ptr))


def _mf_device_pointer(host_ptr: int) -> int:
    """Return the device pointer of an MF host block, 0 on failure."""
    if _MF_POOL is not None:
        return int(_MF_POOL.host_get_device_pointer(host_ptr))
    offload = _load_offload()
    legacy = getattr(offload, "host_get_device_pointer", None)
    if legacy is None:
        return 0
    return int(legacy(host_ptr))


def _uninitialize_memfabric() -> None:
    # Verified against release/1.3: the binding still exports the module-level
    # ``offload.uninitialize`` (process-global entry manager), so this
    # fallback is valid for legacy builds that register blocks through the
    # module-level API.  On release/1.3 it is unreachable, because
    # ``host_register`` only exists on an Offload instance, so MF mappings
    # cannot exist without a published pool.
    if _MF_POOL is not None:
        _MF_POOL.uninitialize()
        return
    _load_offload().uninitialize()


def _register_host_block(host_ptr: int, nbytes: int) -> HostMemoryMapping:
    """Register one MF-allocated block and return its kernel-visible DVA.

    MemFabric only accepts the base pointer returned by pool.malloc/empty.
    Registered blocks remain alive until uninitialize_host_memory().
    """
    if host_ptr <= 0 or nbytes <= 0:
        raise ValueError(
            f"host_ptr and nbytes must be positive, got {host_ptr}, {nbytes}"
        )

    ret = _register_mf_block(host_ptr)
    if ret != 0:
        raise RuntimeError(
            f"MF host_register failed for {hex(host_ptr)} "
            f"({nbytes} bytes): ret={ret}; allocate the tensor with "
            "pool.empty before registering it"
        )
    device_ptr = _mf_device_pointer(host_ptr)
    if not device_ptr:
        raise RuntimeError(
            f"MF host_get_device_pointer failed for registered host block {hex(host_ptr)}"
        )
    return HostMemoryMapping(
        device_ptr=int(device_ptr), registered_host_ptr=int(host_ptr), owned=True
    )


def register_host_memory(tensor: torch.Tensor) -> HostMemoryMapping:
    """Register an MF-allocated CPU tensor and record its DVA mapping."""
    if not tensor.is_cpu:
        raise RuntimeError("register_host_memory: tensor must be host (CPU) memory")
    host_ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes <= 0:
        raise RuntimeError("register_host_memory: tensor must be non-empty")

    mapping = _register_host_block(host_ptr, nbytes)
    _G_MAPPINGS.append((host_ptr, nbytes, mapping))
    return mapping


# ---------------------------------------------------------------------------
# aclrt registration (SGLANG_HICACHE_HOST_MEM_BACKEND=acl)
#
# Mirrors the aclrt_aiv branch's host_memory.py (sgl-project/sglang PR
# #39548 style) call-for-call: the aclrt calls go through the ``acl.rt``
# python binding exactly as that branch does -- never ctypes or
# torch.ops.npu.  Key binding contract:
#   * ``acl_rt.host_register(host_ptr, nbytes, ACL_HOST_REGISTER_MAPPED)``
#     returns ``(dev_ptr, ret)``, NOT (ret, dev_ptr);
#   * ``acl_rt.host_get_device_pointer(host_ptr, 0)`` likewise returns
#     ``(dev_ptr, ret)`` -- borrow existing mappings (e.g. torch_npu's
#     pinned-memory allocator) instead of re-registering;
#   * registration is page-aligned (mmap.PAGESIZE); interior offsets are
#     added back to the returned device pointer;
#   * failure is soft (returns None); the caller decides whether a missing
#     mapping is fatal.  aclrtHostRegister maps an existing host range into
#     the device VA space, so acl-registered blocks coexist with MF blocks
#     in the same table and the AIV kv_exchange kernel dereferences either.
# ---------------------------------------------------------------------------

import mmap

# aclrtHostRegisterType: map the host range into the device address space.
ACL_HOST_REGISTER_MAPPED = 0


def _load_acl_rt():
    import acl.rt as acl_rt

    return acl_rt


def _aligned_range(host_ptr: int, nbytes: int) -> Tuple[int, int, int]:
    """Return page-aligned ``(base, size, offset)`` for a host range."""
    page_size = mmap.PAGESIZE
    base = host_ptr - host_ptr % page_size
    offset = host_ptr - base
    size = ((offset + nbytes + page_size - 1) // page_size) * page_size
    return base, size, offset


def _get_device_pointer(acl_rt, host_ptr: int) -> Optional[int]:
    """Return an existing device mapping without taking ownership of it."""
    getter = getattr(acl_rt, "host_get_device_pointer", None)
    if getter is None:
        return None
    try:
        # NOTE: the binding returns (dev_ptr, ret), not (ret, dev_ptr).
        dev_ptr, ret = getter(host_ptr, 0)
    except Exception:  # an unregistered range is an expected miss
        return None
    if ret == 0 and dev_ptr:
        return int(dev_ptr)
    return None


def _register_acl_host_block(host_ptr: int, nbytes: int) -> Optional[HostMemoryMapping]:
    """Map ``[host_ptr, host_ptr + nbytes)`` into the device address space.

    Existing mappings (including mappings owned by torch_npu's pinned-memory
    allocator) are borrowed and must not be unregistered here.  New mappings
    are page-aligned to satisfy ``aclrtHostRegister`` and marked as owned.
    Returns None on failure; the caller decides whether that is fatal.
    """
    if host_ptr <= 0 or nbytes <= 0:
        return None
    registered_host_ptr, registered_nbytes, offset = _aligned_range(host_ptr, nbytes)
    try:
        acl_rt = _load_acl_rt()

        existing_ptr = _get_device_pointer(acl_rt, registered_host_ptr)
        if existing_ptr is not None:
            logger.info(
                "Using an existing host-memory device mapping: host=%#x -> device=%#x",
                registered_host_ptr,
                existing_ptr,
            )
            return HostMemoryMapping(
                device_ptr=existing_ptr + offset,
                registered_host_ptr=registered_host_ptr,
                owned=False,
                kind="acl",
            )

        # NOTE: the binding returns (dev_ptr, ret), not (ret, dev_ptr).
        dev_ptr, ret = acl_rt.host_register(
            registered_host_ptr, registered_nbytes, ACL_HOST_REGISTER_MAPPED
        )
    except Exception as exc:  # depends on the CANN install
        logger.warning("aclrtHostRegister unavailable: %s", exc)
        return None
    if ret != 0 or not dev_ptr:
        # Another component may have registered the range between the lookup
        # and the register call. Query once more and borrow that mapping if
        # present.
        existing_ptr = _get_device_pointer(acl_rt, registered_host_ptr)
        if existing_ptr is not None:
            return HostMemoryMapping(
                device_ptr=existing_ptr + offset,
                registered_host_ptr=registered_host_ptr,
                owned=False,
                kind="acl",
            )
        logger.warning(
            "aclrtHostRegister failed for %#x (%d bytes): ret=%s",
            registered_host_ptr,
            registered_nbytes,
            ret,
        )
        return None
    logger.info(
        "Mapped %d bytes of host memory for device access: host=%#x -> device=%#x",
        registered_nbytes,
        registered_host_ptr,
        dev_ptr,
    )
    return HostMemoryMapping(
        device_ptr=int(dev_ptr) + offset,
        registered_host_ptr=registered_host_ptr,
        owned=True,
        kind="acl",
    )


def register_acl_host_memory(tensor: torch.Tensor) -> HostMemoryMapping:
    """Register a plain CPU tensor via aclrtHostRegister (acl backend).

    Unlike the memfabric provider this accepts any host tensor (e.g. a
    pageable torch.empty block allocated by alloc_with_acl in SGLang); the
    runtime pins and maps the range, and the AIV kv_exchange kernel
    dereferences the returned device pointer directly.  A failed mapping is
    fatal here: the host pool would otherwise look registered while AIV
    transfers read stale data.
    """
    if not tensor.is_cpu:
        raise RuntimeError("register_acl_host_memory: tensor must be host (CPU) memory")
    host_ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes <= 0:
        raise RuntimeError("register_acl_host_memory: tensor must be non-empty")

    mapping = _register_acl_host_block(host_ptr, nbytes)
    if mapping is None:
        raise RuntimeError(
            f"aclrtHostRegister failed for {hex(host_ptr)} ({nbytes} bytes); "
            "host memory is not device-mapped and transfers would read "
            "stale data"
        )
    # Interior views resolve against the tensor base pointer through the
    # table, so store the tensor range, not the page-aligned range.
    _G_MAPPINGS.append((host_ptr, nbytes, mapping))
    return mapping


def lookup_host_dva(tensor: torch.Tensor) -> int:
    """Resolve a registered MF host tensor (or interior view) to its DVA.

    Returns 0 when the tensor does not fall into any registered block; there
    is no fallback because the caller decides how to handle unregistered
    memory (e.g. resolve through ``get_dva`` on the Offload instance).
    """
    host_ptr = tensor.data_ptr() if tensor.is_cpu else 0
    mapping, reg_ptr = _lookup_mapping(host_ptr)
    if mapping is None:
        return 0
    return int(mapping.device_ptr + (host_ptr - reg_ptr))


def resolve_host_dva(tensor: torch.Tensor) -> int:
    """Resolve a registered host tensor (or interior view) to its DVA.

    Unlike lookup_host_dva, unregistered CPU memory is an error: AIV
    kv_exchange kernels can only de-reference host virtual addresses mapped
    into the device VA space (MemFabric or aclrt registration), so there is
    no fallback.
    """
    dva = lookup_host_dva(tensor)
    if not dva:
        raise RuntimeError(
            "Host memory is not registered; call register_host_memory "
            "(MemFabric pool.empty) or register_acl_host_memory "
            "(aclrt) with the host tensor before transfer"
        )
    return dva


def uninitialize_host_memory() -> None:
    """Release every owned mapping and invalidate the mapping table."""
    global _MF_POOL
    acl_mappings = [mapping for _, _, mapping in _G_MAPPINGS if mapping.kind == "acl"]
    if acl_mappings:
        try:
            acl_rt = _load_acl_rt()
        except Exception as exc:  # depends on the CANN install
            logger.warning("aclrtHostUnregister unavailable: %s", exc)
            acl_rt = None
        if acl_rt is not None:
            for mapping in acl_mappings:
                # Borrowed mappings (e.g. torch_npu pinned-memory allocator
                # registrations) are left untouched; only owned ones go.
                if not mapping.owned:
                    continue
                try:
                    ret = acl_rt.host_unregister(mapping.registered_host_ptr)
                except Exception as exc:  # depends on the CANN install
                    logger.warning("aclrtHostUnregister unavailable: %s", exc)
                    continue
                if ret != 0:
                    logger.warning(
                        "aclrtHostUnregister failed for %#x: ret=%s",
                        mapping.registered_host_ptr,
                        ret,
                    )
    has_mf = any(mapping.kind == "memfabric" for _, _, mapping in _G_MAPPINGS)
    try:
        if has_mf:
            _uninitialize_memfabric()
    finally:
        _G_MAPPINGS.clear()
        # Drop the published pool so a later register/lookup cannot reach a
        # stale, uninitialized Offload instance.
        _MF_POOL = None
