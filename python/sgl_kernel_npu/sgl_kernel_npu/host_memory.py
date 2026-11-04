"""MemFabric host-memory device-address registration.

Host pool tensors allocated through ``memfabric_hybrid.offload.empty`` are
not visible to ``offload.get_dva`` until they are registered with
``offload.host_register`` (the acc_offload entry manager maps registered host
virtual addresses to device-visible addresses). The AIV kv_exchange kernel dereferences
those DVAs directly, so every MF-allocated host buffer must go through
:meth:`register_host_memory` at allocation time; interior tensor views then
resolve by byte offset through the mapping table.
"""

from typing import List, NamedTuple, Optional, Tuple

import torch


class HostMemoryMapping(NamedTuple):
    """Device-visible mapping of one registered MemFabric host block."""

    device_ptr: int
    registered_host_ptr: int
    owned: bool


# Registered MF blocks: (base_hva, nbytes, HostMemoryMapping). The table also
# resolves interior tensor views to DVA + byte offset.
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


def _register_host_block(host_ptr: int, nbytes: int) -> HostMemoryMapping:
    """Register one MF-allocated block and return its kernel-visible DVA.

    MemFabric only accepts the base pointer returned by offload.malloc/empty.
    Registered blocks remain alive until uninitialize_host_memory().
    """
    if host_ptr <= 0 or nbytes <= 0:
        raise ValueError(
            f"host_ptr and nbytes must be positive, got {host_ptr}, {nbytes}"
        )

    offload = _load_offload()
    ret = offload.host_register(host_ptr)
    if ret != 0:
        raise RuntimeError(
            f"MF host_register failed for {hex(host_ptr)} "
            f"({nbytes} bytes): ret={ret}; allocate the tensor with "
            "offload.empty before registering it"
        )
    device_ptr = offload.host_get_device_pointer(host_ptr)
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


def lookup_host_dva(tensor: torch.Tensor) -> int:
    """Resolve a registered MF host tensor (or interior view) to its DVA.

    Returns 0 when the tensor does not fall into any registered block; there
    is no fallback because the caller decides how to handle unregistered
    memory (e.g. resolve through offload.get_dva).
    """
    host_ptr = tensor.data_ptr() if tensor.is_cpu else 0
    mapping, reg_ptr = _lookup_mapping(host_ptr)
    if mapping is None:
        return 0
    return int(mapping.device_ptr + (host_ptr - reg_ptr))


def resolve_host_dva(tensor: torch.Tensor) -> int:
    """Resolve a registered MF host tensor (or interior view) to its DVA.

    Unlike lookup_host_dva, unregistered CPU memory is an error: AIV
    kv_exchange kernels can only de-reference host virtual addresses mapped
    through the MemFabric offload entity, so there is no fallback.
    """
    dva = lookup_host_dva(tensor)
    if not dva:
        raise RuntimeError(
            "MF host memory is not registered; call register_host_memory "
            "with an offload.empty tensor before transfer"
        )
    return dva


def uninitialize_host_memory() -> None:
    """Release all MF blocks and invalidate every Python-side DVA mapping."""
    try:
        _load_offload().uninitialize()
    finally:
        _G_MAPPINGS.clear()
