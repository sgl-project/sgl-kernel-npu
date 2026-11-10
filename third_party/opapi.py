"""Keep wheel-private CANN APIs out of bare-name dynamic-library lookup."""

import hashlib
import stat
import struct
from pathlib import Path


def _soname_range(data):
    """Locate DT_SONAME without changing ELF offsets, segments or relocations."""
    if len(data) < 64 or data[:5] != b"\x7fELF\x02" or data[5] not in (1, 2):
        raise RuntimeError("CANN op-api library must be ELF64")
    order = "<" if data[5] == 1 else ">"
    header = struct.unpack_from(order + "HHIQQQIHHHHHH", data, 16)
    phoff, phsize, phnum = header[4], header[8], header[9]
    if phsize < 56 or phoff + phsize * phnum > len(data):
        raise RuntimeError("Invalid CANN ELF program headers")
    loads, dynamic = [], None
    for index in range(phnum):
        kind, _, offset, address, _, size, _, _ = struct.unpack_from(
            order + "IIQQQQQQ", data, phoff + index * phsize
        )
        if offset + size > len(data):
            raise RuntimeError("CANN ELF segment exceeds file size")
        if kind == 1:
            loads.append((address, address + size, offset))
        elif kind == 2:
            dynamic = (offset, size)
    if dynamic is None or dynamic[1] % 16:
        raise RuntimeError("Missing or invalid CANN ELF dynamic section")
    tags, needed = {}, []
    for offset in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, value = struct.unpack_from(order + "qQ", data, offset)
        if tag == 0:
            break
        if tag == 1:
            needed.append(value)
        tags[tag] = value
    if 14 not in tags:  # No DT_SONAME: an absolute-path load has no SONAME alias.
        return None
    if 5 not in tags or 10 not in tags or tags[14] >= tags[10]:
        raise RuntimeError("Invalid CANN ELF dynamic string table")
    if tags[14] in needed:
        raise RuntimeError("CANN ELF SONAME overlaps a DT_NEEDED entry")
    address = tags[5] + tags[14]
    for first, last, offset in loads:
        if first <= address < last:
            start = offset + address - first
            limit = min(offset + last - first, start + tags[10] - tags[14])
            end = data.find(b"\0", start, limit)
            if end < 0:
                raise RuntimeError("Unterminated CANN ELF SONAME")
            return start, end
    raise RuntimeError("CANN ELF SONAME is outside a file-backed LOAD segment")


def isolate_opapi_library(path, owner):
    """Give a staged libcust_opapi.so a private, deterministic SONAME.

    CANN and our adapters still open the unchanged filename by absolute path.
    DeepEP's dlopen("libcust_opapi.so") must not reuse this private library.
    A short alias fits the existing string slot; code and all ELF offsets stay
    unchanged. Only the staged copy is edited, never the dependency build cache.
    """
    path = Path(path)
    data = bytearray(path.read_bytes())
    span = _soname_range(data)
    if span is None:
        return None
    start, end = span
    previous = bytes(data[start:end]).decode("ascii")
    # The alias is an internal loader identity, not a public filename. Detect
    # alias collisions across the complete staged vendor set before packaging.
    name = "sgl_" + hashlib.sha256(owner.encode()).hexdigest()[:8] + ".so"
    if previous == name:
        return name
    if previous != "libcust_opapi.so":
        raise RuntimeError("Unexpected private CANN SONAME: " + previous)
    if len(name) > end - start:
        raise RuntimeError("Private CANN SONAME does not fit its ELF string slot")
    data[start:end] = name.encode().ljust(end - start, b"\0")
    mode = stat.S_IMODE(path.stat().st_mode)
    try:
        path.chmod(mode | stat.S_IWUSR)
        path.write_bytes(data)
    finally:
        path.chmod(mode)
    return name


def isolate_staged_opapis(package):
    """Isolate vendors staged so far before an adapter computes payload hashes.

    Adapters call this after copying their vendor. Rechecking earlier vendors is
    idempotent, so their recorded hashes stay valid. The repository-owned vendor
    is staged before dependencies and is isolated on the first adapter call.
    """
    vendors = Path(package) / "_cann/vendors"
    names = {}
    for library in sorted(vendors.glob("*/op_api/lib/libcust_opapi.so")):
        if vendors.resolve() not in library.resolve().parents:
            raise RuntimeError("CANN op-api library escapes the staged vendor root")
        owner = library.parents[2].name
        name = isolate_opapi_library(library, owner)
        if name is not None and name in names.values():
            raise RuntimeError("Private CANN SONAME collision: " + name)
        names[owner] = name
    return names
