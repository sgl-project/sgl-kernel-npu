import json
import os
import pathlib
from functools import lru_cache, wraps

if (pathlib.Path(__file__).parent / "_cann").is_dir():
    # Enable the bundled CANN operators before initializing torch_npu.
    _cann_vendor = (
        pathlib.Path(__file__).resolve().parent / "_cann/vendors/sgl_kernel_npu"
    )
    _manifest_path = pathlib.Path(__file__).resolve().parent / "_build_manifest.json"
    _manifest = (
        json.loads(_manifest_path.read_text()) if _manifest_path.is_file() else {}
    )
    _vendors = []
    # CPP-only wheels may bundle third-party CANN vendors without a repository-
    # owned vendor. Older manifests retain their original required-vendor check.
    if _manifest.get("operators_by_type", {}).get("cann", True):
        _cann_library = _cann_vendor / "op_api/lib/libcust_opapi.so"
        if not _cann_library.is_file():
            raise RuntimeError(
                "The wheel is missing its CANN library: {}".format(_cann_library)
            )
        _vendors.append(str(_cann_vendor))
    for _dependency in _manifest.get("third_party", []):
        _name = _dependency["vendor_directory"]
        if pathlib.Path(_name).name != _name or _name in (".", ".."):
            raise RuntimeError("Invalid bundled vendor name: " + _name)
        _vendor = _cann_vendor.parent / _name
        if not (_vendor / "op_api/lib/libcust_opapi.so").is_file():
            raise RuntimeError("Missing bundled third-party vendor: " + str(_vendor))
        _vendors.append(str(_vendor))
    _opp_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        list(dict.fromkeys(_vendors + [path for path in _opp_paths if path]))
    )

# Activate the pinned A5 LIM OPP after the repository-owned CANN vendors are
# staged, but before torch_npu captures custom-OPP discovery. Activation is a
# no-op for A2/A3 wheels, which do not contain this bundle.
from ._original_opp import activate, verify_library  # isort: skip

_original_opp_manifest = activate(pathlib.Path(__file__).resolve().parent)

import torch
import torch_npu


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()

if hasattr(torch.ops.npu, "a5_c8_backend"):
    _a5_c8_backend = torch.ops.npu.a5_c8_backend()
    if _a5_c8_backend == "original":
        if _original_opp_manifest is None:
            raise ImportError(
                "the original A5 C8 backend requires its wheel-bundled OPP"
            )
        for _symbol, _library in dict(
            torch.ops.npu.resolve_a5_c8_original_libraries()
        ).items():
            verify_library(
                pathlib.Path(__file__).resolve().parent,
                _original_opp_manifest,
                _library,
            )
