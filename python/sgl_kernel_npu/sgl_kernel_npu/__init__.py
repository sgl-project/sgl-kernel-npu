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

import torch
import torch_npu


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()

# Register the bundled CANN 9.1 bridge before the first NPU use. Loading it
# lazily on the first operator call after device initialization can yield
# zero indexer outputs even when matching metadata and ACLNN symbols are present.
if any(
    dependency.get("name") == "sgl_cann_recipes_infer"
    for dependency in globals().get("_manifest", {}).get("third_party", [])
):
    from ._third_party import sgl_cann_recipes_infer  # noqa: F401
