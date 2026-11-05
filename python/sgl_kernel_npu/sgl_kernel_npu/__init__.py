import json
import os
import pathlib
from functools import lru_cache, wraps

if (pathlib.Path(__file__).parent / "_cann").is_dir():
    # Enable the bundled CANN operators before initializing torch_npu.
    _cann_vendor = (
        pathlib.Path(__file__).resolve().parent / "_cann/vendors/sgl_kernel_npu"
    )
    _cann_library = _cann_vendor / "op_api/lib/libcust_opapi.so"
    if not _cann_library.is_file():
        raise RuntimeError(
            "The wheel is missing its CANN library: {}".format(_cann_library)
        )
    _vendors = [str(_cann_vendor)]
    _manifest_path = pathlib.Path(__file__).resolve().parent / "_build_manifest.json"
    if _manifest_path.is_file():
        for _dependency in json.loads(_manifest_path.read_text()).get(
            "third_party", []
        ):
            if _dependency.get("binding") == "precompiled_native":
                _package_name = _dependency["python_package"]
                if pathlib.Path(
                    _package_name
                ).name != _package_name or _package_name in (".", ".."):
                    raise RuntimeError(
                        "Invalid bundled native package: " + _package_name
                    )
                _native_library = (
                    pathlib.Path(__file__).resolve().parent
                    / "_third_party"
                    / _package_name
                    / "lib/libsgl_native_compressor.so"
                )
                if not _native_library.is_file():
                    raise RuntimeError(
                        "Missing bundled native compressor: " + str(_native_library)
                    )
                continue
            _name = _dependency["vendor_directory"]
            if pathlib.Path(_name).name != _name or _name in (".", ".."):
                raise RuntimeError("Invalid bundled vendor name: " + _name)
            _vendor = _cann_vendor.parent / _name
            if not (_vendor / "op_api/lib/libcust_opapi.so").is_file():
                raise RuntimeError(
                    "Missing bundled third-party vendor: " + str(_vendor)
                )
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
