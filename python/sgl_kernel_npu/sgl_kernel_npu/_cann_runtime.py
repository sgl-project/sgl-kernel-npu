"""Enable the vendor shipped beside this package before loading its Torch library."""

import os
from pathlib import Path


def prepare():
    vendor = Path(__file__).resolve().parent / "_cann/vendors/sgl_kernel_npu"
    library = vendor / "op_api/lib/libcust_opapi.so"
    if not library.is_file():
        raise RuntimeError("The wheel is missing its CANN library: {}".format(library))
    paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        [str(vendor)] + [path for path in paths if path and path != str(vendor)])
    return vendor
