import os
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
src = Path(os.environ["VLLM_ASCEND_SOURCE"]) / "csrc"
torch_npu_root = Path(torch_npu.__file__).resolve().parent
build = Path(os.environ["ORIGINAL_BINDING_BUILD"])
build.mkdir(parents=True, exist_ok=True)
load(
    name="original_lim_binding",
    sources=[
        str(root / "original_binding.cpp"),
        str(src / "aclnn_torch_adapter/NPUBridge.cpp"),
        str(src / "aclnn_torch_adapter/NPUStorageImpl.cpp"),
    ],
    extra_include_paths=[
        str(src),
        str(torch_npu_root.parent),
        str(torch_npu_root / "include"),
        str(Path(os.environ["ASCEND_HOME_PATH"]) / "include"),
    ],
    extra_ldflags=[
        "-L" + str(torch_npu_root / "lib"),
        "-ltorch_npu",
        "-L" + str(Path(os.environ["ASCEND_HOME_PATH"]) / "lib64"),
        "-lascendcl",
    ],
    build_directory=str(build),
    is_python_module=False,
    verbose=True,
)
print("BINDING_BUILD_OK", flush=True)
