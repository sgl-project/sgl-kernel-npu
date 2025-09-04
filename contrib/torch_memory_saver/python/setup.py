
import logging
import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup

logger = logging.getLogger(__name__)

def _find_ascend_home():
    """
    Find the ASCEND toolkit home directory.
    It prioritizes the ASCEND_TOOLKIT_HOME environment variable.
    If not set, it falls back to the common default installation path:
    /usr/local/Ascend/ascend-toolkit/latest
    """
    home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if home:
        return home
    default_home = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.isdir(default_home):
        return default_home
    maybe = "/usr/local/Ascend/ascend-toolkit"
    latest = os.path.join(maybe, "latest")
    return latest if os.path.isdir(latest) else default_home

ascend_home = Path(_find_ascend_home()).resolve()

include_dirs = [
    str((ascend_home / "include").resolve()),
]

library_dirs = [
    str((ascend_home / "lib64").resolve()),
]

logger.warning(f"Using ASCEND_TOOLKIT_HOME at: {ascend_home}")
logger.warning(f"Include dirs: {include_dirs}")
logger.warning(f"Library dirs: {library_dirs}")

extra_compile_args = ["-std=c++17"]

common_macros = [
    ("Py_LIMITED_API", "0x03090000"),
    ("TMS_BACKEND_ASCEND", "1"),
]

repo_root = Path(__file__).resolve().parents[3]  # sgl-kernel-npu/
csrc_dir = repo_root / "contrib" / "torch_memory_saver" / "csrc"
setup(
    name='torch_memory_saver',
    version='0.0.8',
    ext_modules=[
        setuptools.Extension(
            name,
            # sources=[
            #     'csrc/api_forwarder.cpp',
            #     'csrc/core.cpp',
            #     'csrc/entrypoint.cpp',
            # ],
            sources=[
                str(csrc_dir / "api_forwarder.cpp"),
                str(csrc_dir / "core.cpp"),
                str(csrc_dir / "entrypoint.cpp"),
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            #CUDA -> ACL
            libraries=['ascendcl'],
            define_macros=[
                *common_macros,
                *extra_macros,
            ],
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch',   [('TMS_HOOK_MODE_TORCH',   '1')]),
        ]
    ],
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
