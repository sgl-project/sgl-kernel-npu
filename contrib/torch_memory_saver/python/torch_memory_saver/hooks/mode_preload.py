import logging
import os
from contextlib import contextmanager

import torch
from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package

logger = logging.getLogger(__name__)


class HookUtilModePreload(HookUtilBase):
    def get_path_binary(self):
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        assert "torch_memory_saver" in env_ld_preload, (
            f"TorchMemorySaver observes invalid LD_PRELOAD. "
            f"You can use configure_subprocess() utility, "
            f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
            f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
        )
        return env_ld_preload


@contextmanager
def configure_subprocess():
    """Configure environment variables for subprocesses. Only needed for hook_mode=preload."""
    # Currently, torch_memory_saver does not support preload for npu, therefore LD_PRELOAD interception is not implemented.
    if hasattr(torch, "npu") and torch.npu.is_available():
        yield
    else:
        with _change_env(
            "LD_PRELOAD",
            str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")),
        ):
            yield


@contextmanager
def _change_env(key: str, value: str):
    old_value = os.environ.get(key, "")
    os.environ[key] = value
    logger.debug(f"change_env set key={key} value={value}")
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = old_value
        logger.debug(f"change_env restore key={key} value={old_value}")
