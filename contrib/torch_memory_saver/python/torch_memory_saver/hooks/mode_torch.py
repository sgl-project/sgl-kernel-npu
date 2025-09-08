import logging

from torch.npu.memory import NPUPluggableAllocator
from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package

logger = logging.getLogger(__name__)


class HookUtilModeTorch(HookUtilBase):
    def __init__(self):
        self.allocator = NPUPluggableAllocator(
            self.get_path_binary(), "tms_torch_malloc", "tms_torch_free"
        )
        logger.debug(f"HookUtilModeTorch {self.allocator=} {self.get_path_binary()=}")

    def get_path_binary(self):
        return str(get_binary_path_from_package("torch_memory_saver_hook_mode_torch"))

    def get_allocator(self):
        return self.allocator.allocator()
