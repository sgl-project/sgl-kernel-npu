from abc import ABC
from typing import Literal

HookMode = Literal["preload", "torch"]


class HookUtilBase(ABC):
    @staticmethod
    def create(hook_mode: HookMode) -> "HookUtilBase":
        from torch_memory_saver.hooks.mode_preload import HookUtilModePreload
        from torch_memory_saver.hooks.mode_torch import HookUtilModeTorch
        return {
            "preload": HookUtilModePreload,
            "torch": HookUtilModeTorch,
        }[hook_mode]()

    def get_path_binary(self):
        raise NotImplementedError

    def get_allocator(self):
        return None
