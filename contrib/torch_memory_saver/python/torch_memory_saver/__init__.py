from .entrypoint import TorchMemorySaver
from .hooks.mode_preload import configure_subprocess

# Global singleton
torch_memory_saver = TorchMemorySaver()
