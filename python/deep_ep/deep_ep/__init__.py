import os

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
opp_path = os.path.join(current_dir, "vendors", "hwcomputing")
lib_path = os.path.join(current_dir, "vendors", "hwcomputing", "op_api", "lib")
# Set environment variables related to custom operators
os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
    f"{opp_path}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
)
os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from deep_ep_cpp import Config

# Import strategies to register them
from . import strategies
from .buffer import Buffer
from .ep_strategy import LowLatencyStrategy, NormalStrategy
from .utils import EventOverlap
