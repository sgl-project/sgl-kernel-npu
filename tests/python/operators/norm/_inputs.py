"""Shared random seed setup for public norm operator tests."""

import torch


def seed_inputs(seed=42):
    torch.npu.manual_seed(seed)
