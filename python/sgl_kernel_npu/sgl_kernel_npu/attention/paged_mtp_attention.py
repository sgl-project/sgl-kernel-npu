# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for SGLang callers of the original attention API."""

from ..operators.attention import paged_mtp_attention_decode

__all__ = ["paged_mtp_attention_decode"]
