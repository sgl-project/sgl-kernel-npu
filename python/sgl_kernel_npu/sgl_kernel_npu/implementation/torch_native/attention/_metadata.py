"""Reference-only metadata descriptors; not CANN tiling bytes."""

from typing import NamedTuple


class ReferenceMetadata(NamedTuple):
    parameters: dict

    def to(self, device):
        return self
