import numpy as np

from dataclasses import dataclass
from models.base import ModelBaseABC


@dataclass
class Zone(ModelBaseABC):
    """A zone is an area which can have multiple fixed fog nodes and mobile fogs moving within."""

    x: float = 0
    y: float = 0
    radius: float = 0  # The coverage radius that can respond to

    def is_in_coverage(self, dst_x: float, dst_y: float) -> bool:
        dst = np.sqrt((self.x - dst_x) ** 2 + (self.y - dst_y) ** 2)
        return dst <= self.radius
