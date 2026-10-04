from typing import Protocol

import numpy as np

from blackbox.schemas.perception import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class EmptyDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []