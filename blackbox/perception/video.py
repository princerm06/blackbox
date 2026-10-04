from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[VideoFrame]:
        frame_index = 0

        while True:
            success, frame = self.capture.read()

            if not success:
                break

            timestamp_ms = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))

            yield VideoFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                image=frame,
            )

            frame_index += 1

    def close(self) -> None:
        self.capture.release()