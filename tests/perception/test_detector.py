import numpy as np

from blackbox.perception.detector import EmptyDetector


def test_empty_detector_returns_no_detections() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    detector = EmptyDetector()
    detections = detector.detect(frame)

    assert detections == []