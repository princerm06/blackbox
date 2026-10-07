import numpy as np

from blackbox.perception.detector import EmptyDetector, YoloDetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        return list(self.values)


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = FakeTensor([class_id])
        self.conf = FakeTensor([confidence])
        self.xyxy = [FakeTensor(xyxy)]


class FakeResult:
    names = {
        0: "person",
        24: "backpack",
        63: "laptop",
    }

    def __init__(self):
        self.boxes = [
            FakeBox(24, 0.91, [10, 20, 110, 220]),
            FakeBox(63, 0.20, [200, 100, 400, 300]),
        ]


class FakeModel:
    def predict(self, source, verbose=False):
        assert source.shape == (360, 640, 3)
        assert verbose is False
        return [FakeResult()]


def test_empty_detector_returns_no_detections() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    detector = EmptyDetector()
    detections = detector.detect(frame)

    assert detections == []


def test_yolo_detector_converts_and_filters_results() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detector = YoloDetector(
        confidence_threshold=0.30,
        model=FakeModel(),
    )

    detections = detector.detect(frame)

    assert len(detections) == 1

    detection = detections[0]

    assert detection.class_name == "backpack"
    assert detection.confidence == 0.91
    assert detection.bbox.x1 == 10
    assert detection.bbox.y1 == 20
    assert detection.bbox.x2 == 110
    assert detection.bbox.y2 == 220


def test_yolo_detector_rejects_invalid_confidence_threshold() -> None:
    try:
        YoloDetector(
            confidence_threshold=1.5,
            model=FakeModel(),
        )
    except ValueError as error:
        assert "confidence_threshold" in str(error)
    else:
        raise AssertionError("Expected invalid threshold to raise ValueError")
