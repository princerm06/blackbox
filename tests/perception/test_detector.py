import numpy as np

from blackbox.perception.detector import (
    EmptyDetector,
    HybridDetector,
    OpenVocabularyDetector,
    YoloDetector,
)


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



class FakeZeroShotPipeline:
    def __call__(self, image, candidate_labels):
        assert image.size == (640, 360)
        assert candidate_labels == ["keys", "wallet"]
        return [
            {
                "score": 0.82,
                "label": "keys",
                "box": {
                    "xmin": 12,
                    "ymin": 18,
                    "xmax": 62,
                    "ymax": 55,
                },
            },
            {
                "score": 0.10,
                "label": "wallet",
                "box": {
                    "xmin": 100,
                    "ymin": 100,
                    "xmax": 160,
                    "ymax": 150,
                },
            },
        ]


def test_open_vocabulary_detector_uses_prompted_labels() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detector = OpenVocabularyDetector(
        prompts=("keys", "wallet"),
        confidence_threshold=0.25,
        pipeline_model=FakeZeroShotPipeline(),
    )

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "keys"
    assert detections[0].confidence == 0.82
    assert detections[0].bbox.x1 == 12
    assert detections[0].bbox.y2 == 55


def test_open_vocabulary_detector_rejects_empty_prompts() -> None:
    try:
        OpenVocabularyDetector(
            prompts=(),
            pipeline_model=FakeZeroShotPipeline(),
        )
    except ValueError as error:
        assert "prompts" in str(error)
    else:
        raise AssertionError("Expected empty prompts to raise ValueError")


class StaticDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def test_hybrid_detector_combines_detector_outputs() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    first = StaticDetector(
        [
            type(
                "D",
                (),
                {"class_name": "backpack"},
            )()
        ]
    )
    second = StaticDetector(
        [
            type(
                "D",
                (),
                {"class_name": "keys"},
            )()
        ]
    )

    detections = HybridDetector((first, second)).detect(frame)

    assert [detection.class_name for detection in detections] == [
        "backpack",
        "keys",
    ]
