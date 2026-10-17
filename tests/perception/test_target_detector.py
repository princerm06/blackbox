import numpy as np

from blackbox.perception.detector import TargetAwareOpenVocabularyDetector
from blackbox.perception.targets import TargetRegistry


class RecordingZeroShotPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, image, candidate_labels):
        self.calls.append(tuple(candidate_labels))
        return [
            {
                "score": 0.77,
                "label": candidate_labels[0],
                "box": {
                    "xmin": 5,
                    "ymin": 6,
                    "xmax": 30,
                    "ymax": 40,
                },
            }
        ]


def test_target_aware_detector_uses_current_registry() -> None:
    registry = TargetRegistry()
    model = RecordingZeroShotPipeline()
    detector = TargetAwareOpenVocabularyDetector(
        registry=registry,
        pipeline_model=model,
    )
    frame = np.zeros((50, 60, 3), dtype=np.uint8)

    assert detector.detect(frame) == []

    registry.register("My Keys", prompt="keys")
    detections = detector.detect(frame)

    assert model.calls == [("keys",)]
    assert detections[0].class_name == "keys"

    registry.register("Black Wallet", prompt="wallet")
    detector.detect(frame)

    assert model.calls[-1] == ("keys", "wallet")
