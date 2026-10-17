from pathlib import Path

import pytest

from blackbox.app_processing import SessionVideoProcessor
from blackbox.perception.targets import TargetRegistry
from blackbox.schemas.perception import BoundingBox, Detection


class FakeDetector:
    def detect(self, frame):
        return [
            Detection(
                class_name="keys",
                confidence=0.9,
                bbox=BoundingBox(0, 0, 10, 10),
            )
        ]


class RecordingFactory:
    def __init__(self):
        self.prompts = None

    def __call__(self, registry):
        self.prompts = registry.prompts()
        return FakeDetector()


def test_session_processor_requires_enabled_target(tmp_path) -> None:
    registry = TargetRegistry()
    processor = SessionVideoProcessor(detector_factory=RecordingFactory())

    with pytest.raises(ValueError, match="enabled tracking target"):
        processor.process(
            registry,
            tmp_path / "missing.mp4",
            tmp_path / "out",
        )


def test_session_processor_builds_detector_from_registry() -> None:
    registry = TargetRegistry()
    registry.register("My Keys", prompt="keys")
    factory = RecordingFactory()

    processor = SessionVideoProcessor(detector_factory=factory)

    assert factory.prompts is None

    # Detector construction happens before video IO. We only verify the
    # session target contract here; pipeline/video behavior has separate tests.
    with pytest.raises(ValueError):
        processor.process(
            registry,
            Path("does-not-exist.mp4"),
            Path("runs/test"),
        )

    assert factory.prompts == ("keys",)
