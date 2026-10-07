from dataclasses import asdict
from pathlib import Path
from typing import Callable

from blackbox.perception.detector import (
    Detector,
    HybridDetector,
    TargetAwareOpenVocabularyDetector,
    YoloDetector,
)
from blackbox.perception.pipeline import (
    ProcessingSummary,
    VideoDetectionPipeline,
)
from blackbox.perception.targets import TargetRegistry


DetectorFactory = Callable[[TargetRegistry], Detector]


def default_detector_factory(registry: TargetRegistry) -> Detector:
    return HybridDetector(
        (
            YoloDetector(),
            TargetAwareOpenVocabularyDetector(registry=registry),
        )
    )


class SessionVideoProcessor:
    """Runs a session's registered targets through the perception pipeline."""

    def __init__(
        self,
        detector_factory: DetectorFactory = default_detector_factory,
    ) -> None:
        self.detector_factory = detector_factory

    def process(
        self,
        registry: TargetRegistry,
        input_path: Path,
        output_dir: Path,
    ) -> ProcessingSummary:
        if not registry.enabled_targets():
            raise ValueError(
                "at least one enabled tracking target is required"
            )

        detector = self.detector_factory(registry)
        pipeline = VideoDetectionPipeline(detector=detector)
        return pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            write_annotated_video=True,
        )


def processing_summary_to_dict(
    summary: ProcessingSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["observations_path"] = str(summary.observations_path)
    payload["annotated_video_path"] = (
        str(summary.annotated_video_path)
        if summary.annotated_video_path is not None
        else None
    )
    return payload
