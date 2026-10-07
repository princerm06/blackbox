from dataclasses import asdict, dataclass
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
from blackbox.world.projector import WorldStateProjector


DetectorFactory = Callable[[TargetRegistry], Detector]


@dataclass(frozen=True)
class ProcessedRun:
    summary: ProcessingSummary
    world_state: WorldStateProjector
    source_video_path: Path


class RunStore:
    """Keeps processed run state available to the first app API."""

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], ProcessedRun] = {}

    def put(
        self,
        session_id: str,
        run_id: str,
        run: ProcessedRun,
    ) -> None:
        self._runs[(session_id, run_id)] = run

    def get(self, session_id: str, run_id: str) -> ProcessedRun:
        try:
            return self._runs[(session_id, run_id)]
        except KeyError as error:
            raise KeyError(
                f"unknown run_id for session: {run_id}"
            ) from error


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
    ) -> ProcessedRun:
        if not registry.enabled_targets():
            raise ValueError(
                "at least one enabled tracking target is required"
            )

        detector = self.detector_factory(registry)
        world_state = WorldStateProjector()
        pipeline = VideoDetectionPipeline(
            detector=detector,
            world_state=world_state,
        )
        summary = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            write_annotated_video=True,
        )
        return ProcessedRun(
            summary=summary,
            world_state=world_state,
            source_video_path=input_path,
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
