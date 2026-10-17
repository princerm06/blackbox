from dataclasses import asdict, dataclass
import os

import numpy as np
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
from blackbox.perception.identity import ObjectIdentityResolver
from blackbox.perception.tracker import IoUTracker
from blackbox.perception.targets import TargetRegistry
from blackbox.schemas.perception import BoundingBox, Detection
from blackbox.world.projector import WorldStateProjector


DetectorFactory = Callable[[TargetRegistry], Detector]


class LightweightCloudDetector:
    """Low-memory open-vocabulary detector for constrained cloud demos."""

    def __init__(self, registry: TargetRegistry) -> None:
        from ultralytics import YOLOWorld

        prompts = list(registry.prompts())
        if "person" not in prompts:
            prompts.append("person")
        self.prompts = tuple(prompts)
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(list(self.prompts))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=0.20,
            imgsz=480,
        )
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names[class_id])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections


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
    if os.getenv("BLACKBOX_LIGHTWEIGHT_CLOUD", "").strip().lower() in {
        "1", "true", "yes",
    }:
        return LightweightCloudDetector(registry)

    return HybridDetector(
        (
            YoloDetector(allowed_class_names=("person",)),
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
            tracker=IoUTracker(
                iou_threshold=0.15,
                max_missed_frames=5,
            ),
            identity_resolver=ObjectIdentityResolver(
                reid_iou_threshold=0.25,
                max_reid_gap_frames=20,
            ),
            world_state=world_state,
            max_processing_fps=float(
                os.getenv("BLACKBOX_MAX_PROCESSING_FPS", "2.0")
            ),
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
