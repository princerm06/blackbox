import os
from typing import Any, Protocol

import numpy as np

from blackbox.schemas.perception import BoundingBox, Detection
from blackbox.perception.targets import TargetRegistry



def _maybe_quantize_pipeline(pipeline_model: Any) -> Any:
    """Optionally quantize zero-shot detector weights for low-memory CPU deploys."""
    if os.getenv("BLACKBOX_QUANTIZE_OPEN_VOCAB", "").strip().lower() not in {
        "1", "true", "yes",
    }:
        return pipeline_model

    import torch

    torch.ao.quantization.quantize_dynamic(
        pipeline_model.model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    return pipeline_model


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    union = first.area + second.area - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def _deduplicate_same_class(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Suppress overlapping boxes for the same semantic concept.

    Open-vocabulary detectors can emit several boxes for one physical object.
    Keeping only the strongest overlapping detection prevents a single keyring
    from spawning several tracks while preserving distinct same-class objects
    that occupy different image regions.
    """
    kept: list[Detection] = []

    for detection in sorted(
        detections,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        duplicate = any(
            existing.class_name == detection.class_name
            and _intersection_over_union(existing.bbox, detection.bbox)
            >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)

    return kept


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class EmptyDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


class YoloDetector:
    """Adapter that converts Ultralytics YOLO results into BlackBox detections."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.30,
        allowed_class_names: tuple[str, ...] | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_name)

        self.model = model
        self.confidence_threshold = confidence_threshold
        self.allowed_class_names = (
            frozenset(allowed_class_names)
            if allowed_class_names is not None
            else None
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(source=frame, verbose=False)

        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = str(result.names[class_id])
            if (
                self.allowed_class_names is not None
                and class_name not in self.allowed_class_names
            ):
                continue

            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                )
            )

        return detections



class OpenVocabularyDetector:
    """Text-conditioned detector for concepts outside a fixed class list.

    By default this uses a Hugging Face zero-shot object detection pipeline.
    The pipeline is injected in tests so CI does not download a model.
    """

    def __init__(
        self,
        prompts: tuple[str, ...],
        confidence_threshold: float = 0.25,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        pipeline_model: Any | None = None,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not contain empty values")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0 and 1")

        self.prompts = tuple(prompt.strip() for prompt in prompts)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if pipeline_model is None:
            from transformers import pipeline

            pipeline_model = pipeline(
                task="zero-shot-object-detection",
                model=model_name,
            )

        self.pipeline_model = _maybe_quantize_pipeline(pipeline_model)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from PIL import Image

        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        results = self.pipeline_model(
            image,
            candidate_labels=list(self.prompts),
        )

        detections: list[Detection] = []
        for result in results:
            confidence = float(result["score"])
            if confidence < self.confidence_threshold:
                continue

            box = result["box"]
            detections.append(
                Detection(
                    class_name=str(result["label"]),
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=float(box["xmin"]),
                        y1=float(box["ymin"]),
                        x2=float(box["xmax"]),
                        y2=float(box["ymax"]),
                    ),
                )
            )

        return _deduplicate_same_class(
            detections,
            self.nms_iou_threshold,
        )


class HybridDetector:
    """Runs multiple detectors and concatenates their evidence.

    This keeps fixed-class and prompted detection separate instead of
    pretending they share the same confidence calibration.
    """

    def __init__(self, detectors: tuple[Detector, ...]) -> None:
        if not detectors:
            raise ValueError("detectors must not be empty")
        self.detectors = detectors

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(frame))
        return detections



class TargetAwareOpenVocabularyDetector:
    """Open-vocabulary detector backed by a mutable target registry."""

    def __init__(
        self,
        registry: TargetRegistry,
        confidence_threshold: float = 0.25,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        pipeline_model: Any | None = None,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0 and 1")

        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if pipeline_model is None:
            from transformers import pipeline

            pipeline_model = pipeline(
                task="zero-shot-object-detection",
                model=model_name,
            )

        self.pipeline_model = pipeline_model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        prompts = self.registry.prompts()
        if not prompts:
            return []

        from PIL import Image

        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        results = self.pipeline_model(
            image,
            candidate_labels=list(prompts),
        )

        detections: list[Detection] = []
        for result in results:
            confidence = float(result["score"])
            if confidence < self.confidence_threshold:
                continue

            box = result["box"]
            detections.append(
                Detection(
                    class_name=str(result["label"]),
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=float(box["xmin"]),
                        y1=float(box["ymin"]),
                        x2=float(box["xmax"]),
                        y2=float(box["ymax"]),
                    ),
                )
            )

        return _deduplicate_same_class(
            detections,
            self.nms_iou_threshold,
        )
