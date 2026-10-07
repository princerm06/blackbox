from typing import Any, Protocol

import numpy as np

from blackbox.schemas.perception import BoundingBox, Detection


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
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_name)

        self.model = model
        self.confidence_threshold = confidence_threshold

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
    ) -> None:
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not contain empty values")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        self.prompts = tuple(prompt.strip() for prompt in prompts)
        self.confidence_threshold = confidence_threshold

        if pipeline_model is None:
            from transformers import pipeline

            pipeline_model = pipeline(
                task="zero-shot-object-detection",
                model=model_name,
            )

        self.pipeline_model = pipeline_model

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

        return detections


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
