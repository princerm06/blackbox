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
