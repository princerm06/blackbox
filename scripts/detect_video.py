import sys
from pathlib import Path

from blackbox.perception.detector import YoloDetector
from blackbox.perception.video import VideoReader


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/detect_video.py <video_path>")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    reader = VideoReader(path)
    detector = YoloDetector()

    try:
        for video_frame in reader.frames():
            detections = detector.detect(video_frame.image)

            print(
                f"frame={video_frame.frame_index} "
                f"timestamp={video_frame.timestamp_ms}ms"
            )

            if not detections:
                print("  no detections")

            for detection in detections:
                bbox = detection.bbox
                print(
                    f"  {detection.class_name} "
                    f"confidence={detection.confidence:.2f} "
                    f"bbox=("
                    f"{bbox.x1:.0f}, "
                    f"{bbox.y1:.0f}, "
                    f"{bbox.x2:.0f}, "
                    f"{bbox.y2:.0f})"
                )

            # First-frame smoke test only. Full-video processing comes next.
            break
    finally:
        reader.close()


if __name__ == "__main__":
    main()
