import sys
from pathlib import Path

from blackbox.perception.detector import YoloDetector
from blackbox.perception.pipeline import VideoDetectionPipeline


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python scripts/process_video.py "
            "<video_path> [output_dir]"
        )
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_dir = (
        Path(sys.argv[2])
        if len(sys.argv) == 3
        else Path("runs") / input_path.stem
    )

    pipeline = VideoDetectionPipeline(
        detector=YoloDetector(),
    )

    summary = pipeline.process(
        input_path=input_path,
        output_dir=output_dir,
        write_annotated_video=True,
    )

    print(f"Processed frames: {summary.frames_processed}")
    print(f"Detections emitted: {summary.detections_emitted}")
    print(
        "Track observations emitted: "
        f"{summary.track_observations_emitted}"
    )
    print(f"Observations: {summary.observations_path}")

    if summary.annotated_video_path is not None:
        print(
            "Annotated video: "
            f"{summary.annotated_video_path}"
        )


if __name__ == "__main__":
    main()
