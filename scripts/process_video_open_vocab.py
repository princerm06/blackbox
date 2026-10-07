import argparse
from pathlib import Path

from blackbox.perception.detector import (
    HybridDetector,
    OpenVocabularyDetector,
    YoloDetector,
)
from blackbox.perception.pipeline import VideoDetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a video with fixed-class YOLO plus prompted "
            "open-vocabulary detection."
        )
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        required=True,
        help=(
            "Object concept to detect. Repeat for multiple concepts, "
            "for example --prompt keys --prompt wallet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--yolo-threshold",
        type=float,
        default=0.30,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = (
        args.output_dir
        or Path("runs") / f"{args.video_path.stem}-open-vocab"
    )

    detector = HybridDetector(
        (
            YoloDetector(
                confidence_threshold=args.yolo_threshold,
            ),
            OpenVocabularyDetector(
                prompts=tuple(args.prompts),
                confidence_threshold=args.open_threshold,
            ),
        )
    )
    pipeline = VideoDetectionPipeline(detector=detector)
    summary = pipeline.process(
        input_path=args.video_path,
        output_dir=output_dir,
        write_annotated_video=True,
    )

    print(f"frames={summary.frames_processed}")
    print(f"detections={summary.detections_emitted}")
    print(f"events={summary.events_emitted}")
    print(f"observations={summary.observations_path}")
    print(f"annotated_video={summary.annotated_video_path}")


if __name__ == "__main__":
    main()
