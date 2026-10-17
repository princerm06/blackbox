import argparse
from pathlib import Path

from blackbox.perception.detector import (
    HybridDetector,
    TargetAwareOpenVocabularyDetector,
    YoloDetector,
)
from blackbox.perception.pipeline import VideoDetectionPipeline
from blackbox.perception.targets import TargetRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a video while registering user-facing tracking targets."
        )
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        required=True,
        help=(
            "Target in DISPLAY=PROMPT form, for example "
            "--target 'My Keys=keys'. If '=' is omitted, the same text "
            "is used for both."
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


def parse_target(value: str) -> tuple[str, str]:
    if "=" not in value:
        clean = value.strip()
        return clean, clean

    display_name, prompt = value.split("=", 1)
    return display_name.strip(), prompt.strip()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = (
        args.output_dir
        or Path("runs") / f"{args.video_path.stem}-targets"
    )

    registry = TargetRegistry()
    for raw_target in args.targets:
        display_name, prompt = parse_target(raw_target)
        registry.register(display_name, prompt=prompt)

    detector = HybridDetector(
        (
            YoloDetector(
                confidence_threshold=args.yolo_threshold,
            ),
            TargetAwareOpenVocabularyDetector(
                registry=registry,
                confidence_threshold=args.open_threshold,
            ),
        )
    )

    print("tracking targets:")
    for target in registry.enabled_targets():
        print(
            f"- {target.display_name} "
            f"(id={target.target_id}, prompt={target.prompt!r})"
        )

    summary = VideoDetectionPipeline(detector=detector).process(
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
