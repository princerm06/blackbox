import sys
from pathlib import Path

from blackbox.perception.video import VideoReader


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_video.py <video_path>")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    reader = VideoReader(path)

    print(f"Video: {path}")
    print(f"Resolution: {reader.width}x{reader.height}")
    print(f"FPS: {reader.fps}")
    print(f"Frames: {reader.frame_count}")
    print()

    try:
        for frame in reader.frames():
            print(
                f"frame={frame.frame_index} "
                f"timestamp={frame.timestamp_ms}ms"
            )
    finally:
        reader.close()


if __name__ == "__main__":
    main()