import cv2
import numpy as np


def main() -> None:
    width = 640
    height = 360
    fps = 30
    seconds = 3

    output_path = "data/samples/test.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    for frame_index in range(fps * seconds):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        x = 20 + frame_index * 4

        cv2.rectangle(
            frame,
            (x, 140),
            (x + 80, 220),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            f"frame {frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()

    print(f"Created {output_path}")


if __name__ == "__main__":
    main()