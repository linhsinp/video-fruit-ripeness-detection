import logging
import os

import cv2

# Directory paths
DATA_DIR = "./data"
COMPRESSED_DIR = "./compressed_videos"
os.makedirs(COMPRESSED_DIR, exist_ok=True)

# Compression settings
FIXED_WIDTH = 864
CODEC = "mp4v"


def compress_video(input_path: str, output_path: str, fixed_width=720, codec="mp4v"):
    """Compress a video to a fixed width while maintaining the aspect ratio."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {input_path}")
        return

    # Get the original frame rate and resolution
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(fixed_width * aspect_ratio)
    resolution = (fixed_width, new_height)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the new resolution
        resized_frame = cv2.resize(frame, resolution)

        # Write the resized frame to the output video
        out.write(resized_frame)

    cap.release()
    out.release()
    logging.info(f"Compressed video saved to: {output_path}")


def main():
    """Compress all videos in the data directory."""
    for video_name in os.listdir(DATA_DIR):
        input_path = os.path.join(DATA_DIR, video_name)
        output_path = os.path.join(COMPRESSED_DIR, video_name)

        # Skip non-video files
        if not input_path.lower().endswith((".mp4", ".mov", ".avi")):
            logging.info(f"Skipping non-video file: {input_path}")
            continue

        logging.info(f"Compressing video: {input_path}")
        compress_video(input_path, output_path, fixed_width=FIXED_WIDTH, codec=CODEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
