import cv2
import numpy as np
import sys

def convert_to_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit(1)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize an empty numpy array to store the frames
    frames = np.empty((frame_count, frame_height, frame_width), dtype=np.uint8)

    # Read frames and convert to grayscale
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            sys.exit(1)

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Store the grayscale frame in the numpy array
        frames[i] = frame_gray

    # Close the video file
    cap.release()

    return frames

def save_as_npz(data, output_path):
    np.savez(output_path, data=data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_video.mp4 output_file.npz")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_npz_path = sys.argv[2]

    # Convert video to grayscale
    grayscale_frames = convert_to_grayscale(input_video_path)

    # Save as NPZ
    save_as_npz(grayscale_frames, output_npz_path)

    print(f"Conversion successful. Grayscale frames saved to {output_npz_path}.")
