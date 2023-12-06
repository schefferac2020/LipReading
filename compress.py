import cv2
from tqdm import tqdm

def compress_video(input_path, output_path, target_width=300):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_aspect_ratio = original_width / original_height

    # Calculate the target height to maintain the aspect ratio
    target_height = int(target_width / original_aspect_ratio)

    # Create VideoWriter object to save the compressed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec based on your preference
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, target_height))

    print("Trying to crop down video with", total_frames, "frames")
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    while True:
        # Read a frame from the video file
        ret, frame = cap.read()
        pbar.update(1)

        # Break if no more frames are available
        if not ret:
            break

        # Resize the frame while maintaining the aspect ratio
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Write the resized frame to the output video file
        out.write(resized_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Compression complete. Compressed video saved at: {output_path}")

if __name__ == "__main__":
    # Replace 'input_video.mov' and 'output_compressed_video.mp4' with your input and output file paths
    input_video_path = '/Users/drewscheffer/Dev/EECS542/VPN/princess_bride.MOV'
    output_compressed_path = '/Users/drewscheffer/Dev/EECS542/VPN/output_compressed.MOV'
    
    compress_video(input_video_path, output_compressed_path)