import dlib
import cv2
import numpy as np
import argparse
import os
import fnmatch
from tqdm import tqdm
import shutil

# Load the pre-trained facial landmark predictor from dlib
predictor_path = "./models/shape_predictor_68_face_landmarks.dat"  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

bad_crops = []
last_landmarks = None

past_angles = np.array([])
moving_average_filter_length = 5

def get_face_landmarks(image):
    faces = face_detector(image, 1)
    
    if len(faces) == 0:
        return None
    
    face = faces[0]
    
    landmarks = landmark_predictor(image, face)
    landmarks_np = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])
    
    return landmarks_np

def correct_face_rotation(image, landmarks):
    global past_angles
    # Calculate the angle of rotation
    curr_angle = np.degrees(np.arctan2(landmarks[45, 1] - landmarks[36, 1], landmarks[45, 0] - landmarks[36, 0]))
    avg_angle = (np.sum(past_angles[-moving_average_filter_length:]) + curr_angle) / (len(past_angles) + 1)
    past_angles = np.append(past_angles, curr_angle)

    center = tuple(np.mean(landmarks, axis=0).astype(int))
    center = (int(center[0]), int(center[1]))
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    return rotated_image

def crop_lips(image, landmarks):
    lips_indices = list(range(48, 60))

    # Calculate the center of the lips
    lips_center = np.mean([(landmarks[i, 0], landmarks[i, 1]) for i in lips_indices], axis=0)

    # Crop a 96x96 region around the center of the lips
    crop_size = 96
    x, y = int(lips_center[0] - crop_size / 2), int(lips_center[1] - crop_size / 2)
    cropped_image = image[y:y+crop_size, x:x+crop_size]
    
    return cropped_image


def video_cropper(video_path, output_video_path, debug):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a video writer object
    if debug:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (96, 96), isColor=False)

    frames = np.empty((frame_count, 96, 96), dtype=np.uint8)

    # Process each frame in the video
    i = 0
    while cap.isOpened():
        if i % 100 == 0:
            print(f"On frame {i}")
        ret, frame = cap.read()
        
        
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        # Get facial landmarks
        landmarks = get_face_landmarks(frame)

        if landmarks is None:
            landmarks = last_landmarks
        
        if landmarks is not None:
            last_landmarks = landmarks
            # Correct face rotation
            rotated_frame = correct_face_rotation(frame, landmarks)
            
            # Crop lips and resize to 96x96
            lips_crop = crop_lips(rotated_frame, landmarks)
            if lips_crop.shape != (96,96):
                # Bad crop! Release the video that we were making
                bad_crops.append(video_file)
                if debug:
                    out.release()
                    os.remove(output_video_path)
                return False, []

            frames[i] = lips_crop

            if debug:
                out.write(lips_crop)
            # for point in landmarks:
            #     cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)
            
            # cv2.imshow("Facial Landmarks", frame)
            # cv2.imshow("Lips Crop", lips_crop)
        else:
            print("There are no crops!")
            bad_crops.append(video_path)
            # if debug:
            #     out.release()
            #     os.remove(output_video_path)
            # return False, []

        i += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if debug:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return True, frames

def find_files_with_extension(directory, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, f'*{extension}'):
            file_paths.append(os.path.join(root, file))
    
    return file_paths


def save_as_npz(data, output_path):
    np.savez(output_path, data=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video to Frames Converter')
    parser.add_argument("-v", '--video-path', default=None ,type=str, help='Path to the input video file')
    parser.add_argument("-d", "--dataset-dir",  type=str, help='Path to dataset directory containing many video files')
    parser.add_argument("--debug", action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Make the output directory..
    output_dir = "preprocessing_output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if (args.video_path != None):
        cropped_video_path = "cropped_video.mp4"
        success, frames = video_cropper(args.video_path, cropped_video_path, args.debug)
        if success:
            save_as_npz(frames, "output.npz")
    else:
        # Do video cropper on every file in the dataset...
        video_files = find_files_with_extension(args.dataset_dir, "mp4")

        for video_file in tqdm(video_files, desc="Lip Cropping"):
            dir_name = os.path.basename(os.path.dirname(video_file))
            file_name, extension = os.path.splitext(os.path.basename(video_file))

            new_mp4_file_name = dir_name + "-" + file_name + ".mp4"
            new_mp4_file_path = os.path.join(output_dir, new_mp4_file_name)

            npz_file_name = dir_name + "-" + file_name + ".npz"
            new_npz_file_path = os.path.join(output_dir, npz_file_name)

            new_text_file_name = dir_name + "-" + file_name + ".txt"
            new_text_file_path = os.path.join(output_dir, new_text_file_name)

            
            
            success, frames = video_cropper(video_file, new_mp4_file_path, args.debug)
            if success:
                save_as_npz(frames, new_npz_file_path)

                # Copy the txt file to preprocessing_output folder...
                old_text_file_path = os.path.splitext(video_file)[0] + ".txt"
                shutil.copy(old_text_file_path, new_text_file_path)

        print("FINISHED PREPROCESSING SUCCESSFULLY!")
        print(f"Successfully preprocessed {len(video_files) - len(bad_crops)}/{len(video_files)} videos")



        pass