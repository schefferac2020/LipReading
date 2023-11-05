'''
Description:    This script ... 
Usage:          python3 LipSegmenter --video_path=[vid_path]
'''

import cv2
import os
import argparse
import numpy as np
import imutils
import dlib
from tqdm import tqdm
import json


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

# Takes in a shape (68, 2) numpy array and gets 
# the min bbox around the lips
def get_min_lip_bbox(shape):
    lip_points = shape[48:, :]
    
    min_x = lip_points[:, 0].min()
    max_x = lip_points[:, 0].max()
    min_y = lip_points[:, 1].min()
    max_y = lip_points[:, 1].max()


    return (min_x, min_y, max_x - min_x, max_y - min_y)


# TODO: I can get the FPS here
def extract_frames(vid_path, output_dir):
    cap = cv2.VideoCapture(vid_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
    
    cap.release()

    print(f"Extracted {frame_count} frames from {vid_path}")
    print(f"The frames_per_sec are {cap.get(cv2.CAP_PROP_FPS)}")

    return frame_count

def get_lips(input_images_dir, output_images_dir, DEBUG=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

    image_files = [f for f in os.listdir(input_images_dir) if os.path.isfile(os.path.join(input_images_dir, f)) and f != ".DS_Store"]
    image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))


    lip_widths = []
    lip_heights = []

    file_num = 0
    for fname in tqdm(image_files, desc="Lip Cropping"):
        image = cv2.imread(os.path.join(input_images_dir, fname))
        # image = imutils.resize(image, width=500) # Test to see if we need this? 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces in the image... 
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            (x, y, w, h) = get_min_lip_bbox(shape)
            lip_widths.append(w)
            lip_heights.append(h)

            image = image[y:y + h, x:x + w]

            # (x, y, w, h) = rect_to_bb(rect)
            # cv2.rectangle(image, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 0), 2)
            # cv2.putText(image, "Lips".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Put the dots on the face!
            if DEBUG:
                num = 1
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    num+=1
            
            # Write the new image to a file
            output_path = os.path.join(output_images_dir, f"{file_num}.png")
            cv2.imwrite(output_path, image)
        file_num += 1
	
    print("Finished extracting those nice smoochers!!!")

def get_phoneme_for_every_frame(json_filepath, num_frames, frame_rate = 30):
    phonemes = ["silence"]*num_frames
    with open(json_filepath, 'r') as json_file:
        json_data = json.load(json_file)

    for word in json_data["words"]:
        phone_start_time = word["start"]

        for phone in word["phones"]:
            phone_end_time = phone_start_time + phone["duration"]

            phone_start_frame = int(phone_start_time*frame_rate)
            phone_end_frame = int(phone_end_time*frame_rate)
            for i in range(phone_start_frame, phone_end_frame+1):
                phonemes[i] = phone["phone"]

            phone_start_time = phone_end_time


    return phonemes    

def phoneme_visualization(lip_frames_dir, phonemes_per_frame, output_video_name):
    final_width = 300
    final_height = 150

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 30, (final_width, final_height))
    
    lip_files = [f for f in os.listdir(lip_frames_dir) if os.path.isfile(os.path.join(lip_frames_dir, f)) and f != ".DS_Store"]
    lip_files = sorted(lip_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    frame_num = 0
    for lip_file_name in lip_files:
        lip_file_path = os.path.join(lip_frames_dir, lip_file_name)
        
        lips_img = cv2.imread(lip_file_path)
        lips_height, lips_width, _ = lips_img.shape
    
        larger_image = np.full((final_height, final_width, 3), [141, 227, 240], dtype=np.uint8)
    
        x = int(final_width/2 - lips_width/2)
        y = int(final_height/2 - lips_height/2)
        larger_image[y:y+lips_height, x:x+lips_width] = lips_img

        phoneme = phonemes_per_frame[frame_num]
        cv2.putText(larger_image, f"GT Phoneme: {phoneme}", (int(final_width/2 - 100), final_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(larger_image)
        frame_num += 1

    # Release the VideoWriter object and close the video file
    out.release()

    print("Finished writing visualization to the video file:", output_video_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video to Frames Converter')
    parser.add_argument("-v", '--video_path', type=str, help='Path to the input video file')
    parser.add_argument("-j", '--json_path', type=str, help='Path to the input video file')
    args = parser.parse_args()

    print("Reading in", args.video_path)
    face_frames_dir = f"{args.video_path}_frames"
    lips_frames_dir = f"{args.video_path}_frames/lips"
    os.makedirs(face_frames_dir, exist_ok=True)
    os.makedirs(lips_frames_dir, exist_ok=True)

    #1. Dump the frames of the video to folder
    num_frames = extract_frames(args.video_path, face_frames_dir)

    #2. Convert images to images of lips...
    get_lips(face_frames_dir, lips_frames_dir)

    # Get phonemes for each frame
    phonemes_per_frame = get_phoneme_for_every_frame(args.json_path, num_frames)

    #3. Do some visualization if we can...
    phoneme_visualization(lips_frames_dir, phonemes_per_frame, "visualization.mp4")


