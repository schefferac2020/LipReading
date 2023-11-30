'''
CUDA_VISIBLE_DEVICES=0 python main.py --modality video \
                                      --extract-feats \
                                      --config-path ./configs/lrw_resnet18_dctcn.json \
                                      --model-path ./models/lrw_resnet18_dctcn_video.pth.tar \
                                      --mouth-patch-path mouth_vid.npz \
                                      --mouth-embedding-out-path embeddings/emb_out.npz

'''


'''
Usage: This bitty will combine a bunch of npzs into list of objects to be used for frozen pretrained

'''
import gzip
import pickle
import torch
import numpy as np
import argparse
import os
import subprocess
import fnmatch
from tqdm import tqdm

def find_files_with_extension(directory, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, f'*{extension}'):
            file_paths.append(os.path.join(root, file))
    
    return file_paths


def convert_images_to_embeddings(path_to_npzs):
    for path_to_npz in tqdm(path_to_npzs):
        # Execute the embedding command
        embedding_output_path = os.path.splitext(path_to_npz)[0] + "_emb.npz"

        subprocess.run(['python3', './LipEmbeddings/main.py', '--modality', 'video', '--extract-feats',
                                      '--config-path', './LipEmbeddings/configs/lrw_resnet18_dctcn.json',
                                      '--model-path', './LipEmbeddings/models/lrw_resnet18_dctcn_video.pth.tar',
                                      '--mouth-patch-path', path_to_npz,
                                      '--mouth-embedding-out-path', embedding_output_path])



def main():
    parser = argparse.ArgumentParser(description='Video to Frames Converter')
    parser.add_argument("-d", '--data-dir', default=None ,type=str, help='Path to the data directory')
    args = parser.parse_args()
    
    # Get all NPZs
    path_to_npzs = find_files_with_extension(args.data_dir, "npz")

    convert_images_to_embeddings(path_to_npzs)

if __name__ == "__main__":
    main()