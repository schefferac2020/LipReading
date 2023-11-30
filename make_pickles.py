'''
Usage: This bitty will combine a bunch of npzs into list of objects to be used for frozen pretrained

'''
import gzip
import pickle
import torch
import numpy as np
import argparse
import os
import fnmatch


# wanna do the inverse of this
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def find_files_with_extension(directory, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, f'*.{extension}'):
            file_paths.append(os.path.join(root, file))
    
    return file_paths


def extract_text_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Text: '):
                return line[len('Text: '):].strip().lower()


def convert_npz_to_pickle(npz_filenames, pickle_filename):

    combined_data = []

    # Load embeddings from each NPZ file and store in the dictionary
    for npz_filename in npz_filenames:
        if '_emb' in npz_filename:
            continue
        
        text_filename = os.path.splitext(npz_filename)[0] + ".txt"
        obj = {}

        embedding_filename = os.path.splitext(npz_filename)[0] + "_emb.npz"
        
        embedding = torch.tensor(np.load(embedding_filename)["data"]).squeeze()
        #folder, file = os.basename(npz_filename).split('-')
        obj["name"] = npz_filename
        obj['gloss'] = 'my name is gloss'
        obj['signer'] = 'i <3 jjpark'
        obj['text'] = extract_text_from_file(text_filename)
        obj['sign'] = embedding

        combined_data.append(obj)

    # Save combined data to a single pickle file
    print(combined_data)
    with gzip.open(pickle_filename, "wb") as f:
        pickle.dump(combined_data, f)


def main():
    parser = argparse.ArgumentParser(description='Video to Frames Converter')
    parser.add_argument("-d", '--data-dir', default=None ,type=str, help='Path to the data directory')
    parser.add_argument("-o", '--output-file', default='embeddings.pkl.gz' ,type=str, help='Filename of where to put the embeddings')
    parser.add_argument("--debug", action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Get all NPZs (images + embeddings)
    path_to_npzs = find_files_with_extension(args.data_dir, "npz")

    convert_npz_to_pickle(path_to_npzs, args.output_file)


if __name__ == "__main__":
    main()