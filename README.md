# LipReading
Christian Foreman, Ashwin Saxena, and Andrew Scheffer

## Usage
download this [dlib pretrained model](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) for facial feature detection and place it in the `models/` directory. Then run one of the following:

To run the lip extractor on just one video file, run the following. This will output a `.npz` file that can be used to extract lip embeddings. 
```console
foo@bar:~$ python3 preprocess_videos.py -v path/to/video.mp4
```

To run the lip extractor on a whole dataset use the following.
```console
foo@bar:~$ python3 preprocess_videos.py -d path/to/dataset
```

Note: Use the `--debug` option to generate **.mp4** files as well as **.npz** files.