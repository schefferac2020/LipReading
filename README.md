# LipReading
Christian Foreman, Ashwin Saxena, and Andrew Scheffer

## Usage
download this [dlib pretrained model](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) for facial feature detection and place it in the `models/` directory. Then run the following:

```console
foo@bar:~$ python3 LipSegmenter.py --video_path=test_1/initial_test.mov --json_path=test_1/align.json
```
