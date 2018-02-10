# Multi-Speaker Tacotron in TensorFlow

TensorFlow implementation of:

- [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947)
- [Listening while Speaking: Speech Chain by Deep Learning](https://arxiv.org/abs/1707.04879)
- [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)

Samples audios (in Korean) can be found [here](http://carpedm20.github.io/tacotron/en.html).

![model](./assets/model.png)


## Prerequisites

- Python 3.6+
- FFmpeg
- [Tensorflow 1.3](https://www.tensorflow.org/install/)


## Usage

### 1. Install prerequisites

After preparing [Tensorflow](https://www.tensorflow.org/install/), install prerequisites with:

    pip3 install -r requirements.txt
    python -c "import nltk; nltk.download('punkt')"

If you want to synthesize a speech in Korean dicrectly, follow [2-3. Download pre-trained models](#2-3-download-pre-trained-models).


### 2-1. Generate custom datasets

The `datasets` directory should look like:

    datasets
    ├── son
    │   ├── alignment.json
    │   └── audio
    │       ├── 1.mp3
    │       ├── 2.mp3
    │       ├── 3.mp3
    │       └── ...
    └── YOUR_DATASET
        ├── alignment.json
        └── audio
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            └── ...

and `YOUR_DATASET/alignment.json` should look like:

    {
        "./datasets/YOUR_DATASET/audio/001.mp3": "My name is Taehoon Kim.",
        "./datasets/YOUR_DATASET/audio/002.mp3": "The buses aren't the problem.",
        "./datasets/YOUR_DATASET/audio/003.mp3": "They have discovered a new particle.",
    }

After you prepare as described, you should genearte preprocessed data with:

    python3 -m datasets.generate_data ./datasets/YOUR_DATASET/alignment.json


### 2-2. Generate Korean datasets

Follow below commands. (explain with `son` dataset)

0. To automate an alignment between sounds and texts, prepare `GOOGLE_APPLICATION_CREDENTIALS` to use [Google Speech Recognition API](https://cloud.google.com/speech/). To get credentials, read [this](https://developers.google.com/identity/protocols/application-default-credentials).

       export GOOGLE_APPLICATION_CREDENTIALS="YOUR-GOOGLE.CREDENTIALS.json"

1. Download speech(or video) and text.

       python3 -m datasets.son.download

2. Segment all audios on silence.

       python3 -m audio.silence --audio_pattern "./datasets/son/audio/*.wav" --method=pydub

3. By using [Google Speech Recognition API](https://cloud.google.com/speech/), we predict sentences for all segmented audios.

       python3 -m recognition.google --audio_pattern "./datasets/son/audio/*.*.wav"

4. By comparing original text and recognised text, save `audio<->text` pair information into `./datasets/son/alignment.json`.

       python3 -m recognition.alignment --recognition_path "./datasets/son/recognition.json" --score_threshold=0.5

5. Finally, generated numpy files which will be used in training.

       python3 -m datasets.generate_data ./datasets/son/alignment.json

Because the automatic generation is extremely naive, the dataset is noisy. However, if you have enough datasets (20+ hours with random initialization or 5+ hours with pretrained model initialization), you can expect an acceptable quality of audio synthesis.

### 2-3. Generate English datasets

1. Download speech dataset at https://keithito.com/LJ-Speech-Dataset/ into datasets/LJSpeech_1_0 folder (folder "wavs" and metadata.csv should be placed inside the folder)

2. Convert metadata CSV file to json file. (arguments are available for changing preferences)
		
		python3 -m datasets.LJSpeech_1_0.prepare

3. Finally, generate numpy files which will be used in training.
		
		python3 -m datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json
		

### 3. Train a model

The important hyperparameters for a models are defined in `hparams.py`.

(**Change `cleaners` in `hparams.py` from `korean_cleaners` to `english_cleaners` to train with English dataset**)

To train a single-speaker model:

    python3 train.py --data_path=datasets/son
    python3 train.py --data_path=datasets/son --initialize_path=PATH_TO_CHECKPOINT

To train a multi-speaker model:

    # after change `model_type` in `hparams.py` to `deepvoice` or `simple`
    python3 train.py --data_path=datasets/son1,datasets/son2

To restart a training from previous experiments such as `logs/son-20171015`:

    python3 train.py --data_path=datasets/son --load_path logs/son-20171015

If you don't have good and enough (10+ hours) dataset, it would be better to use `--initialize_path` to use a well-trained model as initial parameters.


### 4. Synthesize audio

You can train your own models with:

    python3 app.py --load_path logs/son-20171015 --num_speakers=1

or generate audio directly with:

    python3 synthesizer.py --load_path logs/son-20171015 --text "이거 실화냐?"
	
### 4-1. Synthesizing non-korean(english) audio

For generating non-korean audio, you must set the argument --is_korean False.
		
	python3 app.py --load_path logs/LJSpeech_1_0-20180108 --num_speakers=1 --is_korean=False
	python3 synthesizer.py --load_path logs/LJSpeech_1_0-20180108 --text="Winter is coming." --is_korean=False

## Results

Training attention on single speaker model:

![model](./assets/attention_single_speaker.gif)

Training attention on multi speaker model:

![model](./assets/attention_multi_speaker.gif)


## Disclaimer

This is not an official [DEVSISTERS](http://devsisters.com/) product. This project is not responsible for misuse or for any damage that you may cause. You agree that you use this software at your own risk.


## References

- [Keith Ito](https://github.com/keithito)'s [tacotron](https://github.com/keithito/tacotron)
- [DEVIEW 2017 presentation](https://www.slideshare.net/carpedm20/deview-2017-80824162)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
