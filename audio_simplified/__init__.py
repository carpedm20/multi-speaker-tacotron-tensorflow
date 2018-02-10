# Code based on https://github.com/keithito/tacotron/blob/master/util/audio.py
import math
import numpy as np
from scipy import signal

import librosa
import librosa.filters


SAMPLING_RATE=16000


def load_audio(path, pre_silence_length=0, post_silence_length=0):
	audio, sr = librosa.core.load(path, sr=None)
	if pre_silence_length > 0 or post_silence_length > 0:
		audio = np.concatenate([
				get_silence(pre_silence_length, sr),
				audio,
				get_silence(post_silence_length, sr),
		])
	return audio, sr

def save_audio(audio, path, sample_rate=SAMPLING_RATE):
	audio *= 32767 / max(0.01, np.max(np.abs(audio)))
	librosa.output.write_wav(path, audio.astype(np.int16),
			sample_rate)

	print(" [*] Audio saved: {}".format(path))


def resample_audio(audio, original_sr, target_sample_rate):
	return librosa.core.resample(
			audio, original_sr, target_sample_rate)


def get_duration(audio, sr=SAMPLING_RATE):
	return librosa.core.get_duration(audio, sr=sr)


def get_silence(sec, sr=SAMPLING_RATE):
	return np.zeros(sr * sec)

def convert_to_int16(y):
	y_int=y*np.iinfo(np.int16).max
	y_int=y_int.astype(np.int16)
	return y_int
