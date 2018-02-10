import os
import re
import sys
import json
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from pydub import silence
from pydub import AudioSegment
from functools import partial

from hparams import hparams
from utils import parallel_run, add_postfix, str2bool
from audio import load_audio, save_audio, get_duration, get_silence

def abs_mean(x):
	return abs(x).mean()

def remove_breath(audio):
	edges = librosa.effects.split(
			audio, top_db=40, frame_length=128, hop_length=32)

	for idx in range(len(edges)):
		start_idx, end_idx = edges[idx][0], edges[idx][1]
		if start_idx < len(audio):
			if abs_mean(audio[start_idx:end_idx]) < abs_mean(audio) - 0.05:
				audio[start_idx:end_idx] = 0

	return audio

def split_on_silence_with_librosa(
		audio_path, top_db=40, frame_length=1024, hop_length=256,
		skip_idx=0, out_ext="wav",
		min_segment_length=3, max_segment_length=8,
		pre_silence_length=0, post_silence_length=0):

	filename = os.path.basename(audio_path).split('.', 1)[0]
	in_ext = audio_path.rsplit(".")[1]

	audio = load_audio(audio_path)

	edges = librosa.effects.split(audio,
			top_db=top_db, frame_length=frame_length, hop_length=hop_length)

	new_audio = np.zeros_like(audio)
	for idx, (start, end) in enumerate(edges[skip_idx:]):
		new_audio[start:end] = remove_breath(audio[start:end])
		
	save_audio(new_audio, add_postfix(audio_path, "no_breath"))
	audio = new_audio
	edges = librosa.effects.split(audio,
			top_db=top_db, frame_length=frame_length, hop_length=hop_length)

	audio_paths = []
	for idx, (start, end) in enumerate(edges[skip_idx:]):
		segment = audio[start:end]
		duration = get_duration(segment)

		if duration <= min_segment_length or duration >= max_segment_length:
			continue

		output_path = "{}/{}.{:04d}.{}".format(
				os.path.dirname(audio_path), filename, idx, out_ext)

		padded_segment = np.concatenate([
				get_silence(pre_silence_length),
				segment,
				get_silence(post_silence_length),
		])


		
		save_audio(padded_segment, output_path)
		audio_paths.append(output_path)

	return audio_paths

def read_audio(audio_path):
	return AudioSegment.from_file(audio_path)

def split_on_silence_with_pydub(
		audio_path, deepspeech, skip_idx=0, out_ext="wav",
		silence_thresh=-40, min_silence_len=400,
		silence_chunk_len=100, keep_silence=100,
		min_segment_length=0): #silence_chunk_len 100->200

	filename = os.path.basename(audio_path).split('.', 1)[0]
	in_ext = audio_path.rsplit(".")[1]

	audio = read_audio(audio_path)
	audio= audio.set_channels(1)
	audio_sample_width=audio.sample_width
	min_chunk_len=int(float(1000)*min_segment_length)

	
	not_silence_ranges = silence.detect_nonsilent(
		audio, min_silence_len=silence_chunk_len,
		silence_thresh=silence_thresh)

	edges = [not_silence_ranges[0]]

	for idx in range(1, len(not_silence_ranges)-1):
		cur_start = not_silence_ranges[idx][0]
		prev_end = edges[-1][1]
		prev_start = edges[-1][0]
		prev_len = prev_end - prev_start

		# if silence is too short or nonsilent is too short
		# merge current nonsilence with prev one 
		if cur_start - prev_end < min_silence_len or (min_chunk_len!=0 and prev_len < min_chunk_len) : 
			edges[-1][1] = not_silence_ranges[idx][1]
		else:
			edges.append(not_silence_ranges[idx])
	print("Finished finding Edges")
	
	audio_paths = []
	for idx, (start_idx, end_idx) in enumerate(edges[skip_idx:]):
		start_idx = max(0, start_idx - keep_silence)
		end_idx += keep_silence

		target_audio_path = "{}/{}.{:04d}.{}".format(
				os.path.dirname(audio_path), filename, idx, out_ext)
		segment=audio[start_idx:end_idx]
		
		# Set this to deepspeech compatible 
		if deepspeech:
			temp=segment.set_frame_rate(16000)
			temp=temp.set_sample_width(2)
			temp=temp.set_channels(1)
			temp.export(target_audio_path, out_ext)	# for soundsegment
		else:
			segment.export(target_audio_path, out_ext)	# for soundsegment
		audio_paths.append(target_audio_path)

	return audio_paths

def split_on_silence_batch(audio_paths, method, **kargv):
	audio_paths.sort()
	method = method.lower()
	deepspeech=kargv['deepspeech']
	min_segment_length=kargv['min_segment_length']
	if deepspeech:
		print('DeepSpeech compatibility enabled, using 16000Hz/16bit setting')
	
	if method == "librosa":
		fn = partial(split_on_silence_with_librosa, **kargv)
	elif method == "pydub":
		fn = partial(split_on_silence_with_pydub, **kargv)

	parallel_run(fn, audio_paths,
			desc="Split on silence", parallel=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio_pattern', required=True)
	parser.add_argument('--out_ext', default='wav')
	parser.add_argument('--method', choices=['librosa', 'pydub'], required=True)
	parser.add_argument('--deepspeech', default=False, type=str2bool)
	parser.add_argument('--min_segment_length', default=0 ,type=float)
	parser.add_argument('--silence_chunk_len',default=100, type=int)
	config = parser.parse_args()

	audio_paths = glob(config.audio_pattern)

	split_on_silence_batch(
			audio_paths, config.method,
			out_ext=config.out_ext, deepspeech=config.deepspeech,
			min_segment_length =config.min_segment_length, 
			silence_chunk_len=config.silence_chunk_len
	)
