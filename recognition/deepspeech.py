# -*- coding: utf-8 -*-
"""
Created on 1:04 2018-02-06
Instead of using Google Speech Recognition API, this creates recognition.json file
from given audioclips (given in pattern) using DeepSpeech Speech Recognition 
Implementation. (Available at https://github.com/mozilla/DeepSpeech )

There are several limitations and workarounds. 
1. DeepSpeech is designed for Linux (As this uses customized TF). 
-> CPU based usage is possible on Windows using WSL (Windows Subsystem for Linux)
(See https://fotidim.com/deepspeech-on-windows-wsl-287cb27557d4 )
2. DeepSpeech is assuming alphabet-based language. More work is required for Korean representations.
(Supports for different characterset is currently available)
3. DeepSpeech pretrained model(English) is workable with low-noise environment with english native speaker
(The performance is severely affected by non-native accents or background noise)
4. The speed performance is limited. Inference time ~~ 1.5*speech
(On Xeon E3-1245v3 in WSL environment)

However, at least, everybody loves free lunch!! Enjoy!
@author: engiecat (github)
"""

import io
import os
import json
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
from glob import glob
from functools import partial
from utils import parallel_run, remove_file, backup_file, write_json, load_json


from deepspeech.model import Model
from audio_simplified import load_audio, save_audio, resample_audio, get_duration, convert_to_int16
from timeit import default_timer as timer

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

def text_recognition(path, args, ds):
	root, ext = os.path.splitext(path)
	txt_path = root + ".txt"	
	
	if os.path.exists(txt_path): # if it is already done
		out=load_json(txt_path)
		print(' [!] Skip {} because recognition txt already exists'.format(path))
		return out
	
	content, content_sr = load_audio(
		path, pre_silence_length=args.pre_silence_length,
		post_silence_length=args.post_silence_length)

	max_duration = args.max_duration - \
			args.pre_silence_length - args.post_silence_length
	min_duration = args.min_duration + \
			args.pre_silence_length + args.post_silence_length
	audio_duration = get_duration(content, content_sr)

	if audio_duration >= max_duration:
		print(" [!] Skip {} because of duration: {} > {}". \
				format(path, audio_duration, max_duration))
		return {}
	if audio_duration <= min_duration:
		print(" [!] Skip {} because of duration: {} < {}". \
				format(path, audio_duration, min_duration))
		return {}

	content = resample_audio(content, content_sr, args.sample_rate)
	content = convert_to_int16(content)
	
	result = ds.stt(content, args.sample_rate)
	
	print("{} - {}".format(path,result))
	out={path:result} # in dict.
	
	with open(txt_path, 'w') as f:
		json.dump(out, f, indent=2, ensure_ascii=False)
	return out

def text_recognition_batch(paths, args, ds):
	paths.sort()
	results = {}
	items = parallel_run(
			partial(text_recognition, args=args, ds=ds), paths,
			desc="text_recognition_batch_deepspeech", parallel=False)
	for item in items:
		results.update(item)
	return results


def load_model(model_path, alphabet_path, lm_path, trie_path):
	print('Loading model from file %s' % (model_path), file=sys.stderr)
	model_load_start = timer()
	ds = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
	model_load_end = timer() - model_load_start
	print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)
	
	if lm_path and trie_path:
		print('Loading language model from files %s %s' % (lm_path, trie_path), file=sys.stderr)
		lm_load_start = timer()
		ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
							   WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
		lm_load_end = timer() - lm_load_start
		print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)
	return ds

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('model', type=str,
						help='Path to the model (protocol buffer binary file)')
	parser.add_argument('alphabet', type=str,
						help='Path to the argsuration file specifying the alphabet used by the network')
	parser.add_argument('lm', type=str, nargs='?',
						help='Path to the language model binary file')
	parser.add_argument('trie', type=str, nargs='?',
						help='Path to the language model trie file created with native_client/generate_trie')
	parser.add_argument('--audio_pattern', required=True, help=' Path to the audioclips')
	parser.add_argument('--recognition_filename', default="recognition.json")
	parser.add_argument('--pre_silence_length', default=1, type=int)
	parser.add_argument('--post_silence_length', default=1, type=int)
	parser.add_argument('--max_duration', default=60, type=int)
	parser.add_argument('--min_duration', default=2, type=int)
	parser.add_argument('--sample_rate', default=16000, type=int)
	
	args=parser.parse_args()
	
	ds_model = load_model(args.model, args.alphabet, args.lm, args.trie)
	
	audio_dir = os.path.dirname(args.audio_pattern)

	for tmp_path in glob(os.path.join(audio_dir, "*.tmp.*")):
		remove_file(tmp_path)

	paths = glob(args.audio_pattern)
	paths.sort()
	results = text_recognition_batch(paths, args, ds_model)

	base_dir = os.path.dirname(audio_dir)
	recognition_path = \
			os.path.join(base_dir, args.recognition_filename)

	if os.path.exists(recognition_path):
		backup_file(recognition_path)

	write_json(recognition_path, results)
	
	
	