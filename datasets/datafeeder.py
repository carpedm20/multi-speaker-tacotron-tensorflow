import os
import time
import pprint
import random
import threading
import traceback
import numpy as np
from glob import glob
import tensorflow as tf
from collections import defaultdict

import text
from utils.infolog import log
from utils import parallel_run, remove_file
from audio import frames_to_hours
from audio.get_duration import get_durations


_pad = 0
_p_cmudict=0.5

def get_frame(path):
    data = np.load(path)
    n_frame = data["linear"].shape[0]
    n_token = len(data["tokens"])
    return (path, n_frame, n_token)

def get_path_dict(
        data_dirs, hparams, config,
        data_type, n_test=None,
        rng=np.random.RandomState(123)):

    # Load metadata:
    path_dict = {}
    for data_dir in data_dirs:
        paths = glob("{}/*.npz".format(data_dir))

        if data_type == 'train':
            rng.shuffle(paths)

        if not config.skip_path_filter:
            items = parallel_run(
                    get_frame, paths, desc="filter_by_min_max_frame_batch", parallel=True)

            min_n_frame = hparams.reduction_factor * hparams.min_iters
            max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor
            
            new_items = [(path, n) for path, n, n_tokens in items \
                    if min_n_frame <= n <= max_n_frame and n_tokens >= hparams.min_tokens]

            if any(check in data_dir for check in ["son", "yuinna"]):
                blacklists = [".0000.", ".0001.", "NB11479580.0001"]
                new_items = [item for item in new_items \
                        if any(check not in item[0] for check in blacklists)]

            new_paths = [path for path, n in new_items]
            new_n_frames = [n for path, n in new_items]

            hours = frames_to_hours(new_n_frames)

            log(' [{}] Loaded metadata for {} examples ({:.2f} hours)'. \
                    format(data_dir, len(new_n_frames), hours))
            log(' [{}] Max length: {}'.format(data_dir, max(new_n_frames)))
            log(' [{}] Min length: {}'.format(data_dir, min(new_n_frames)))
        else:
            new_paths = paths

        if data_type == 'train':
            new_paths = new_paths[:-n_test]
        elif data_type == 'test':
            new_paths = new_paths[-n_test:]
        else:
            raise Exception(" [!] Unkown data_type: {}".format(data_type))

        path_dict[data_dir] = new_paths

    return path_dict

class DataFeeder(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, data_dirs,
            hparams, config, batches_per_group, data_type, batch_size):
        super(DataFeeder, self).__init__()

        self._coord = coordinator
        self._hp = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._step = 0
        self._offset = defaultdict(lambda: 2)
        self._batches_per_group = batches_per_group
        self._cmudict=None

        self.rng = np.random.RandomState(config.random_seed)
        self.data_type = data_type
        self.batch_size = batch_size

        self.min_tokens = hparams.min_tokens
        self.min_n_frame = hparams.reduction_factor * hparams.min_iters
        self.max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor
        self.skip_path_filter = config.skip_path_filter

        # Load metadata:
        self.path_dict = get_path_dict(
                data_dirs, self._hp, config, self.data_type,
                n_test=self.batch_size, rng=self.rng)

        self.data_dirs = list(self.path_dict.keys())
        self.data_dir_to_id = {
                data_dir: idx for idx, data_dir in enumerate(self.data_dirs)}

        data_weight = {
                data_dir: 1. for data_dir in self.data_dirs
        }

        if self._hp.main_data_greedy_factor > 0 and \
                any(main_data in data_dir for data_dir in self.data_dirs \
                                         for main_data in self._hp.main_data):
            for main_data in self._hp.main_data:
                for data_dir in self.data_dirs:
                    if main_data in data_dir:
                        data_weight[data_dir] += self._hp.main_data_greedy_factor

        weight_Z = sum(data_weight.values())
        self.data_ratio = {
                data_dir: weight / weight_Z for data_dir, weight in data_weight.items()
        }

        log("="*40)
        log(pprint.pformat(self.data_ratio, indent=4))
        log("="*40)

        #audio_paths = [path.replace("/data/", "/audio/"). \
        #        replace(".npz", ".wav") for path in self.data_paths]
        #duration = get_durations(audio_paths, print_detail=False)

        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.

        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, [None], 'loss_coeff'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
        ]

        # Create queue for buffering data:
        dtypes = [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32]

        self.is_multi_speaker = len(self.data_dirs) > 1

        if self.is_multi_speaker:
            self._placeholders.append(
                    tf.placeholder(tf.int32, [None], 'inputs'),
            )
            dtypes.append(tf.int32)

        num_worker = 8 if self.data_type == 'train' else 1
        queue = tf.FIFOQueue(num_worker, dtypes, name='input_queue')

        self._enqueue_op = queue.enqueue(self._placeholders)

        if self.is_multi_speaker:
            self.inputs, self.input_lengths, self.loss_coeff, \
                    self.mel_targets, self.linear_targets, self.speaker_id = queue.dequeue()
        else:
            self.inputs, self.input_lengths, self.loss_coeff, \
                    self.mel_targets, self.linear_targets = queue.dequeue()

        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.loss_coeff.set_shape(self._placeholders[2].shape)
        self.mel_targets.set_shape(self._placeholders[3].shape)
        self.linear_targets.set_shape(self._placeholders[4].shape)

        if self.is_multi_speaker:
            self.speaker_id.set_shape(self._placeholders[5].shape)
        else:
            self.speaker_id = None

        if self.data_type == 'test':
            examples = []
            while True:
                for data_dir in self.data_dirs:
                    examples.append(self._get_next_example(data_dir))
                    #print(data_dir, text.sequence_to_text(examples[-1][0], False, True))
                    if len(examples) >= self.batch_size:
                        break
                if len(examples) >= self.batch_size:
                    break
            self.static_batches = [examples for _ in range(self._batches_per_group)]

        else:
            self.static_batches = None

        # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
        # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
        # synthesis (useful for proper nouns, etc.)
        if hparams.use_cmudict:
          cmudict_path = os.path.join(self.data_dirs[0], 'cmudict-0.7b')
          if not os.path.isfile(cmudict_path):
            raise Exception('If use_cmudict=True, you must download ' +
              'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s'  % cmudict_path)
          self._cmudict = text.cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
          log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
        else:
          self._cmudict = None


    def start_in_session(self, session, start_step):
        self._step = start_step
        self._session = session
        self.start()


    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)


    def _enqueue_next_group(self):
        start = time.time()

        # Read a group of examples:
        n = self.batch_size
        r = self._hp.reduction_factor

        if self.static_batches is not None:
            batches = self.static_batches
        else:
            examples = []
            for data_dir in self.data_dirs:
                if self._hp.initial_data_greedy:
                    if self._step < self._hp.initial_phase_step and \
                            any("krbook" in data_dir for data_dir in self.data_dirs):
                        data_dir = [data_dir for data_dir in self.data_dirs if "krbook" in data_dir][0]

                if self._step < self._hp.initial_phase_step:
                    example = [self._get_next_example(data_dir) \
                            for _ in range(int(n * self._batches_per_group // len(self.data_dirs)))]
                else:
                    example = [self._get_next_example(data_dir) \
                            for _ in range(int(n * self._batches_per_group * self.data_ratio[data_dir]))]
                examples.extend(example)
            examples.sort(key=lambda x: x[-1])

            batches = [examples[i:i+n] for i in range(0, len(examples), n)]
            self.rng.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r, self.rng, self.data_type)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)
            self._step += 1


    def _get_next_example(self, data_dir):
        '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
        data_paths = self.path_dict[data_dir]

        while True:
            if self._offset[data_dir] >= len(data_paths):
                self._offset[data_dir] = 0

                if self.data_type == 'train':
                    self.rng.shuffle(data_paths)

            data_path = data_paths[self._offset[data_dir]]
            self._offset[data_dir] += 1

            try:
                if os.path.exists(data_path):
                    data = np.load(data_path)
                else:
                    continue
            except:
                remove_file(data_path)
                continue

            if not self.skip_path_filter:
                break

            if self.min_n_frame <= data["linear"].shape[0] <= self.max_n_frame and \
                    len(data["tokens"]) > self.min_tokens:
                break

        input_data = data['tokens']
        mel_target = data['mel']

        # cmu_dict enabled -> convert some chararcter in known words to arpabet (p_cmudict possibilty)
        if self._cmudict and random.random()<_p_cmudict:
            txt = text.sequence_to_text(input_data, False, True)
            txt = ' '.join([self._maybe_get_arpabet(word) for word in txt.split(' ')])
            input_data = (text.text_to_sequence(txt, as_token=False))


        if 'loss_coeff' in data:
            loss_coeff = data['loss_coeff']
        else:
            loss_coeff = 1
        linear_target = data['linear']

        return (input_data, loss_coeff, mel_target, linear_target, 
                self.data_dir_to_id[data_dir], len(linear_target))
    
    def _maybe_get_arpabet(self, word):
        arpabet = self._cmudict.lookup(word)
        return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def _prepare_batch(batch, reduction_factor, rng, data_type=None):
    if data_type == 'train':
        rng.shuffle(batch)

    inputs = _prepare_inputs([x[0] for x in batch])
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
    loss_coeff = np.asarray([x[1] for x in batch], dtype=np.float32)

    mel_targets = _prepare_targets([x[2] for x in batch], reduction_factor)
    linear_targets = _prepare_targets([x[3] for x in batch], reduction_factor)

    if len(batch[0]) == 6:
        speaker_id = np.asarray([x[4] for x in batch], dtype=np.int32)
        return (inputs, input_lengths, loss_coeff,
                mel_targets, linear_targets, speaker_id)
    else:
        return (inputs, input_lengths, loss_coeff, mel_targets, linear_targets)


def _prepare_inputs(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
