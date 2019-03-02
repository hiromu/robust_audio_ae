# recognize.py -- actually classify a sequence with DeepSpeech
#
# Copyright (C) 2018, Hiromu Yakura <hiromu1996@gmail.com>.
# 
# This source code is based on https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
# The original source code is subject to the terms of the Mozilla Public License, v.2.0.
# You may obtain a copy of the License at http://mozilla.org/MPL/2.0/.

from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import argparse
import sys
import scipy.io.wavfile as wav

import glob
import os

from deepspeech.model import Model

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

def main():
    parser = argparse.ArgumentParser(description='Benchmarking tooling for DeepSpeech native_client.')
    parser.add_argument('model', type=str,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('audio', type=str,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('alphabet', type=str,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('lm', type=str, nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('trie', type=str, nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    args = parser.parse_args()

    print('Loading model from file %s' % (args.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

    if args.lm and args.trie:
        print('Loading language model from files %s %s' % (args.lm, args.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

    for path in sorted(glob.glob(args.audio))[::1]:
        target = os.path.splitext(path)[0] + '.txt'
        if os.path.exists(target):
            continue

        fs, audio = wav.read(path)
        # We can assume 16kHz
        audio_length = len(audio) * (1 / 16000)
        assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"
    
        print('Running inference of %s.' % path, file=sys.stderr)
        inference_start = timer()
        text = ds.stt(audio, fs)
        print(text)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

        with open(target, 'w') as out:
            out.write(text)

if __name__ == '__main__':
    main()
