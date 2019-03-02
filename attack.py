# attack.py -- generate audio adversarial examples
#
# Copyright (C) 2018, Hiromu Yakura <hiromu1996@gmail.com>.
# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

from __future__ import print_function

import argparse
import math
import os
import random
import shutil
import string
import struct
import sys
import tempfile
import time

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeepSpeech'))

tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits
from weight_decay_optimizers import AdamWOptimizer

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Sampling rate of the input files
Fs = 16000


class Attack:
    def __init__(self, sess, restore_path, audio, impulse, phrase, freq_min, freq_max, batch_size, learning_rate, weight_decay):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to actually generate the adversarial examples.
        """

        assert len(audio.shape) == 1
        assert len(impulse.shape) == 2

        self.sess = sess
        self.phrase = phrase

        self.impulse_size = impulse.shape[0]
        self.batch_size = batch_size

        # Store arguments as constant tensors.
        original = tf.constant(audio.astype(np.float32))

        # Create impulse filters in a frequency domain
        conv_length = audio.shape[0] + impulse.shape[1] - 1
        nfft = 2 ** int(math.ceil(math.log(conv_length, 2)))
        imp_filters = tf.constant(np.fft.rfft(impulse, nfft).astype(np.complex64))

        # Change filters to apply dynamically
        self.imp_indices = tf.Variable(np.zeros((batch_size, ), dtype=np.int32), name='qq_filters')
        apply_filters = tf.gather(imp_filters, self.imp_indices)

        # Create all the variables necessary they are prefixed with qq_ just so that we know which ones are ours
        # so when we restore the session we don't clobber them.
        self.delta = tf.Variable(np.random.normal(0, np.sqrt(np.abs(audio).mean()), audio.shape).astype(np.float32), name='qq_delta')

        # Create a band pass filter to be applied to the perturbation.
        freq = np.fft.rfftfreq(audio.shape[0], 1.0 / Fs)
        bp_filter = ((freq_min < freq) & (freq < freq_max)).astype(np.int32)
        bp_filter = tf.constant(bp_filter.astype(np.complex64))

        # Apply the filter for the delta to simulate the real-world and create an adversarial example.
        self.delta_filtered = tf.spectral.irfft(tf.spectral.rfft(self.delta) * bp_filter)
        self.ae_input = original + tf.pad(self.delta_filtered, [[0, original.get_shape().as_list()[0] - self.delta_filtered.get_shape().as_list()[0]]])

        # Convolve the impulse responses to the input
        fft_length = tf.constant(np.array([nfft], dtype=np.int32))
        ae_frequency = tf.spectral.rfft(self.ae_input, fft_length=[nfft]) * apply_filters
        ae_convolved = tf.spectral.irfft(ae_frequency, fft_length=[nfft])[:, :conv_length]

        # Normalize the convolved audio
        max_audio = tf.reduce_max(tf.abs(ae_convolved), axis=1, keepdims=True)
        self.ae_transformed = ae_convolved / max_audio * tf.reduce_max(tf.abs(self.ae_input))

        # Add a tiny bit of noise to help make sure that we can clip our values to 16-bit integers and not break things.
        self.noise_ratio = tf.Variable(np.ones((1, ), dtype=np.float32), name='qq_noise_ratio')
        small_noise = tf.random_normal(self.ae_transformed.get_shape().as_list(), stddev=2 ** 14) * ([1] - self.noise_ratio)
        final_input = tf.clip_by_value(self.ae_transformed + small_noise, -2 ** 15, 2 ** 15 - 1)

        # Feed this final value to get the logits.
        lengths = tf.constant(np.array([(conv_length - 1) // 320] * batch_size, dtype=np.int32))
        self.logits = get_logits(final_input, lengths)

        # And finally restore the graph to make the classifier actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        # Calculate CTC loss.
        target_phrase = tf.constant(np.array([list(phrase) for _ in range(batch_size)], dtype=np.int32))
        target_phrase_lengths = tf.constant(np.array([len(phrase) for _ in range(batch_size)], dtype=np.int32))
        target = ctc_label_dense_to_sparse(target_phrase, target_phrase_lengths, batch_size)
        self.ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=self.logits, sequence_length=lengths)

        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = AdamWOptimizer(weight_decay, learning_rate)

        gradients = optimizer.compute_gradients(self.ctcloss, [self.delta])
        self.train = optimizer.apply_gradients(gradients, decay_var_list=[self.delta]) # tf.sign(grad)?
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars + [self.delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(self.logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, outdir, num_iterations=5000):
        sess = self.sess

        # Initialize all of the variables
        sess.run(tf.variables_initializer([self.delta, self.noise_ratio]))

        # Record loss of each impulse response
        imp_losses = np.full(self.impulse_size, 1e10)

        # Create misc variables
        prefix = ''.join([random.choice(string.lowercase) for _ in range(3)])
        time_last, time_start = time.time(), time.time()

        # We'll make a bunch of iterations of gradient descent here
        for itr in xrange(num_iterations + 1):
            indice = np.random.choice(self.impulse_size, self.batch_size, p=(imp_losses / imp_losses.sum()))

            # Actually do the optimization step
            decoded, logits, ctcloss, ae_transformed, ae_input, delta_filtered, _ = sess.run([self.decoded, self.logits, self.ctcloss, self.ae_transformed, self.ae_input, self.delta_filtered, self.train], {self.imp_indices: indice})
            imp_losses[indice] = ctcloss

            # Report progress
            print('Iter: %d, Elapsed Time: %.3f, Iter Time: %.3f\n\tLosses: %s\n\t Delta: %s' % \
                  (itr, time.time() - time_start, time.time() - time_last, ' '.join('% 6.2f' % x for x in ctcloss), np.array_str(delta_filtered, max_line_width=120)))
            time_last = time.time()

            # Print out some debug information every 5 iterations.
            if itr % 5 == 0:
                res = np.zeros(decoded[0].dense_shape) + len(toks) - 1
                for j in xrange(len(decoded[0].values)):
                    x, y = decoded[0].indices[j]
                    res[x, y] = decoded[0].values[j]

                # Here we print the strings that are recognized.
                res = [''.join(toks[int(x)] for x in y).replace('-', '') for y in res]
                print('Recognition:\n\t' + '\n\t'.join(res))

                # And here we print the argmax of the alignment.
                res_al = np.argmax(logits, axis=2).T
                res_al = [''.join(toks[int(x)] for x in y) for y in res_al]
                print('Alignment:\n\t' + '\n\t'.join(res_al))

                # Check if we've succeeded then we should record our progress and decrease the rescale constant.
                matched = filter(lambda index: res[index] == ''.join([toks[x] for x in self.phrase]), range(self.batch_size))
                if len(matched) > self.batch_size * 0.5:
                    # Get the current constant
                    ratio = sess.run(self.noise_ratio)
                    print('=> It: %d, Noise Ratio: %.3f' % (itr, 1.0 - ratio[0]))

                    # Update with the new noise
                    sess.run(self.noise_ratio.assign(ratio * 0.99))

                if itr % 100 == 0 or len(matched) > self.batch_size * 0.5:
                    # Just for debugging, save the adversarial example so we can see it if we want
                    wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr)), Fs, np.array(np.clip(np.round(ae_input), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    wav.write(os.path.join(outdir, '%s-delta-%d.wav' % (prefix, itr)), Fs, np.array(np.clip(np.round(delta_filtered), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    for i in xrange(ae_transformed.shape[0]):
                        wav.write(os.path.join(outdir, '%s-conv-%d-%d.wav' % (prefix, itr, i)), Fs, np.array(np.clip(np.round(ae_transformed[i]), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))

                    # Save also the logits
                    np.save(os.path.join(outdir, '%s-logit-%d.npy' % (prefix, itr)), logits[:, matched, :])

    
def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest='input', required=True,
                        help='Input audio .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--imp', type=str, dest='impulse', nargs='+', required=True,
                        help='Input impulse response .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--target', type=str, required=True,
                        help='Target transcriptions')
    parser.add_argument('--out', type=str, required=True,
                        help='Directory for saving intermediate files')
    parser.add_argument('--batch_size', type=int, required=False, default=20,
                        help='Batch size for generation')
    parser.add_argument('--freq_min', type=int, required=False, default=1000,
                        help='Lower limit of band pass filter for adversarial noise')
    parser.add_argument('--freq_max', type=int, required=False, default=4000,
                        help='Higher limit of band pass filter for adversarial noise')
    parser.add_argument('--lr', type=float, required=False, default=200,
                        help='Learning rate for optimization')
    parser.add_argument('--decay', type=float, required=False, default=0.001,
                        help='Weight decay for optimization')
    parser.add_argument('--iterations', type=int, required=False, default=1000,
                        help='Maximum number of iterations of gradient descent')
    parser.add_argument('--session', type=str, required=False,
                        default=os.path.join(os.path.dirname(__file__), 'models/session_dump'),
                        help='Path for the session file taken from DeepSpeech')
    args = parser.parse_args()

    print('Command line:', args)
    
    with tf.Session() as sess:
        # Load the inputs that we're given
        fs, audio = wav.read(args.input)
        assert fs == Fs
        print('Source dB:', 20 * np.log10(np.max(np.abs(audio))))

        irs = []
        for i in range(len(args.impulse)):
            fs, ir = wav.read(args.impulse[i])
            assert fs == Fs
            irs.append(ir)

        # Pad the impulse responses
        maxlen = max(map(len, irs))
        for i in range(len(irs)):
            irs[i] = np.concatenate((irs[i], np.zeros(maxlen - irs[i].shape[0], dtype=irs[i].dtype)))
        irs = np.array(irs)

        # Set up the attack class and run it
        attack = Attack(sess, args.session, audio, irs, [toks.index(x) for x in args.target], freq_min=args.freq_min, freq_max=args.freq_max,
                        batch_size=args.batch_size, learning_rate=args.lr, weight_decay=args.decay)
        attack.attack(outdir=args.out, num_iterations=args.iterations)


if __name__ == '__main__':
    main()
