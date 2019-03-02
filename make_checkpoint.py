# make_checkpoint.py -- convert a .pb to a TF session dump
#
# Copyright (C) 2018, Hiromu Yakura <hiromu1996@gmail.com>.
# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

from __future__ import print_function

import argparse
import os
import sys

from tensorflow.core.framework.graph_pb2 import *
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), 'DeepSpeech'))
from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse

# Okay, so this is ugly. We don't want DeepSpeech to crash
# when we haven't built the language model.
# So we're just going to monkeypatch TF and make it a no-op.
# Sue me.
tf.load_op_library = lambda x: x
import DeepSpeech


def make_checkpoint(model_path, audio_path, save_path):
    graph_def = GraphDef()
    loaded = graph_def.ParseFromString(open(model_path, 'rb').read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, [None, None, None],
                                   name='new_input')
        # Load the saved .pb into the current graph to let us grab
        # access to the weights.
        logits, = tf.import_graph_def(
            graph_def,
            input_map={'input_node:0': new_input},
            return_elements=['logits:0'],
            name='newname',
            op_dict=None,
            producer_op_list=None
        )

        # Now let's dump these weights into a new copy of the network.
        with tf.Session(graph=graph) as sess:
            # Sample sentence, to make sure we've done it right
            mfcc = audiofile_to_input_vector(audio_path, 26, 9)

            # Okay, so this is ugly again.
            # We just want it to not crash.
            tf.app.flags.FLAGS.alphabet_config_path = \
                os.path.join(os.path.dirname(__file__), 'DeepSpeech/data/alphabet.txt')
            DeepSpeech.initialize_globals()
            logits2 = DeepSpeech.BiRNN(new_input, [len(mfcc)], [0]*10)

            # Here's where all the work happens. Copy the variables
            # over from the .pb to the session object.
            for var in tf.global_variables():
                sess.run(var.assign(sess.run('newname/'+var.name)))

            # Test to make sure we did it right.
            res = (sess.run(logits, {new_input: [mfcc],
                                     'newname/input_lengths:0': [len(mfcc)]}).flatten())
            res2 = (sess.run(logits2, {new_input: [mfcc]})).flatten()
            print('This value should be small', np.sum(np.abs(res - res2)))

            # And finally save the constructed session.
            saver = tf.train.Saver()
            saver.save(sess, save_path)


def main():
    get_path = lambda x: os.path.join(os.path.dirname(__file__), x)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, default=get_path('models/output_graph.pb'),
                        help='Input TensorFlow graph file taken from DeepSpeech')
    parser.add_argument('--audio', type=str,
                        default=get_path('DeepSpeech/data/smoke_test/LDC93S1.wav'),
                        help='Sample audio file for testing')
    parser.add_argument('--out', type=str, default=get_path('models/session_dump'),
                        help='Path for saving the constructed session')
    args = parser.parse_args()

    make_checkpoint(args.model, args.audio, args.out)


if __name__ == '__main__':
    main()
