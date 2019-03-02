# record.py -- playback and record audio adversarial examples
#
# Copyright (C) 2018, Hiromu Yakura <hiromu1996@gmail.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

import sys
import commands
import time
import math
import os
import tempfile
import pyaudio
import librosa
import numpy as np

fs = 16000
nchannel = 1
chunk = 512
reverb = 1000

frame = 0
indata, outdata = [], []

def callback(in_data, frame_count, time_info, status):
    global frame, indata, outdata
    indata.append(in_data)
    frame += 1
    return (outdata[frame * nchannel * chunk: (frame + 1) * nchannel * chunk], pyaudio.paContinue)

def main(out_path, in_path):
    global frame, indata, outdata

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=nchannel, rate=fs, input=True, output=True, frames_per_buffer=chunk, stream_callback=callback)
    
    outdata = np.array(librosa.load(out_path, fs)[0], dtype=np.float32)
    nframe = int(math.ceil((len(outdata) + reverb * 1.5) / chunk))
    outdata = np.concatenate([np.zeros(int(reverb * 0.5), dtype=np.float32), outdata, np.zeros(reverb * 2, dtype=np.float32)])

    stream.start_stream()
    while frame < nframe:
        pass
    stream.stop_stream()

    rcv_data = np.frombuffer(b''.join(indata), dtype=np.float32)
    rcv_sig = np.zeros((nchannel, len(outdata) + reverb), dtype=np.float32)

    for i in range(nchannel):
        for j in range(len(outdata) + reverb):
            if j * nchannel + i < len(rcv_data):
                rcv_sig[i][j] = rcv_data[j * nchannel + i].astype(np.float32)

    if nchannel == 1:
        rcv_sig = rcv_sig.reshape((-1, ))

    _, tmppath = tempfile.mkstemp()
    librosa.output.write_wav(tmppath, rcv_sig, fs)
    commands.getoutput('ffmpeg -y -i %s -acodec pcm_s16le -ac %d -ar %d %s' % (tmppath, nchannel, fs, in_path))
    os.remove(tmppath)

    stream.close()
    pa.terminate()


if __name__== '__main__':
    main(sys.argv[1], sys.argv[2])
