# Robust Audio Adversarial Example for a Physical Attack

This repository includes the implementation of our paper: [Robust Audio Adversarial Example for a Physical Attack](https://www.ijcai.org/proceedings/2019/741).

You can find generated examples at [our project page](https://yumetaro.info/projects/audio-ae/).

## Usage

### Preparation

1. Install the dependencies

- librosa
- numpy
- pyaudio
- scipy
- tensorflow (<= 1.8.0)

2. Clone Mozilla's DeepSpeech into a directory named `DeepSpeech`

```
$ git clone https://github.com/mozilla/DeepSpeech.git
```

3. Change the version of DeepSpeech

```
$ cd DeepSpeech
$ git checkout tags/v0.1.0
$ cd ..
```


4. Download the DeepSpeech model

```
$ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
$ tar -xzf deepspeech-0.1.0-models.tar.gz
```

5. Verify that you have downloaded the correct file

```
$ md5sum models/output_graph.pb 
08a9e6e8dc450007a0df0a37956bc795  output_graph.pb
```

6. Convert the .pb to a TensorFlow checkpoint file

```
$ python make_checkpoint.py
```

7. Collect room impulse responses and convert them into 16kHz mono wav files.

> You can refer many databases from this script: [https://github.com/kaldi-asr/kaldi/blob/master/egs/aspire/s5/local/multi_condition/prepare_impulses_noises.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/aspire/s5/local/multi_condition/prepare_impulses_noises.sh)

### Generate adversarial examples

Let us `sample.wav` to be a 16kHz mono wav file to add perturbations, and `rir` to be a directory containing room impulse responses.

Then, you can generate audio adversarial examples recognized as "hello world" as follows:

```
$ mkdir results
$ python attack.py --in sample.wav --imp rir/*.wav --target "hello world" --out results
```

### Test generated adversarial examples

1. Playback and record the generated adversarial example

```
$ python record.py ae.wav ae_recorded.wav
```

2. Recognize with the DeepSpeech

```
$ python recognize.py models/output_graph.pb ae_recorded.wav models/alphabet.txt models/lm.binary models/trie
```

3. Check the obtained transcription

```
$ cat ae_recorded.txt
hello world
```

## Notice

The most of the source code in this repository is based on [https://github.com/carlini/audio_adversarial_examples](https://github.com/carlini/audio_adversarial_examples) and distributed under the same license (2-clause BSD) except for the following files.

- `recognize.py`: This is based on [https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py](https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py) and distributed under the Mozilla Public License v2.
- `weight_decay_optimizers.py`: This is taken from [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/opt/python/training/weight_decay_optimizers.py) and originally distributed under the Apache License v2.

## Citation

If you use this code for your research, please cite this paper:

```
@inproceedings{yakura2019robust,
  title={Robust Audio Adversarial Example for a Physical Attack},
  author={Yakura, Hiromu and Sakuma, Jun},
  journal={Proceedings of the 28th International Joint Conference on Artificial Intelligence},
  pages={5334--5341},
  year={2019}
}
 ```
