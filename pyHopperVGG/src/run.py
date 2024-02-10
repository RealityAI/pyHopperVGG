# from models.models.research.audioset.vggish.vggish_input import waveform_to_examples
# pip install git+ssh://git@github.com/RealityAI/hopper-v2.git@main

from utils.utils import get_windows_from_paths
from utils.vggish_input import waveform_to_examples
import utils.vggish_postprocess as vggish_postprocess
import utils.vggish_slim as vggish_slim
import pandas as pd
import tensorflow.compat.v1 as tf

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


windows, target = get_windows_from_paths("data/audio_data/convertedData.hd5", "data/audio_data/indexCSV.csv")

sample_rate = pd.read_csv("data/audio_data/indexCSV.csv")
sample_rate = sample_rate['sampleRate'][0]
feature_array = []
for window in windows:
    example = waveform_to_examples(window[0], sample_rate)
    feature_array.append(example)


X = feature_array
y = target



