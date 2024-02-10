# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""
from __future__ import print_function
import os
import sys
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'src' directory (assuming 'hopper_controller.py' is inside 'pyHopper')
src_dir = os.path.abspath(os.path.join(script_dir, ".."))

# Get the path of the package directory (assuming main_script.py is in the same directory as my_package).
package_dir = os.path.dirname(os.path.abspath(__file__))

# Add the package directory to sys.path.
sys.path.insert(0, package_dir)

# Add the 'src' directory to sys.path
sys.path.insert(0, src_dir)
import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf
import pandas as pd

import utils.vggish_input as vggish_input
from utils.vggish_input import waveform_to_examples
import utils.vggish_params as vggish_params
import utils.vggish_postprocess as vggish_postprocess
import utils.vggish_slim as vggish_slim
# from utils.utils import get_windows_from_paths, load_data_from_hd5


flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'pyHopperVGG/src/pre_trained_models/vggish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'pyHopperVGG/src/pre_trained_models/vggish/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def feature_extraction_vgg(HData, sample_rate):
    feature_array = []
    for i in range(0,len(HData)):
        example = waveform_to_examples(HData[i], sample_rate)
        feature_array.append(np.squeeze(example, axis=0))

    examples_batch = np.array(feature_array)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    # If needed, prepare a record writer to store the postprocessed embeddings.
    writer = tf.python_io.TFRecordWriter(
      FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        # print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        return postprocessed_batch

# def main(_):
#     windows, target = get_windows_from_paths("data/audio_data/convertedData.hd5", "data/audio_data/indexCSV.csv")
#     HData, HLabels, HGroups = load_data_from_hd5("data/audio_data/convertedData.hd5", "data/audio_data/indexCSV.csv")
#     sample_rate = pd.read_csv("data/audio_data/indexCSV.csv")
#     sample_rate = sample_rate['sampleRate'][0]
#
#     X = feature_extraction_vgg(HData, sample_rate)






if __name__ == '__main__':
  tf.app.run()
