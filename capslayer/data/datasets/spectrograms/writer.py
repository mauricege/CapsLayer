# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
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
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from capslayer.data.utils.TFRecordHelper import int64_feature, bytes_feature


SPECTROGRAM_FILES = {
    'train': ('train.npy', 'train-labels.npy'),
    'eval': ('eval.npy', 'eval-labels.npy'),
    'test': ('test.npy', 'test-labels.npy')
}


def load_spectrograms(path, split):
    split = split.lower()
    image_file, label_file = [os.path.join(path, file_name) for file_name in SPECTROGRAM_FILES[split]]

    images = np.load(image_file)
    images = images.reshape(-1, images.shape[1], images.shape[2], 1).astype("float32")
    labels = np.load(label_file)
    return(zip(images, labels))


def encode_and_write(dataset, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            height, width, depth = image.shape
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(
                                       feature={'image': bytes_feature(image_raw),
                                                'label': int64_feature(label),
                                                'height': int64_feature(height),
                                                'width': int64_feature(width),
                                                'depth': int64_feature(depth)}))
            writer.write(example.SerializeToString())


def tfrecord_runner(path, force=True, splitting='TV'):
    train_set = load_spectrograms(path, 'train')
    train_set_outpath = os.path.join(path, "train.tfrecord")
    if not os.path.exists(train_set_outpath) or force:
        encode_and_write(train_set, train_set_outpath)
    if splitting == 'TV':
        eval_set = load_spectrograms(path, 'eval')
        eval_set_outpath = os.path.join(path, "eval.tfrecord")
        if not os.path.exists(eval_set_outpath) or force:
            encode_and_write(eval_set, eval_set_outpath)

    if splitting == 'TVT':
        test_set = load_spectrograms(path, 'test')
        test_set_outpath = os.path.join(path, "test.tfrecord")
        if not os.path.exists(test_set_outpath) or force:
            encode_and_write(test_set, test_set_outpath)







