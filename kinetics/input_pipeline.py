# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-01-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2

import common.video_utils

################################################################################

# Defines the input shape to the network
SEQ_NUM_FRAMES = 64
CROP_SIZE = 224
RANDOM_LEFT_RIGHT_FLIP = True
BATCH_SIZE = 64
NUM_EPOCHS = 100


def decode(serialized_example, sess):
    '''
    Given a serialized example in which the frames are stored as
    compressed JPG images 'frames/0001', 'frames/0002' etc., this
    function samples SEQ_NUM_FRAMES from the frame list, decodes them from
    JPG into a tensor and packs them to obtain a tensor of shape (N,H,W,3).
    Returns the the tuple (frames, class_label (tf.int64)

    :param serialized_example: serialized example from tf.data.TFRecordDataset
    :return: tuple: (frames (tf.uint8), class_label (tf.int64)
    '''

    # Prepare feature list; read encoded JPG images as bytes
    features = dict()
    features["class_label"] = tf.FixedLenFeature((), tf.int64)
    for i in range(SEQ_NUM_FRAMES):
        features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.parse_single_example(serialized_example, features)

    # Decode the encoded JPG images
    images = []
    for i in range(SEQ_NUM_FRAMES):
        images.append(tf.image.decode_jpeg(parsed_features["frames/{:04d}".format(i)]))

    # Pack the frames into one big tensor of shape (N,H,W,3)
    images = tf.stack(images)
    label  = tf.cast(parsed_features['class_label'], tf.int64)

    # Randomly sample offset ... ? Need to produce strings for dict indices after this

    num_frames = tf.cast(parsed_features['num_frames'], tf.int64)
    offset = tf.random_uniform(shape=(), minval=0, maxval=label, dtype=tf.int64)

    return images, label

################################################################################

if  __name__ == "__main__":

    import glob
    tfrecord_files = glob.glob("/home/tomrunia/data/Kinetics/Full/tfrecords/val/*.tfrecords")
    tfrecord_files.sort()

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.map(decode)
    #dataset = dataset.map(common.video_utils.preprocess_video)

    # The parameter is the queue size
    dataset = dataset.shuffle(1000 + 3 * BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    sess.run(init_op)

    while True:

        # Fetch a new batch from the dataset
        batch_videos, batch_labels = sess.run(next_batch)

        for sample_idx in range(BATCH_SIZE):
            print("Class label = {}".format(batch_labels[sample_idx]))
            for frame_idx in range(SEQ_NUM_FRAMES):
                cv2.imshow("image", batch_videos[sample_idx,frame_idx])
                cv2.waitKey(20)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()