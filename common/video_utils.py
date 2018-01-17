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
# Date Created: 2018-01-17

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def video_left_right_flip(images):
    '''
    Performs tf.image.flip_left_right on entire list of video frames.
    Work around since the random selection must be consistent for entire video

    :param images: Tensor constaining video frames (N,H,W,3)
    :return: images: Tensor constaining video frames left-right flipped (N,H,W,3)
    '''
    images_list = tf.unstack(images)
    for i in range(len(images_list)):
        images_list[i] = tf.image.flip_left_right(images_list[i])
    return tf.stack(images_list)

# def preprocess_video(images, label):
#     '''
#     Given the 'images' Tensor of video frames (N,H,W,3) perform the following
#     preprocessing steps:
#
#     1. Takes a random crop of size CROP_SIZExCROP_SIZE from the video frames.
#     2. Optionally performs random left-right flipping of the video.
#     3. Performs video normalization, to the range [-0.5, +0.5]
#
#     :param images: Tensor (tf.uint8) constaining video frames (N,H,W,3)
#     :param label:  Tensor (tf.int64) constaining video frames ()
#     :return:
#     '''
#
#     # Take a random crop of the video, returns tensor of shape (N,CROP_SIZE,CROP_SIZE,3)
#     images = tf.random_crop(images, (SEQ_NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3))
#
#     if RANDOM_LEFT_RIGHT_FLIP:
#         # Consistent left_right_flip for entire video
#         sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
#         option = tf.less(sample, 0.5)
#         images = tf.cond(option,
#                          lambda: video_left_right_flip(images),
#                          lambda: tf.identity(images))
#
#     # Normalization: [0, 255] => [-0.5, +0.5] floats
#     images = tf.cast(images, tf.float32) * (1./255.) - 0.5
#     return images, label