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

import os
import glob
import time
import h5py

import numpy as np
import tensorflow as tf
import jpeg4py as jpeg

#from mini_kinetics.mini_kinetics import *
import common.video_utils

################################################################################

# Defines the input shape to the network
SEQ_NUM_FRAMES = 64
CROP_SIZE = 224
RANDOM_LEFT_RIGHT_FLIP = True
BATCH_SIZE = 64
NUM_EPOCHS = 100

################################################################################

def read_examples_hdf5(filename, label):

    start = time.process_time()
    with h5py.File(filename, 'r') as hf:

        labels = hf['labels'].value
        example_shapes = hf['example_shapes'].value
        success = hf['success'].value
        num_examples = int(np.sum(success))

        example_idx = 0

        #if not success[example_idx]: continue

        vid_length, vid_height, vid_width, vid_channels = \
            example_shapes[example_idx]

        if vid_length < SEQ_NUM_FRAMES:
            print("Video too short...")
            #continue

        offset_time   = np.random.randint(0, max(vid_length-SEQ_NUM_FRAMES, 1))
        offset_height = np.random.randint(0, vid_height-CROP_SIZE)
        offset_width  = np.random.randint(0, vid_width-CROP_SIZE)

        frames = np.zeros((min(SEQ_NUM_FRAMES, vid_length), CROP_SIZE, CROP_SIZE, vid_channels), np.uint8)

        for frame_idx in range(offset_time, min(offset_time+SEQ_NUM_FRAMES, vid_length)):

            # Read encoded JPG as bytes string
            dset_name = "video_{:06d}".format(example_idx)
            frame_jpg_encode = np.fromstring(hf[dset_name][frame_idx], dtype=np.uint8)

            # Using libturbojpeg is much faster than cv2.decode()
            frame_decode = jpeg.JPEG(frame_jpg_encode).decode(pixfmt=jpeg.TJPF_BGR) # BGR
            #frame_decode = cv2.imdecode(frame_jpg_encode, -1) # BGR

            # Cropping the frames here as almost no speed penalty
            frames[frame_idx-offset_time] = \
                frame_decode[offset_height:offset_height+CROP_SIZE,
                             offset_width:offset_width+CROP_SIZE,:]

        # If the video is too short, loop it to satisfy the length
        # This is directly copied from Carreira (CVPR, 2017)
        if vid_length < SEQ_NUM_FRAMES:
            num_reps = int(np.ceil(SEQ_NUM_FRAMES/vid_length))
            frames = np.tile(frames, reps=(num_reps,1,1,1))
            frames = frames[0:SEQ_NUM_FRAMES,]

    end = time.process_time()
    duration = end-start
    example_per_second = num_examples/duration
    # print("Read and decoded {} JPG image(s) from HDF5 in {:.2f} seconds, "
    #       "{:.2f} examples/seconds".format(
    #     1, duration, example_per_second))

    return frames, labels[example_idx]


if  __name__ == "__main__":

    split = 'val'
    mini_kinetics_path = "/home/tomrunia/data/Kinetics/Mini-200/"
    hdf5_data_path = os.path.join(mini_kinetics_path, 'videos_as_hdf5', split)

    # File that contains YouTube video ids for Mini-Kinetics.
    # Obtain these files here: https://github.com/s9xie/Mini-Kinetics-200
    mini_kinetics_split_file = os.path.join(mini_kinetics_path, "{}_ytid_list.txt".format(split))

    # Text file containing label mapping, each line contains <cls_idx>,<cls_name>
    label_map_file = os.path.join(mini_kinetics_path, "label_mapping.txt")

    # Fetch all the HDF5 files
    filenames = glob.glob(os.path.join(hdf5_data_path, "*.h5"))
    labels = [0]*len(filenames)

    # MAYBE INSTEAD USE FROM_GENERATOR ??
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            read_examples_hdf5, [filename, label], [tf.uint8, tf.int64])),
        num_parallel_calls=4
    )

    dataset = dataset.repeat(NUM_EPOCHS)
    #dataset = dataset.map(common.video_utils.preprocess_video)

    # The parameter is the queue size
    #dataset = dataset.shuffle(1000 + 3 * BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    sess.run(init_op)

    while True:

        # Fetch a new batch from the dataset
        batch_videos, batch_labels = sess.run(next_batch)
        print("video shape: {}, video label: {}".format(batch_videos.shape, batch_labels[0]))