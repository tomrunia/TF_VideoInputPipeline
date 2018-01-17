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

import os
import glob
import multiprocessing

import numpy as np
import tensorflow as tf
import cv2

from cortex.vision.video_reader import VideoReaderOpenCV


################################################################################

def write_label_mapping(train_split_file, label_map_file):
    class_index = 1
    classes = []
    f_train_split = open(train_split_file, 'r')
    f_label_map   = open(label_map_file, 'w')
    f_train_split.readline()  # skip file header
    for line in f_train_split:
        parts = line.rstrip().split(',')
        class_label = parts[0].replace('"', '')
        if class_label not in classes:
            f_label_map.write("{},{}\n".format(class_index, class_label))
            classes.append(class_label)
            class_index += 1
    f_train_split.close()
    f_label_map.close()

def read_label_mapping(label_mapping_file):
    assert os.path.exists(label_mapping_file)
    label_mapping = {}
    with open(label_mapping_file) as f:
        for line in f:
            parts = line.rstrip().split(',')
            label_mapping[parts[1]] = int(parts[0])
    return label_mapping

def collect_all_videos(video_path):
    examples = []
    # Get all action categories (subdirectories)
    categories_dirs = [x[0] for x in os.walk(video_path)]
    categories_dirs = categories_dirs[1:] # skip current dir
    categories_dirs.sort()
    for cat_id, categories_dir in enumerate(categories_dirs):
        category = os.path.basename(categories_dir)
        # Get all videos in current action class (video files)
        video_files = glob.glob(os.path.join(categories_dir, "*.mp4"))
        video_files.sort()
        for vid_id, video_file in enumerate(video_files):
            video = {'category': category, 'video_file': video_file}
            examples.append(video)
    return examples

def read_example_list(split_list_file, label_mapping):
    print("Reading example from: {}".format(split_list_file))
    assert os.path.exists(split_list_file)
    examples = []
    with open(split_list_file, 'r') as f:
        f.readline()  # skip file header
        for line in f:
            parts = line.rstrip().split(',')
            class_label = parts[0].rstrip().replace('"', '')
            assert class_label in label_mapping.keys()
            example = {
                'video_id': parts[1],
                'frame_start': int(parts[2]),
                'frame_end': int(parts[3]),
                'class_id': label_mapping[class_label],
                'class_label': class_label,
            }
            examples.append(example)
    print("Found {} examples.".format(len(examples)))
    return examples

################################################################################

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def video_to_tfrecord(video_dir, example, resize_small_side=None):

    # Build the full video file path
    video_path = os.path.join(
        video_dir, example['class_label'], "{}_{:06d}_{:06d}.mp4".format(
            example['video_id'], example['frame_start'], example['frame_end']))

    # For some examples the video could not be downloaded...
    if not os.path.exists(video_path):
        return None

    # Open the video and set resizing option
    try:

        video = VideoReaderOpenCV(
            video_path, as_float=False, resize_small_side=resize_small_side)

    except ValueError:
        print("WARNING - video is unreadable: {}".format(video_path))
        return None

    # Read all the frames into memory (resized)
    frames = video.all_frames()

    features = {}
    features['num_frames']  = _int64_feature(frames.shape[0])
    features['height']      = _int64_feature(frames.shape[1])
    features['width']       = _int64_feature(frames.shape[2])
    features['channels']    = _int64_feature(frames.shape[3])
    features['filename']    = _bytes_feature(tf.compat.as_bytes(example['video_id']))
    features['class_label'] = _int64_feature(example['class_id'])
    for i in range(len(frames)):
        ret, buffer = cv2.imencode(".jpg", frames[i])
        features["frames/{:04d}".format(i)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    print("  Num. Frames: {}, Dimension: {}x{}, Class: {}, Label: {}".format(
        frames.shape[0], frames.shape[2], frames.shape[1],
        example['class_id'], example['class_label']))

    if frames.shape[0] < 64:
        raise RuntimeWarning("Video length is smaller than 64 frames...")

    return tf_example

def write_video_batch_tfrecords(video_dir, examples, output_dir,
                                examples_per_file, resize_small_side=None):

    num_tfrecord_files = int(np.ceil(len(examples)/examples_per_file))
    for tfrecord_idx in range(num_tfrecord_files):
        num_existing = len(glob.glob(os.path.join(output_dir, '*.tfrecords')))
        output_file = os.path.join(output_dir, "{:06d}.tfrecords".format(num_existing+1))
        with tf.python_io.TFRecordWriter(output_file) as writer:
            start = tfrecord_idx*examples_per_file
            end   = start+examples_per_file
            for i in range(start, min(end, len(examples))):
                # Convert video to tfrecord and write it to tfrecords
                example = video_to_tfrecord(video_dir, examples[i], resize_small_side)
                if example is not None:
                    writer.write(example.SerializeToString())


def convert_kinetics_to_tfrecords(kinetics_path, split, output_path,
                                  max_dim_resize=256, examples_per_file=20,
                                  num_workers=10, limit_examples=None):

    assert split in ('train', 'val', 'test')

    # Setup directories
    video_path = os.path.join(kinetics_path, "videos")
    split_list_file = os.path.join(kinetics_path, "splits", "kinetics_{}.csv".format(split))
    label_mapping_file = os.path.join(kinetics_path, "label_mapping.txt")

    # Create output directory
    output_path = os.path.join(output_path, split)
    if not os.path.exists(output_path): os.mkdir(output_path)

    # Read label mapping from file
    label_mapping = read_label_mapping(label_mapping_file)

    # Read 'train', 'val' or 'test' split from CSV file
    examples = read_example_list(split_list_file, label_mapping)

    if limit_examples is not None:
        examples = examples[0:limit_examples]

    num_tfrecord_files = int(np.ceil(len(examples)/examples_per_file))
    example_idx_offset = 0

    examples_per_worker = len(examples) // (max(num_workers, 2)-1)

    jobs = []
    for worker_idx in range(num_workers):
        start = worker_idx*examples_per_worker
        end   = start+examples_per_worker
        batch_examples = examples[start:end]

        p = multiprocessing.Process(
            target=write_video_batch_tfrecords,
            args=(video_path, batch_examples, output_path,
                  examples_per_file, resize_small_side,))
        jobs.append(p)
        p.start()


################################################################################

if __name__ == "__main__":

    kinetics_path = "/home/tomrunia/data/Kinetics/Full/"
    tfrecords_path = os.path.join(kinetics_path, "tfrecords")

    split = 'val'

    # First step: build the label mapping
    #write_label_mapping(split_list_file, label_name_file)

    # Resize smaller side to this dimension (px)
    resize_small_side = 256

    # Number of examples per TFRecords file
    examples_per_file = 100

    # Number of parallel threads
    number_of_workers = 10

    # Process a subset of the dataset
    limit_examples = 1000

    # Size computation of Kinetics dataset
    # Number of videos per tfrecords file
    # ~300 images per video, ~120 kilobytes per image (uncompressed) = 36Mb per video (!)
    # Kinetics has 300k videos => 11 Terabytes (!!)

    # Even with tfrecord's GZIP compression this is 1.6 Terabytes !!
    convert_kinetics_to_tfrecords(
        kinetics_path, split, tfrecords_path, resize_small_side,
        examples_per_file, number_of_workers, limit_examples)
