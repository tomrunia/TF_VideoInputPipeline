# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import sys
import glob
import time
import multiprocessing

import h5py
import numpy as np
import cv2
import jpeg4py as jpeg


import cortex.utils
from cortex.vision.video_reader import VideoReaderOpenCV

################################################################################

def collect_all_kinetics_videos(video_path):
    examples = {}
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
            filename, _ = os.path.splitext(os.path.basename(video_file))
            video_name = filename[0:11]
            examples[video_name] = video
    return examples

def parse_example_list(example_list_file):
    with open(example_list_file) as f:
        examples = f.readlines()
    examples = [x.strip() for x in examples]
    return examples

def collect_mini_kinetics_200(full_kinetics_path, example_list_file, label_mapping=None):
    kinetics_examples = collect_all_kinetics_videos(full_kinetics_path)
    mini_kinetics_examples = []
    mini_kinetics_list = parse_example_list(example_list_file)
    for i in range(len(mini_kinetics_list)):
        query = mini_kinetics_list[i]
        match = query in kinetics_examples
        if not match:
            print("Warning. Video not found in Full Kinetics: {}".format(query))
            continue
        video_info = kinetics_examples[query]
        video_info['video_idx'] = query
        if label_mapping is not None:
            video_info['class_index'] = label_mapping.index(video_info['category'])
        mini_kinetics_examples.append(video_info)
    print("Found {} examples of the {} total.".format(len(mini_kinetics_examples), len(mini_kinetics_list)))
    return mini_kinetics_examples

def write_label_mapping(examples, label_map_file):
    classes = [example['category'] for example in examples]
    classes = list(set(classes))
    classes.sort()
    with open(label_map_file, 'w') as f:
        for class_idx in range(len(classes)):
            f.write('{},{}\n'.format(class_idx, classes[class_idx]))

def read_label_mapping(label_map_file):
    label_map = []
    with open(label_map_file, 'r') as f:
        for line in f.readlines():
            cls_idx, cls_name = line.strip().split(',')
            label_map.append(cls_name)
    return label_map


def write_examples_hdf5(examples, hdf5_output_dir, index_first_file=0,
                        resize_small_side=256, jpg_quality=95,
                        examples_per_file=100):

    num_examples = len(examples)
    num_hdf5_files = int(np.ceil(num_examples/examples_per_file))
    thread_id = multiprocessing.current_process()._identity[0]

    for hdf5_file_idx in range(num_hdf5_files):

        # Build filename for the current HDF5 file
        hdf5_file = os.path.join(hdf5_output_dir, "{:06d}.h5".format(
            index_first_file+hdf5_file_idx))

        example_start = hdf5_file_idx*examples_per_file
        example_end   = min(len(examples), example_start+examples_per_file)
        examples_in_file = example_end-example_start

        # Start adding examples to the current HDF5 container
        with h5py.File(hdf5_file, 'w') as hf:

            # Initialize dataset for storing raw images
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))

            # Create datasets for storing labels, number of frames and success status
            dset_labels = hf.create_dataset("labels", shape=(examples_in_file,), dtype=int)
            dset_example_shapes = hf.create_dataset("example_shapes", shape=(examples_in_file,4), dtype=int)
            dset_success = hf.create_dataset("success", shape=(examples_in_file,), dtype=bool)

            for example_idx in range(example_start, example_end):

                example = examples[example_idx]
                video_filename = example['video_file']
                class_index = example['class_index']
                video_name = cortex.utils.basename(video_filename)

                try:

                    # Custom video reader that resizes video frames.
                    video = VideoReaderOpenCV(
                        filename=video_filename,
                        as_float=False,
                        resize_small_side=resize_small_side)

                except ValueError:
                    print("WARNING - video is unreadable: {}".format(video_filename))
                    continue

                # Create dataset storing frames for this video
                num_frames = video.length
                index_in_hdf5 = example_idx-example_start

                dset_frames = hf.create_dataset(
                    "video_{:06d}".format(index_in_hdf5), shape=(num_frames,), dtype=dt)

                # Store all the frames as JPG compressed images
                success = True
                size_uncompressed = 0
                size_compressed = 0

                for frame_idx in range(num_frames):

                    ret, frame = video.next_frame()
                    if not ret:
                        print("WARNING: unable to read video frame {}".format(frame_idx))
                        success = False
                        break

                    size_uncompressed += sys.getsizeof(frame)/1024

                    # Apply JPG compression to the raw video frame
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
                    frame_jpg_encode = cv2.imencode(".jpg", frame, encode_params)
                    frame_jpg_encode = frame_jpg_encode[1].tostring()
                    size_compressed += sys.getsizeof(frame_jpg_encode)/1024

                    # Save JPG compressed frame to the dataset
                    dset_frames[frame_idx] = np.fromstring(frame_jpg_encode, dtype='uint8')

                # Store example information
                dset_labels[index_in_hdf5] = example['class_index']
                dset_example_shapes[index_in_hdf5,:] = [num_frames, frame.shape[0], frame.shape[1], frame.shape[2]]
                dset_success[index_in_hdf5] = success

                compression_rate = 100-(size_compressed/size_uncompressed)*100.0
                print("[Thread {:02d}] Container {:04d}/{:04d}. Video {:04d}/{:04d}. Frames: {}. Before Compression: {:.0f}MB. "
                      "After Compression: {:.0f}MB. Compression: {:.1f}%".format(
                    thread_id, hdf5_file_idx, num_hdf5_files,
                    example_idx+1, num_examples, num_frames, size_uncompressed/1024,
                    size_compressed/1024, compression_rate))

def convert_mini_kinetics_to_hdf5(full_kinetics_path, mini_kinetics_list_file,
                                  label_mapping_file, hdf5_output_path,
                                  resize_small_side=256, jpg_quality=90,
                                  examples_per_hdf5=100, num_workers=10):

    label_mapping = read_label_mapping(label_mapping_file)
    examples = collect_mini_kinetics_200(full_kinetics_path, mini_kinetics_list_file, label_mapping)

    # Calculate number of examples per worker (upround to closest multiple)
    examples_per_worker = len(examples)/num_workers
    examples_per_worker  = examples_per_worker + (examples_per_hdf5/2)
    examples_per_worker -= examples_per_worker % examples_per_hdf5

    jobs = []
    for worker_idx in range(num_workers):

        file_index_offset = worker_idx*int(examples_per_worker/examples_per_hdf5)
        start = int(worker_idx*examples_per_worker)
        end   = start+int(examples_per_worker)

        p = multiprocessing.Process(
            target=write_examples_hdf5,
            args=(examples[start:end], hdf5_output_path, file_index_offset,
                  resize_small_side, jpg_quality, examples_per_hdf5,))
        jobs.append(p)
        p.start()

def validate_hdf5_dataset(full_kinetics_path, mini_kinetics_list_file, hdf5_path):

    hdf5_files = glob.glob(os.path.join(hdf5_path, "*.h5"))
    examples = collect_mini_kinetics_200(full_kinetics_path, mini_kinetics_list_file)

    num_success = 0
    num_failure = 0
    all_labels = []

    for hdf5_file in hdf5_files:
        print(hdf5_file)
        with h5py.File(hdf5_file, 'r') as hf:
            labels = hf['labels'].value
            success = hf['success'].value

            print(len(success))

            num_success += int(np.sum(success))
            print(success)
            num_failure += int(np.sum(np.invert(success)))
            all_labels  += list(labels)

    print("#"*80)
    print("Num Total:   {}".format(num_success+num_failure))
    print("Num Success: {}".format(num_success))
    print("Num Failure: {}".format(num_failure))
    print("List file contains {} examples: ".format(len(examples)))
    print("Missing: {}".format(len(examples)-num_success))
    print("#"*80)
    print("Label Distribution:")
    print(np.bincount(all_labels))

def read_examples_hdf5(hdf5_file, sample_length=64,
                       sample_height=224, sample_width=224):

    start = time.process_time()
    num_examples = 0

    with h5py.File(hdf5_file, 'r') as hf:

        labels = hf['labels'].value
        example_shapes = hf['example_shapes'].value
        success = hf['success'].value
        num_examples = int(np.sum(success))

        for example_idx in range(num_examples):

            if not success[example_idx]: continue

            vid_length, vid_height, vid_width, vid_channels = \
                example_shapes[example_idx]

            if vid_length < sample_length:
                print("Video too short...")
                #continue

            offset_time   = np.random.randint(0, max(vid_length-sample_length, 1))
            offset_height = np.random.randint(0, vid_height-sample_height)
            offset_width  = np.random.randint(0, vid_width-sample_width)

            frames = np.zeros((min(sample_length, vid_length), sample_height, sample_width, vid_channels), np.uint8)

            for frame_idx in range(offset_time, min(offset_time+sample_length, vid_length)):

                # Read encoded JPG as bytes string
                dset_name = "video_{:06d}".format(example_idx)
                frame_jpg_encode = np.fromstring(hf[dset_name][frame_idx], dtype=np.uint8)

                # Using libturbojpeg is much faster than cv2.decode()
                frame_decode = jpeg.JPEG(frame_jpg_encode).decode(pixfmt=jpeg.TJPF_BGR) # BGR
                #frame_decode = cv2.imdecode(frame_jpg_encode, -1) # BGR

                # Cropping the frames here as almost no speed penalty
                frames[frame_idx-offset_time] = \
                    frame_decode[offset_height:offset_height+sample_height,
                                 offset_width:offset_width+sample_width,:]

            # If the video is too short, loop it to satisfy the length
            # This is directly copied from Carreira (CVPR, 2017)
            if vid_length < sample_length:
                num_reps = int(np.ceil(sample_length/vid_length))
                frames = np.tile(frames, reps=(num_reps,1,1,1))
                frames = frames[0:sample_length,]

    end = time.process_time()
    duration = end-start
    example_per_second = num_examples/duration
    print("Read and decoded {} JPG images from HDF5 in {:.2f} seconds, "
          "{:.2f} examples/seconds".format(
        num_examples, duration, example_per_second))