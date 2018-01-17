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

from mini_kinetics.mini_kinetics import *

'''

- This script can be used for converting the Mini-Kinetics dataset to HDF5.
- Examples are written as encoded JPG images to the HDF5 containers
- We use several threads to speed-up the extraction process. 

It requires the full Kinetics dataset to be on your disk:
  https://deepmind.com/research/open-source/open-source-datasets/kinetics/
  
Also download the split files associated with Mini-Kinetics:
  https://github.com/s9xie/Mini-Kinetics-200

'''

################################################################################

split = 'val'
full_kinetics_path = "/home/tomrunia/data/Kinetics/Full/videos"
mini_kinetics_path = "/home/tomrunia/data/Kinetics/Mini-200/"
hdf5_output_path = os.path.join(mini_kinetics_path, 'videos_as_hdf5', split)

# File that contains YouTube video ids for Mini-Kinetics.
# Obtain these files here: https://github.com/s9xie/Mini-Kinetics-200
mini_kinetics_split_file = os.path.join(mini_kinetics_path, "{}_ytid_list.txt".format(split))

# Text file containing label mapping, each line contains <cls_idx>,<cls_name>
label_map_file = os.path.join(mini_kinetics_path, "label_mapping.txt")

# Resize smaller side to this dimension (px)
resize_small_side = 256

# Examples per HDF5 containiner
examples_per_hdf5 = 100

# Quality to write JPG files to HDF5 container
jpg_quality = 90

# Number of parallel threads to convert the dataset
num_workers = 10

################################################################################

# 1. Use this to first write the label mapping of the Mini-Kinetics dataset
#write_label_mapping(mini_kinetics_examples, label_map_file)

# 2. To read the label mapping from disk, use this function
label_mapping = read_label_mapping(label_map_file)

# 3. This returns the list of all examples in the Mini-Kinetcs dataset
#mini_kinetics_examples = collect_mini_kinetics_200(
#    full_kinetics_path, mini_kinetics_split_file, label_mapping)

# 4. Main function to convert the entire Mini-Kinetics dataset to HDF5
#    containers each with 'example_per_hdf5' files.

convert_mini_kinetics_to_hdf5(
    full_kinetics_path=full_kinetics_path,
    mini_kinetics_list_file=mini_kinetics_split_file,
    label_mapping_file=label_map_file,
    hdf5_output_path=hdf5_output_path,
    jpg_quality=jpg_quality,
    examples_per_hdf5=examples_per_hdf5,
    num_workers=num_workers
)

# 5. After converting the dataset to HDF5, this can be used to check it
validate_hdf5_dataset(full_kinetics_path, mini_kinetics_split_file, hdf5_output_path)
