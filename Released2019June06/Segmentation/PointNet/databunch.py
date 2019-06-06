#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Description
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "CONG-MINH NGUYEN"
__copyright__ = "Copyright (C) 2019, HANDBOOK"
__credits__ = ["CONG-MINH NGUYEN"]
__license__ = "GPL"
__version__ = "1.0.1"
__date__ = "5/24/2019"
__maintainer__ = "CONG-MINH NGUYEN"
__email__ = "minhnc.edu.tw@gmail.com"
__status__ = "Development"  # ["Prototype", "Development", or "Production"]
# Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
# Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

# ==============================================================================
# Imported Modules
# ==============================================================================
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Union, Any
import copy

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # The GPU id to use, usually either "0" or "1"

import numpy as np

from GeneralUtils import onehot_encoding, onehot_decoding, unzip_list_of_tuple
from KerasUtils import kr
from PointCloudUtils import visualize_pc, coords_labels_to_pc, points_to_pc, down_sample_cloud, estimate_cloud_normals, compute_cloud_feature, sample_point_cloud_feature


# ==============================================================================
# Global Variable Definitions
# ==============================================================================
class DataBunch():
    def __init__(self, data_dir: Union[str, Path], categories, n_points: int, n_classes: int, n_channels: int,
                 bs: int = 32,
                 normalizer=None,
                 split_ratio: float = 0.8,
                 balanced: bool = False,
                 shuffle: bool = False, seed: int = 0):
        # ─── SAVE PARAMETERS ─────────────────────────────────────────────
        self.dir = data_dir
        self.cats = categories

        self.n_points = n_points
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.normalizer = normalizer

        self.bs = bs
        self.split_ratio = split_ratio
        self.balanced = balanced
        self.shuffle = shuffle
        self.seed = seed
        # ─────────────────────────────────────────────────────────────────

        # ─── FILTER VALID CATEGORIES AND FIND THOSE RELATIVE PATHS TO DATA 
        # Find all categories declared in 'synsetoffset2category.txt' and those relative path to data
        cat_file = os.path.join(self.dir, 'synsetoffset2category.txt')  # category file
        cat_dict = {}
        with open(cat_file, 'r') as f:  # get list of category/path in the cat_list_file
            for line in f:
                [category, path] = line.strip().split()
                cat_dict[category] = path  # 'category' information is saved in 'folder'
        # Only keep and Make dict of categories that appear in cat_files
        if not self.cats is None:
            cat_dict = {category: path for category, path in cat_dict.items() if category in self.cats}
        # ─────────────────────────────────────────────────────────────────

        # ─── ATTAIN DATA PATHS AND SPLIT INTO TRAIN AND VAL SETS ─────────
        self.trn_paths = []
        self.val_paths = []
        for item in cat_dict:
            points_path = os.path.join(self.dir, cat_dict[item], "points")  # path to points folder
            labels_path = os.path.join(self.dir, cat_dict[item], "points_label")  # path to labels folder

            filenames = [file for file in sorted(os.listdir(points_path))]
            if shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(filenames)

            trn_filenames = filenames[:int(len(filenames) * self.split_ratio)]
            val_filenames = filenames[int(len(filenames) * self.split_ratio):]

            for fn in trn_filenames:
                token = (os.path.splitext(os.path.basename(fn))[0])
                pts_file = os.path.join(points_path, token + '.pts')
                seg_file = os.path.join(labels_path, token + '.seg')
                self.trn_paths.append((item, pts_file, seg_file))
            for fn in val_filenames:
                token = (os.path.splitext(os.path.basename(fn))[0])
                pts_file = os.path.join(points_path, token + '.pts')
                seg_file = os.path.join(labels_path, token + '.seg')
                self.val_paths.append((item, pts_file, seg_file))
        # ───  ────────────────────────────────────────────────────────────

        self.trn_generator = Generator(data=self.trn_paths, n_classes=self.n_classes, n_points=self.n_points,
                                       n_channels=self.n_channels, bs=self.bs,
                                       normalizer=self.normalizer, balanced=self.balanced)
        self.val_generator = Generator(data=self.val_paths, n_classes=self.n_classes, n_points=self.n_points,
                                       n_channels=self.n_channels, bs=self.bs,
                                       normalizer=self.normalizer, balanced=self.balanced)


class Generator(kr.utils.Sequence):
    def __init__(self, data, n_classes: int = 10, n_points: int = 2048, n_channels: int = 6, bs=64, normalizer=None,
                 balanced=False):
        _, self.x, self.y = unzip_list_of_tuple(l=data)
        self.bs = bs
        self.n_points = n_points
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.normalizer = normalizer
        self.balanced = balanced

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.bs)))

    def __getitem__(self, idx):
        left_bound = idx * self.bs
        right_bound = (idx + 1) * self.bs

        if right_bound > len(self.x):
            right_bound = len(self.x)

        batch_x = []
        batch_y = []
        for i in range(right_bound - left_bound):
            points = np.loadtxt(self.x[left_bound + i]).astype('float32')
            labels = np.loadtxt(self.y[left_bound + i]).astype('int')

            if self.normalizer is not None:
                points = self.normalizer(points)

            # if data is more than required number of points, we should select different points when sample
            # else, we have to duplicate some data to get enough input points
            if len(points) > self.n_points:
                replace = False
            else:
                replace = True

            if self.balanced:
                # Balance loading points
                obj_idxs = np.argwhere(labels > 1).ravel()
                grd_idxs = np.argwhere(labels == 1).ravel()
                choice = np.array([])
                for loop in range(100):
                    taken_grd_idx = np.random.choice(a=grd_idxs, size=len(obj_idxs), replace=replace)
                    choice = np.hstack((choice, obj_idxs))
                    choice = np.hstack((choice, taken_grd_idx))
                    if len(choice) >= self.n_points:
                        break
                np.random.shuffle(choice)
                choice = choice[:self.n_points].astype(np.int)
            else:
                choice = np.random.choice(a=len(points), size=self.n_points, replace=replace)
            points = points[choice, :self.n_channels]
            labels = labels[choice]

            onehot_labels = onehot_encoding(labels=labels, n_classes=self.n_classes)

            batch_x.append(points)
            batch_y.append(onehot_labels)

        return np.array(batch_x), np.array(batch_y)


# ==============================================================================
# Function Definitions
# ==============================================================================
def normalize_points(points: np.ndarray):
    multiply_factor = np.asarray([1, 1, 1, 1 / 255., 1 / 255., 1 / 255.])
    offset_factor = np.asarray([0, 0, 0, 0, 0, 0])

    # * point cloud channels: X, Y, Z, R, G, B,....
    n_channels = np.min((points.shape[1], 6))
    return points * multiply_factor[:n_channels] + offset_factor[:n_channels]
    
def denormalize_points(points: np.ndarray):
    multiple_ratio = np.asarray([1, 1, 1, 1, 1, 1])
    subtract_ratio = np.asarray([0, 0, 0.677, 0, 0, 0])

    # * point cloud channels: X, Y, Z, R, G, B,....
    n_channels = np.min((points.shape[1], 6))
    return (points + subtract_ratio[:n_channels]) * multiple_ratio[:n_channels]


# ==============================================================================
# Main function
# ==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    # config_path = args.conf
    # with open(config_path) as config_buffer:    
    #     config = json.loads(config_buffer.read())

    '''**************************************************************
    I. Set parameters
    '''
    data_directory = 'E:/CLOUD/GDrive(t105ag8409)/data/shapenetcore_partanno_segmentation_benchmark_v0'
    n_points = 2048 * 16
    n_channels = 6
    num_epoches = 10
    bs = 2
    cat_choices = ['pipe_wrench']
    pretrained_model = None
    n_classes = 4

    '''**************************************************************
    II. Prepare the data
    '''
    databunch = DataBunch(data_dir=data_directory, categories=cat_choices, n_points=n_points, n_classes=n_classes,
                          n_channels=n_channels, bs=bs, normalizer=normalize_points, split_ratio=0.8,
                          balanced=False, shuffle=True, seed=0)
    batch_x, batch_y = databunch.trn_generator.__getitem__(3)
    print(batch_x.shape)
    print(batch_y.shape)
    print(batch_x[0][:4])
    # visualize_pc(coords_labels_to_pc(coords=batch_x[0][:, :3], labels=onehot_decoding(probs=batch_y[0], class_axis=1)))
    
    points1 = denormalize_points(batch_x[0])
    print(points1.shape)
    print(points1[:4])
    
    point_cloud1 = points_to_pc(points1)
    print(np.array(point_cloud1.points)[:4])
    print(np.array(point_cloud1.colors)[:4])
    # visualize_pc(point_cloud1)

    # downsampled_pc, fpfh_pc = sample_point_cloud_feature(point_cloud=point_cloud1, voxel_size=0.001)

    voxel_size = 0.001
    point_cloud_down_sample = down_sample_cloud(cloud=point_cloud1, voxel_size=voxel_size)
    visualize_pc(point_cloud_down_sample)

    radius_normal = voxel_size * 1
    point_cloud_normal = estimate_cloud_normals(cloud=point_cloud_down_sample, radius_normal=radius_normal)
    visualize_pc(point_cloud_normal)

    radius_feature = voxel_size * 5
    point_cloud_fpfh = compute_cloud_feature(cloud=point_cloud_normal, radius_feature=radius_feature)
    visualize_pc(point_cloud_fpfh)

    '''**************************************************************
    III. Create the model
    '''
    # 1: Build the model architecture.

    # 2: Load some weights into the model.

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.

    '''**************************************************************
    IV. Kick off the training
    '''
    # 1: Define model callbacks.

    # 2: Train model

    '''**************************************************************
    V. Test & Evaluate
    '''


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
