'''
    File name: HANDBOOK
    Author: minhnc
    Date created(MM/DD/YYYY): 12/10/2018
    Last modified(MM/DD/YYYY HH:MM): 12/10/2018 11:18 AM
    Python Version: 3.6
    Other modules: [None]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
    Orbbec camera calibration tool at: https://3dclub.orbbec3d.com/t/astra-s-intrinsic-and-extrinsic-parameters/302/4
                                   or: https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622
    Orbbec Astra S FoV: 60° horiz x 49.5° vert. (73° diagonal) (https://orbbec3d.com/product-astra/)
    Camera intrinsic: http://ksimek.github.io/2013/08/13/intrinsic/
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import open3d as op3
import copy

from primesense import openni2
from primesense import _openni2 as c_api

import cv2

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--np', type=int, default=8192, help='number of input points(size of input point cloud)')
parser.add_argument('--ptn', type=str, default='./tmp/seg_model_94_0.944126.pth', help='patch of pre-trained model')
parser.add_argument('--idx', type=int, default=0, help='model index')

#==============================================================================
# Function Definitions
#==============================================================================
def load_ply(path, num_points=-1):
    pointcloud = op3.read_point_cloud(path)
    scene_array = np.asarray(pointcloud.points)

    if num_points > 0:
        choice = np.random.choice(a=len(scene_array), size=num_points, replace=True)
        scene_array = scene_array[choice, :]

    return scene_array

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is PointNet-Segmentation and Surface-Matching Program')

    opt = parser.parse_args()

    # Load model
    source_path = 'E:/PROJECTS/NTUT/PointNet/robotic-grasp_itri/models/modelMay10/'
    saving_folder_path = source_path + 'info/'
    if not os.path.exists(saving_folder_path):
        os.makedirs(saving_folder_path)
    '''
    Record
    '''
    # end_idx = 11
    # pc_file_old = source_path + 'PC/pipe/' + str(9000 + end_idx) + '.ply'
    # pc_model_old = load_ply(path=pc_file_old, num_points=-1)
    # for i in range(end_idx):
    #     idx = end_idx-i-1 + 9000
    #
    #     pc_file_new = source_path + 'PC/pipe/' + str(idx) + '.ply'
    #
    #     pc_model_new = load_ply(path=pc_file_new, num_points=-1)
    #
    #     result_icp = match_surface(model_points=pc_model_old, object_points=pc_model_new)
    #     pc = PointCloud()
    #     pc.points = Vector3dVector(pc_model_old)
    #     pc.transform(result_icp.transformation)
    #     pc_model_old = np.concatenate([pc_model_new, np.asarray(pc.points)])



    # grasping_poses = [[[-265, 330, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-265, 280, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-265, 230, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #
    #                   [[-215, 380, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]], # start second line
    #                   [[-215, 330, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-215, 280, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-215, 230, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-215, 180, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]], # end of second line
    #
    #                   [[-165, 380, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]], # start center line
    #                   [[-165, 330, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-165, 280, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]], # center model
    #                   [[-165, 230, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-165, 180, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]], # end of center line
    #
    #                   [[-115, 380, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],  # start fourth line
    #                   [[-115, 330, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-115, 280, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],  # center model
    #                   [[-115, 230, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-115, 180, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],  # end of fourth line
    #
    #                   [[-65, 330, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-65, 280, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]],
    #                   [[-65, 230, -15, 180, 0, 0], [-130, 325, -15, 180, 0, 90]]
    #                   ]
    grasping_poses = [[[-180, 256, -20, 180, 0, 0, 0, 75]],
                      [[-187, 265, -18, 180, 0, 0, 1, 70],
                       [-147, 227, -18, 180, 0, -90, 1, 70]]
                      ]
    grasping_poses = np.array(grasping_poses)

    files = os.listdir(source_path + 'pc/')
    for i, file in enumerate(files):
        filename, ext = os.path.splitext(file)

        pc = load_ply(path=source_path + '/pc/' + file, num_points=-1)
        centroid = np.mean(pc, axis=0)

        np.save(saving_folder_path + filename, [centroid, grasping_poses[i]])

    np.load(saving_folder_path + filename + '.npy', allow_pickle=True)

if __name__ == '__main__':
    main()
