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
__date__ = "5/10/2019"
__maintainer__ = "CONG-MINH NGUYEN"
__email__ = "minhnc.edu.tw@gmail.com"
__status__ = "Development"  # ["Prototype", "Development", or "Production"]
# Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
# Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

#==============================================================================
# Imported Modules
#==============================================================================
import argparse
from pathlib import Path
import os.path
import sys
import time
import copy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

import json
import numpy as np
import cv2
import requests

from Camera.OrbbecAstraS.camera import Camera, rgbd_to_pointcloud

from GeneralUtils import List, Tuple, Dict, Union, Generic, TypeVar
from GeneralUtils import sample_arrays, stack_list_horizontal
from PointCloudUtils import visualize_pc, points_to_pc, coords_labels_to_pc, load_ply_as_pc, load_ply_as_points
from PointCloudUtils import adjust_pc_coords, global_icp
from PointCloudUtils import radian2degree, degree2radian, m2mm, mm2m, create_rotx_matrix, create_roty_matrix, create_rotz_matrix, create_tranl_matrix
from Segmentation.PointNet.learner import PointNetLearner

#==============================================================================
# Constant Definitions
#==============================================================================


#==============================================================================
# Function Definitions
#==============================================================================
def mpose2mmpose(pose: np.ndarray):
    tarr = np.ones(len(pose))
    tarr[:3] *= 1000
    return pose * tarr

def mmpose2mpose(pose: np.ndarray):
    tarr = np.ones(len(pose))
    tarr[:3] *= 0.001
    return pose * tarr

def load_object_models(model_path='./obj_models/modelMay10/'):
    """
    Description:
    :param model_path: str, path to the reference models of known objects
    :return: pc_models, List[2L ndarrays], list of points of target surface
    :return: centroid_models, List[Vector(3 floats)], the list of centroids of model
    :return: pose_models, List[List[Vector(6 floats)]], the list of pose list of each model(each model has a list of poses)
    """
    pc_models = []
    centroid_models = []
    pose_models = []
    files = os.listdir(path=os.path.join(model_path, 'pc/'))
    for _, file in enumerate(files):
        filename, _ = os.path.splitext(file)

        pc_model = load_ply_as_points(file_path=os.path.join(model_path, 'pc/', file))
        centroid, grasping_pose = np.load(os.path.join(model_path, 'info/', filename + '.npy'), allow_pickle=True)
        grasping_pose = np.array(grasping_pose).astype(float)
        grasping_pose[:, :3] = mm2m(grasping_pose[:, :3])

        pc_models.append(pc_model)
        centroid_models.append(centroid)
        pose_models.append(grasping_pose)
    return pc_models, centroid_models, pose_models


def measure_xtran_params(neutral_point, transformation):
    """
    Description: Assume that the transformation from robot coord to camera coord is: RotX -> RotY -> RotZ -> Tranl
                 In this case: RotX = 180, RotY = 0; RotZ = -90; Tranl: unknown
                 But we know coords of a determined neutral point in 2 coord systems,
                 hence we can measure Transl from robot centroid to camera centroid.(Step 2)
    :param neutral_point    : Dict, list of 2 coords of neutral_point in 2 coord systems
    :param transformation   : Dict, list of 3 rotating transformations
    :return: r2c_xtran  : Matrix 4x4 floats, transformation from robot coord to camera coord
    :return: c2r_xtran  : Matrix 4x4 floats, transformation from camera coord to robot coord
    # :return: tranl      : Matrix 4x4 floats, translation from robot coord to camera coord
    """
    # 1: Load coords of the neutral point
    neutral_robot = mm2m(coords=np.array(neutral_point['robot_coord'])) # neutral point coord in robot coord system
    neutral_camera = mm2m(coords=np.array(neutral_point['camera_coord'])) # neutral point coord in camera coord system

    rotx = create_rotx_matrix(theta=-transformation['rotx']) # load transformation matrix of rotation around x
    roty = create_roty_matrix(theta=-transformation['roty']) # load transformation matrix of rotation around y
    rotz = create_rotz_matrix(theta=-transformation['rotz']) # load transformation matrix of rotation around z

    # 2: Find transformation between robot coord centroid and camera coord centroid
    rotxyz = np.dot(np.dot(rotz, roty), rotx) # determine transformation matrix after rotate sequently around x, y, z
    neutral_robot3 = np.dot(rotxyz, np.append(neutral_robot, 1))[:3] # find coord of neutral point after RotXYZ
    Oc_in_3 = neutral_robot3 - neutral_camera # find coord of robot centroid in camera coord system

    tranl = create_tranl_matrix(vector=-Oc_in_3)

    # 3: Find transformation matrix from robot to camera
    # r2c_xtran = np.dot(np.dot(np.dot(tranl, rotz), roty), rotx)
    # c2r_xtran = np.linalg.inv(r2c_xtran)

    return rotx, roty, rotz, tranl


def input_cli():
    user_input = input("Enter CLI commands such as (--NAME VALUE ...): ")
    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument('-vb', '--verbose', type=bool, help='show detail results')
    custom_parser.add_argument('-vs', '--voxel_size', type=float, help='adjust voxel size')
    custom_parser.add_argument('-ft', '--fitness_threshold', type=float, help='adjust voxel size')
    custom_parser.add_argument('-pi', '--selected_pose_id', type=int, help='select pose id that will execute grasp')
    custom_args = custom_parser.parse_args(user_input.split())
    return custom_args


def normalize_pc(points: np.ndarray):
    new_points = copy.deepcopy(points)
    new_points[:, 2] -= 0.677
    new_points[:, 3:6] /= 255.
    return new_points


def segment_obj_in_scene(scene_points, n_points: int=16384, n_channels: int=6, url='http://127.0.0.1:5000/api/'):
    """
    Description: segment the point clouds of wrench and pipe out of scene
    :param learner      : Object, a PointNet Learner that's able to do predict point-wise classification
    :param scene_points : 2L ndarray(shape=(n_points, n_channels)), list of points
    :param n_points     : int > 0, number input points of PointNet Learner
    :param n_channels   : int > 0, number channels of input points of PointNet Learner
    :return: wrench_points  : 2L ndarray, points of wrench
    :return: pipe_points    : 2L ndarray, points of pipe
    """
    # Shuffle points to distribute the points equally in arrays(useful for next step, cut scene into parts to segment)
    n_scene_points = len(scene_points)
    scene_points = sample_arrays(arrs=scene_points, n_samples=n_scene_points)

    # Do segment(cut scene into 2 parts, segment each part then unify results of 2 parts to get overall picture)
    wrench_points = []
    pipe_points = []
    for i in range(2):
        # sample the points to fit the network
        cur_scene_points = scene_points[i * n_scene_points // 2:(i + 1) * n_scene_points // 2]
        cur_scene_points = sample_arrays(arrs=cur_scene_points, n_samples=n_points)

        # predict segment labels(send data to remote server through RESTful API)
        # pred_labels = learner.predict(x=normalize_pc(points=cur_scene_points[:, :n_channels]))
        data = {'points': normalize_pc(points=cur_scene_points[:, :n_channels]).tolist()}
        j_data = json.dumps(data)
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        res = requests.post(url=url, data=j_data, headers=headers)
        pred_labels = np.asarray(json.loads(res.text))

        # extract the points in the scene of each object by labels
        wrench_points.append(cur_scene_points[pred_labels == 2])
        pipe_points.append(cur_scene_points[pred_labels == 3])

    wrench_points = np.vstack(wrench_points) # get entire points of wrench
    pipe_points = np.vstack(pipe_points) # get entire points of pipe

    # visualize_pc(coords_labels_to_pc(coords=cur_scene_points[:, :3], labels=pred_labels))

    return wrench_points, pipe_points


def match_object_surface(surface: np.ndarray, model: np.ndarray, model_centroid: Tuple[float, float, float],
                         voxel_size: float, n_channel: int=6, verbose: bool=False):
    """
    Description:
    :param surface          : 2L ndarray(shape=(n_points, n_channels)), list of points of target surface
    :param model            : 2L ndarray(shape=(n_points, n_channels)), list of points of target surface
    :param model_centroid   : Vector(3 floats), the centroid of `model`
    :param voxel_size       : float, default=0.6, downsampling size of point cloud in `global_icp` algorithm
    :param n_channel        : int > 0, number channels of input points of PointNet Learner
    :param verbose          : bool, show detail results and notification or not
    :return: TYPE, MEAN
    """
    point_cloud_model = adjust_pc_coords(point_cloud=points_to_pc(model[:, :n_channel]), coord=model_centroid)
    point_cloud_target = adjust_pc_coords(point_cloud=points_to_pc(surface[:, :n_channel]), coord=model_centroid)
    xtran = global_icp(source=points_to_pc(point_cloud_model), target=points_to_pc(point_cloud_target),
                       voxel_size=voxel_size, verbose=verbose)
    print(xtran)
    return xtran


def interpolate_pose(ref_pose, surf_xtran, rotx, roty, rotz, tranl, pc_centroid):
    """
    Description: match reference_pose of (x, y, z) (rx, ry, rz) and (mode, aperture) from reference source to target point cloud
    :param ref_pose         : Vector(8 floats), the pose of the reference model
    :param surf_xtran       : Matrix(4x4 floats), the transformation matrix from source model to target point cloud
    :param rotx             : Matrix(4x4 floats), the transformation matrix of rotation around x axis of robot coord
    :param roty             : Matrix(4x4 floats), the transformation matrix of rotation around y axis of robot coord
    :param rotz             : Matrix(4x4 floats), the transformation matrix of rotation around z axis of robot coord
    :param tranl            : Matrix(4x4 floats), the transformation matrix of translation from robot origin to the camera origin
    :param pc_centroid      : Matrix(4x4 floats), the centroid of considered point cloud
    :return: Vector(6 floats), the pose in robot system
    """
    # transformation matrix of robot origin to point cloud center, xyz elements
    tranl2 = create_tranl_matrix(vector=-np.array(pc_centroid))
    r2pc_xyz_xtran = np.dot(np.dot(np.dot(np.dot(tranl2, tranl), rotz), roty), rotx)
    pc2r_xyz_xtran = np.linalg.inv(r2pc_xyz_xtran)

    # measure xyz
    new_xyz = np.append(arr=ref_pose[:3], values=1, axis=None)
    new_xyz = np.dot(r2pc_xyz_xtran, new_xyz)
    new_xyz = np.dot(surf_xtran, new_xyz)
    new_xyz = np.dot(pc2r_xyz_xtran, new_xyz)

    # measure roll-pitch-yaw
    # new_rpy = ref_pose[3:6] + radian2degree(rotation_matrix_to_euler_angles(surf_xtran[:3, :3]))
    new_yaw = ref_pose[5] + radian2degree(rotation_matrix_to_euler_angles(surf_xtran[:3, :3]))[2] # restrict rx, ry because of real robot problem

    new_pose = copy.deepcopy(ref_pose)
    new_pose[:3] = new_xyz[:3]
    # new_pose[3:6] = new_rpy[:3]
    new_pose[5] = new_yaw
    
    return new_pose


def rgbd_to_pointcloud1(rgb, depth, scale=0.001, focal_length_x=520, focal_length_y=513, label=False, offset=0, **kwargs):
    """
    Convert single RGBD image to point cloud
    :param rgb: 3L ndarray of int, RGB image
    :param depth: 1L ndarray of any, depth image
    :param scale: a float value, scale=0.001->scale into Meter unit, scale=1->scale into miliMeter unit
    :param focal_length_x: a float value, focal_length of x axis
    :param focal_length_y: a float value, focal_length of y axis
    :param label: a bool value, enable or disable labeling data
    :param **kwargs: a list of 3L ndarray of int, list of label tables
                     this arguments are only used when 'label' is set True
                     size(h, w) of each label table must equal size of rgb image
    :return: a list of points as [X, Y, Z, label(optional)]
    """
    center_y, center_x = (rgb.shape[0] - 1) / 2, (rgb.shape[1] - 1) / 2
    points = []
    for row in range(rgb.shape[0]):
        for col in range(rgb.shape[1]):
            R, G, B = rgb[row, col]

            # obtain z value and ignore the un-obtained point(z=0)
            Z = depth[row, col]
            if Z == 0: continue

            # measure world coordinates in Meter(scale=0.001) or miliMeter(scale=1)
            Z = Z * scale
            X = (col - center_x) * Z / focal_length_x
            Y = (row - center_y) * Z / focal_length_y

            # label the point if input the label table(in kwargs)
            if label:
                label_point = offset
                for i, (mask_name, label_table) in enumerate(list(kwargs.items())):
                    if label_table[row, col] > 0:
                        label_point += i+1
                points.append([X, Y, Z, R, G, B, row, col, label_point])
            else:
                points.append([X, Y, Z, R, G, B, row, col])

    return np.asarray(points)


import math
# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R: np.array) -> bool:
    """
    Check???
    :param R: a matrix of 4x4
    :return: a boolean, ???
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotation_matrix_to_euler_angles(R):
    """
    Measure rotations around x, y and z from transformation matrix
    :param R: a rotation matrix
    :return: an array of 3 values that describe rotations around x, y and z axis, unit is "radian"
    """
    assert (is_rotation_matrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    '''**************************************************************
    I. Set parameters
    '''
    ## Read config
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    ## Load config to variables
    lbl_names = config['seg_net']['labels']
    n_points = config['seg_net']['n_points']
    n_channels = config['seg_net']['n_channels']
    ptn_model = config['seg_net']['ptn_model']

    redist = config['camera']['astra_redist']
    ## Interpolate some other variables
    n_classes = len(lbl_names)
    input_shape = [n_points, n_channels]
    print("Show parameters \n\tlabels: {} \n\tn_classes: {} \n\tn_points: {} \n\tn_channels: {} \n\tptn_model: {} ".format(lbl_names, n_classes, n_points, n_channels, ptn_model))

    '''**************************************************************
    II. Load segmentation network model
    '''
    # learner = PointNetLearner(input_shape=input_shape, labels=lbl_names)
    # learner.load_weight(weight_path=ptn_model)

    '''**************************************************************
    III. Initialize hardwares: camera and robot
    '''
    # Initialize camera
    camera = Camera(resolution=(640, 480), fps=30, redist=redist)

    url_tx40 = 'http://localhost:5001/v1/tx40/'
    url_tfg = 'http://localhost:5002/v1/tfg/'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}  # config header of REST API
    res = requests.put(url=url_tx40 + 'state/', data=json.dumps(True), headers=headers)
    res = requests.put(url=url_tfg + 'state/', data=json.dumps(True), headers=headers)

    '''**************************************************************
    IV. Load object models
    '''
    pc_models, centroid_models, pose_models = load_object_models(model_path=config['grasp_infer']['model_path'])
    rotx, roty, rotz, tranl = measure_xtran_params(neutral_point=config['grasp_infer']['coords']['neutral_point'],
                                                   transformation=config['grasp_infer']['transformation'])

    '''**************************************************************
    V. Record
    '''
    from OpenCVUtils import draw_xy_axes
    camera.start()
    done = False
    verbose = True
    voxel_size = 0.008
    fitness_threshold = 0.8
    selected_pose_id = 0
    detected_poses = []
    origin = []
    while not done:
        rgb, depth_distance = camera.get_stream(crop=(180, 200))
        camera.display(crop=(180, 200))
        key = cv2.waitKey(1) & 255
        if key == 27 or chr(key) == 'q':  # terminate system
            print("\tESC key detected!")
            done = True

        elif chr(key) == 'v': # on/off verbose
            verbose = not verbose
            print("Verbose was toggled {}".format('ON' if verbose else 'OFF'))

        elif chr(key) == 'a': # adjust parameters
            custom_args = input_cli()
            if custom_args.verbose: verbose = custom_args.verbose
            if custom_args.voxel_size: voxel_size = custom_args.voxel_size
            if custom_args.fitness_threshold: fitness_threshold = custom_args.fitness_threshold
            if custom_args.selected_pose_id: selected_pose_id = custom_args.selected_pose_id

        elif chr(key) == 's':  # segment objects and infer grasp poses
            print("==================================================")
            #STEP: 1. Record and convert to point cloud
            start_time = time.time()
            scene_points = rgbd_to_pointcloud1(rgb=rgb, depth=depth_distance, scale=0.001,
                                              focal_length_x=520, focal_length_y=513, label=False, offset=1)
            print("Time recording: {}".format(time.time() - start_time))
            # scene_point_cloud = PointCloud(points=scene_points[:, :3]); scene_point_cloud.visualize()

            #STEP: 2. Segment object
            start_time = time.time()
            wrench_points, pipe_points = segment_obj_in_scene(scene_points=scene_points,
                                                              n_points=n_points, n_channels=n_channels)
            print("Time segmenting: {}".format(time.time() - start_time))

            #STEP: 3. Match surface and interpolate poses
            start_time = time.time()
            detected_poses = []
            while (detected_poses == []) and ((len(wrench_points) > 1000) or (len(pipe_points) > 1000)):
                #* Wrench
                if len(wrench_points) > 1000: # the area of detected wrench must be big enough
                    # Match surface
                    surf_xtran = match_object_surface(surface=wrench_points[:, :n_channels], model=pc_models[0][:, :n_channels],
                                                      model_centroid=centroid_models[0], voxel_size=voxel_size, verbose=verbose)

                    # Interpolate grasp poses
                    if surf_xtran.fitness > fitness_threshold:
                        for i in range(len(pose_models[0])):
                            print(mpose2mmpose(pose_models[0][i]).astype(int))
                            pose = interpolate_pose(ref_pose=pose_models[0][i], surf_xtran=surf_xtran.transformation,
                                                     rotx=rotx, roty=roty, rotz=rotz, tranl=tranl, pc_centroid=centroid_models[0])
                            print(mpose2mmpose(pose).astype(int))

                            pose_camera_coord = np.dot(np.dot(np.dot(np.dot(tranl, rotz), roty), rotx), np.append(arr=pose[:3], values=1, axis=None))
                            dis = scene_points[:, :3] - pose_camera_coord[:3]
                            dis = np.sum(dis * dis, axis=1)
                            if (origin == []): origin = scene_points[np.argmin(dis), 6:8]

                            detected_poses.append([pose, 1])

                #* Pipe
                if len(pipe_points) > 1000: # the area of detected pipe must be big enough
                    # Match surface
                    surf_xtran = match_object_surface(surface=pipe_points[:, :n_channels], model=pc_models[1][:, :n_channels],
                                                      model_centroid=centroid_models[1], voxel_size=voxel_size, verbose=verbose)

                    # Interpolate grasp poses
                    if surf_xtran.fitness > fitness_threshold:
                        for i in range(len(pose_models[1])):
                            print(mpose2mmpose(pose_models[1][i]).astype(int))
                            pose = interpolate_pose(ref_pose=pose_models[1][i], surf_xtran=surf_xtran.transformation,
                                                     rotx=rotx, roty=roty, rotz=rotz, tranl=tranl, pc_centroid=centroid_models[1])
                            print(mpose2mmpose(pose).astype(int))
                            detected_poses.append([pose, 2])
                break
            print("Time matching: {}".format(time.time() - start_time))

        elif chr(key) == 'g': # execute grasping
            if len(detected_poses) > 0:
                x, y, z, rx, ry, rz, mode, aperture = mpose2mmpose(detected_poses[selected_pose_id][0])
                if rx > 180: rx = 180
                if ry > 180: ry = 180
                res = requests.put(url=url_tfg + 'mode/', data=json.dumps(mode), headers=headers)
                res = requests.put(url=url_tfg + 'position/',
                                   data=json.dumps({"pos": aperture, "speed": 110, "force": 20}),
                                   headers=headers)

                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': -250, 'y': 250, 'z': 50, 'rx': 180, 'ry': 0, 'rz': 0}),
                                   headers=headers)
                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': x, 'y': y, 'z': 50, 'rx': rx, 'ry': ry, 'rz': rz}),
                                   headers=headers)
                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': x, 'y': y, 'z': z, 'rx': rx, 'ry': ry, 'rz': rz}),
                                   headers=headers)

                res = requests.put(url=url_tfg + 'position/',
                                   data=json.dumps({"pos": 255, "speed": 110, "force": 20}),
                                   headers=headers)

                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': x, 'y': y, 'z': 50, 'rx': rx, 'ry': ry, 'rz': rz}),
                                   headers=headers)
                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': -320, 'y': -75, 'z': 50, 'rx': 180, 'ry': 0, 'rz': 0}),
                                   headers=headers)
                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': -320, 'y': -75, 'z': 20, 'rx': 180, 'ry': 0, 'rz': 0}),
                                   headers=headers)

                res = requests.put(url=url_tfg + 'position/',
                                   data=json.dumps({"pos": 0, "speed": 110, "force": 20}),
                                   headers=headers)

                res = requests.put(url=url_tx40 + 'position/',
                                   data=json.dumps({'x': -320, 'y': -75, 'z': 50, 'rx': 180, 'ry': 0, 'rz': 0}),
                                   headers=headers)

            else:
                print("There is no viable pose to grasp yet.")

    res = requests.put(url=url_tx40 + 'state/', data=json.dumps(False), headers=headers)
    res = requests.put(url=url_tfg + 'state/', data=json.dumps(False), headers=headers)
    camera.stop()
    print("System Terminated.")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    argparser.add_argument('-c', '--conf', default='./config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
