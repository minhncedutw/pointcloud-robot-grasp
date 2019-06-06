'''
    File name: HANDBOOK
    Author: minhnc
    Date created(MM/DD/YYYY): 3/28/2019
    Last modified(MM/DD/YYYY HH:MM): 3/28/2019 11:11 PM
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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"
import copy
import matplotlib.pyplot as plt

import numpy as np
import cv2

from primesense import openni2 #, nite2
from primesense import _openni2 as c_api

import sys
from os.path import dirname, abspath
sys.path.append(dirname(abspath(__file__)))

import cv2
from PointCloud.pointcloud import PointCloud, load_ply_as_points, load_ply_as_object, visualize_point_cloud_object
from OpenCVUtils import draw_xy_axes, bgr2rgb, rgb2bgr

#==============================================================================
# Constant Definitions
#==============================================================================
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

#==============================================================================
# Function Definitions
#==============================================================================
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


def invert_image_channel(images):
    """
    Invert channels of image or list of images
    :param crop: a single image or a list of images(1, 2 or 3 ... channels both are ok)
    :return: image or list of images after inverting channels
    """
    return images[..., ::-1]

def distance_to_display(distance_map):
    """
    Convert depth image in type of distance-map to display-able image(gray) to display
    :param distance_map: depth image in type of distance
    :return: depth image in type of gray-scale
    """
    display_image = np.uint8(distance_map.astype(float) * 255 / 2 ** 12 - 1)  # Correct the range. Depth images are 12bits
    display_image = 255 - cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
    return display_image

def rgb_to_hsv(rgb):
    """
    Convert single RGB image to HSV image
    :param rgb: 3L ndarray of int, RGB image
    :return: 3L ndarray of int, HSV image
    """
    return cv2.cvtColor(src=rgb, code=cv2.COLOR_RGB2HSV)

def mark_color(hsv, threshold1:tuple([int, int, int, int]), threshold2:tuple([int, int, int, int])):
    """
    Mark hsv color in range of threshold 1 and threshold 2.
    :param hsv: 3L ndarray, hsv image
    :param threshold1: list of 4 integers, threshold 1
    :param threshold2: list of 4 integers, threshold 2
    :return: 1L ndarray of [0, 255], mask
    """
    mask = cv2.inRange(hsv, threshold1, threshold2)
    # mask = mask * (mask > 0)
    return mask

def rgbd_to_pointcloud(rgb, depth, scale=0.001, focal_length_x=520, focal_length_y=513, label=False, offset=0, **kwargs):
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
                points.append([X, Y, Z, R, G, B, label_point])
            else:
                points.append([X, Y, Z, R, G, B])
    return np.asarray(points)

def pointcloud_to_ply(points):
    """
    Convert point cloud to ply
    :param points: 3L ndarray, point list of [X, Y, Z, label]
    :return: a ply as string
    """
    points_str = []
    for i in range(len(points)):
        X, Y, Z, R, G, B, label = points[i]
        points_str.append(f"{X:f} {Y:f} {Z:f} {R.astype(np.int):d} {G.astype(np.int):d} {B.astype(np.int):d} {label.astype(np.int):d}\n")
    ply = f"""\
ply
format ascii 1.0
element vertex {len(points_str):d}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar label
end_header
{''.join(points_str)}\
"""
    return ply

def save_image(image, folder_path, filename, file_extend='.png'):
    """
    Save image in to a path with a name.
    :param image: 3L ndarray, image
    :param folder_path: a str
    :param filename: a str
    :param file_extend: a str
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename + file_extend)
    print('Saving {}: {}'.format(file_path, cv2.imwrite(file_path, invert_image_channel(image))))

def write_ply(ply, folder_path, filename, file_extend='.ply'):
    """
    Save ply point cloud in to a path with a name.
    :param image: 3L ndarray, image
    :param folder_path: a str
    :param filename: a str
    :param file_extend: a str
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename + file_extend)
    with open(file_path, 'w') as output:
        print('Saving {}: {}'.format(file_path, output.write(ply) > 0))

class Camera(object):
    def __init__(self, resolution:tuple([float,float])=(640, 480), fps=30, redist='./Redist'):
        """
        Initialize the camera object.
        :param resolution: tuple of list int of [sizeW, sizeH]
        :param fps: an integer, frame per second
        :param redist: a str, the path to camera driver
        """
        self.resolution = resolution
        self.image_shape = (resolution[1], resolution[0], 3)
        self.fps = fps
        self.redist = redist
        openni2.initialize(self.redist)
        if (openni2.is_initialized()):
            print("openNI2 initialized")
        else:
            raise ValueError("openNI2 not initialized")

        ## Register the device
        self.device = openni2.Device.open_any()

        ## Create the streams stream
        self.rgb_stream = self.device.create_color_stream()
        self.depth_stream = self.device.create_depth_stream()

        ## Configure the rgb_stream -- changes automatically based on bus speed
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                          resolutionX=self.resolution[0],
                                                          resolutionY=self.resolution[1], fps=self.fps))

        ## Configure the depth_stream -- changes automatically based on bus speed
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                            resolutionX=self.resolution[0],
                                                            resolutionY=self.resolution[1], fps=self.fps))

        ## Check and configure the mirroring -- default is True
        ## Note: I disable mirroring
        self.depth_stream.set_mirroring_enabled(False)
        self.rgb_stream.set_mirroring_enabled(False)

    def start(self):
        """
        Start recording, align the depth image and RGB image
        """
        ## Start the streams
        self.rgb_stream.start()
        self.depth_stream.start()

        ## Synchronize the streams
        self.device.set_depth_color_sync_enabled(True)  # synchronize the streams

        ## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
        self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print("Camera started")

    def stop(self):
        """
        Stop recording
        """
        self.rgb_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        print("Camera terminated")

    def update_stream(self, get_depth_display=True):
        """
        Update the stream(image of rgb and d)
        """
        self.rgb = self.get_rgb()  # RGB
        self.depth_distance = self.get_depth_distance()  # DEPTH

    def get_rgb(self):
        """
        Obtain rgb image from stream
        :return: returns numpy 3L ndarray to represent the rgb image.
        """
        return np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(self.image_shape)

    def get_depth_distance(self):
        """
        Obtain depth image from stream in distance-map type, distance in mm.
        :return: 1L ndarray, dtype=uint16, min=0, max=2**12-1
        """
        distance_map = np.fromstring(self.depth_stream.read_frame().get_buffer_as_uint16(),
                                     dtype=np.uint16).reshape(self.image_shape[:2])  # Works & It's FAST
        return distance_map

    def get_depth_display(self):
        """
        Obtain depth image from stream that is able to display
        :return: numpy 3L ndarray to represent the gray-depth image.
        """
        return distance_to_display(distance_map=self.depth_distance)

    def get_stream(self, crop:tuple([int,int])=(180, 220)):
        """
        Obtain the stream, crop at the centroid if necessary.
        :param crop: tuple of list int of [sizeH, sizeW]
        :param return_rgb: a boolean, that you want to return rgb or not
        :param return_dmap: a boolean, that you want to return dmap or not
        :param return_d4d: a boolean, that you want to return d4d or not
        :return: list of variables in sequence of canvas, rgb, dmap and d4d of they are set return_xxx=True
        """
        if crop is not None: cropH, cropW = crop
        else: cropH, cropW = self.image_shape[:2]

        centerW, centerH = np.asarray(self.resolution) // 2
        grid_step = (cropW // 6, cropH // 6)

        self.update_stream() # get new full size image of RGB and Depth

        # deep copy images to avoid changing original data
        rgb = copy.deepcopy(self.rgb)
        depth_distance = copy.deepcopy(self.depth_distance)

        # crop the images
        rgb = rgb[centerH - cropH // 2:centerH + cropH // 2, centerW - cropW // 2:centerW + cropW // 2]
        depth_distance = depth_distance[centerH - cropH // 2:centerH + cropH // 2, centerW - cropW // 2:centerW + cropW // 2]

        return rgb, depth_distance

    def display(self, crop:tuple([int,int])=(180, 220)):
        """
        Display the RGB-D image
        :param crop: tuple of list int of [sizeH, sizeW], the area is drew rectangle and grid
        """
        if crop is not None: cropH, cropW = crop
        else: cropH, cropW = self.image_shape[:2]

        centerW, centerH = np.asarray(self.resolution) // 2
        grid_step = (cropW // 6, cropH // 6)

        bgr = invert_image_channel(self.rgb)
        depth_gray = distance_to_display(self.depth_distance)
        canvas = np.hstack((bgr, depth_gray))  # stack 2 images into 1

        # draw crop rectangle of first image
        cv2.rectangle(img=canvas,
                      pt1=(centerW - cropW // 2, centerH - cropH // 2),
                      pt2=(centerW + cropW // 2, centerH + cropH // 2),
                      color=(0, 255, 0), thickness=1)
        # draw crop rectangle of second image
        cv2.rectangle(img=canvas,
                      pt1=(centerW - cropW // 2 + self.resolution[0], centerH - cropH // 2),
                      pt2=(centerW + cropW // 2 + self.resolution[0], centerH + cropH // 2),
                      color=(0, 255, 0), thickness=1)

        # draw vertical grid lines on first image. Notice: each point(pt1 or pt2) has format of: (column, row)
        cv2.line(img=canvas,
                 pt1=(centerW - cropW // 2, centerH - grid_step[1] * 2),
                 pt2=(centerW + cropW // 2, centerH - grid_step[1] * 2),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW - cropW // 2, centerH - grid_step[1]),
                 pt2=(centerW + cropW // 2, centerH - grid_step[1]),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW - cropW // 2, centerH),
                 pt2=(centerW + cropW // 2, centerH),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW - cropW // 2, centerH + grid_step[1]),
                 pt2=(centerW + cropW // 2, centerH + grid_step[1]),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW - cropW // 2, centerH + grid_step[1] * 2),
                 pt2=(centerW + cropW // 2, centerH + grid_step[1] * 2),
                 color=(255, 0, 0), thickness=1)
        # draw horizontal grid lines on first image
        cv2.line(img=canvas,
                 pt1=(centerW - grid_step[0] * 2, centerH - cropH // 2),
                 pt2=(centerW - grid_step[0] * 2, centerH + cropH // 2),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW - grid_step[0], centerH - cropH // 2),
                 pt2=(centerW - grid_step[0], centerH + cropH // 2),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW, centerH - cropH // 2),
                 pt2=(centerW, centerH + cropH // 2),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW + grid_step[0], centerH - cropH // 2),
                 pt2=(centerW + grid_step[0], centerH + cropH // 2),
                 color=(255, 0, 0), thickness=1)
        cv2.line(img=canvas,
                 pt1=(centerW + grid_step[0] * 2, centerH - cropH // 2),
                 pt2=(centerW + grid_step[0] * 2, centerH + cropH // 2),
                 color=(255, 0, 0), thickness=1)

        cv2.imshow("RGB-D", canvas)

        # return canvas


def save_scene_pipe_wrench(rgb, depth, ply_scene, ply_wrench, ply_pipe, saving_root, filename):
    """
    Save rgb, depth and ply point clouds of full scene and each objects into a root folder with the same name filename.
    :param rgb: 3L ndarray, image
    :param depth: 1L ndarray, image
    :param ply_scene: a str of ply
    :param ply_wrench: a str of ply
    :param ply_pipe: a str of ply
    :param saving_root: a str
    :param filename: a str
    """
    save_image(rgb, folder_path=os.path.join(saving_root, 'RGB'), filename=filename, file_extend='.png')
    save_image(depth, folder_path=os.path.join(saving_root, 'D'), filename=filename, file_extend='.png')
    write_ply(ply_scene, folder_path=os.path.join(saving_root, 'PC/scene'), filename=filename, file_extend='.ply')
    write_ply(ply_wrench, folder_path=os.path.join(saving_root, 'PC/wrench'), filename=filename, file_extend='.ply')
    write_ply(ply_pipe, folder_path=os.path.join(saving_root, 'PC/pipe'), filename=filename, file_extend='.ply')

def segment_color_save(rgb, depth_distance, count, saving_root):
    """
    Segment the object as in scene according colors then save data of images/point clouds into saving-root.
    :param rgb: 3L ndarray, image
    :param depth_distance: 1L ndarray, depth in distance
    :param count: an integer, is current counting, which is also used as filename
    :param saving_root: a str, root path of saving data
    """
    hsv = rgb_to_hsv(rgb)

    # detect area of color in range [threshold1, threshold2]
    mask_red = mark_color(hsv=hsv, threshold1=(0, 130, 160, 0), threshold2=(20, 255, 255, 0))
    mask_white = mark_color(hsv=hsv, threshold1=(0, 0, 160, 0), threshold2=(180, 55, 255, 0))

    # tred = cv2.inRange(hsv, (70, 100, 155, 0), (125, 255, 255, 0))
    # tred = tred * (tred > 0)

    # obtain the point cloud from rgbd
    points = rgbd_to_pointcloud(rgb=rgb, depth=depth_distance, scale=0.001,
                                focal_length_x=520, focal_length_y=513, label=True, offset=1,
                                mask1=mask_red, mask2=mask_white)
    # point_cloud = PointCloud(points=points[:, :3], labels=points[:, 3]); point_cloud.visualize()

    # obtain ply from point cloud then save to file path
    ply_scene = pointcloud_to_ply(points=points)

    # obtain ply from point cloud of color 1 then save to file path
    ply_wrench = pointcloud_to_ply(points=points[points[:, 6] == 2])

    # obtain ply from point cloud of color 1 then save to file path
    ply_pipe = pointcloud_to_ply(points=points[points[:, 6] == 3])

    save_scene_pipe_wrench(rgb=rgb,
                           depth=depth_distance, ply_scene=ply_scene, ply_wrench=ply_wrench,
                           ply_pipe=ply_pipe,
                           saving_root=saving_root, filename='{0:05d}'.format(count))

import open3d as o3d

def plot_rgbd(rgbd_image):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    # fig.savefig('temp.png', dpi=fig.dpi)

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    MODE = 'Find Color Range' # ['Find Color Range', 'Record Data', 'Create Train Data']
    saving_root = './recorded_data/standard_model_May22/'

    camera = Camera(resolution=(640, 480), fps=30, redist='./Redist')

    # # Find color
    if MODE == 'Find Color Range':
        camera.start()
        done = False
        cv2.namedWindow(window_detection_name)
        cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
        cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
        cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
        cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
        cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
        cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
        while not done:
            rgb, depth_distance = camera.get_stream(crop=(180, 200))
            camera.display(crop=(180, 200))
            key = cv2.waitKey(1) & 255
            if key == 27:  # terminate
                print("\tESC key detected!")
                done = True

            hsv = rgb_to_hsv(rgb)

            ### for red
            # tred = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
            tred = cv2.inRange(hsv, (0, 100, 100, 0), (10, 255, 255, 0)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
            tred = tred * (tred > 0)

            ### for white
            # twhite = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
            twhite = cv2.inRange(hsv, (80, 50, 90, 0), (130, 255, 255, 0))
            twhite = twhite * (twhite > 0)

            cv2.imshow('red', tred)
            cv2.imshow('blue', twhite)

    # # Record
    elif MODE == 'Record Data':
        camera.start()
        done = False
        count = 20
        while not done:
            rgb, depth_distance = camera.get_stream(crop=(180, 200))
            # rgb, depth_distance = camera.get_stream(crop=(480, 640))

            # convert rgb and d images -> rgbd image -> point cloud, then save into *.ply file
            # pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("./orbbec_astra_s.json")
            # rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(o3d.geometry.Image(rgb.astype(np.uint8)),
            #                                                                  o3d.geometry.Image(depth_distance.astype(np.uint16)),
            #                                                                  convert_rgb_to_intensity=False)
            #
            # pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
            #
            # o3d.visualization.draw_geometries([pcd])
            # o3d.io.write_point_cloud(filename="test.ply", pointcloud=pcd)

            camera.display(crop=(180, 200))
            key = cv2.waitKey(1) & 255
            if key == 27:  # terminate
                print("\tESC key detected!")
                done = True
            elif chr(key) == 's':  # screen capture
                count += 1
                start_time = time.time()
                segment_color_save(rgb=rgb, depth_distance=depth_distance, count=count, saving_root=saving_root)
                print("Saving time: {}".format(time.time() - start_time))
        camera.stop()

    # # Create train data of .pts and .seg
    elif MODE == 'Create Train Data':
        train_data_path = os.path.join(saving_root, 'train_data')
        points_path = os.path.join(train_data_path, 'points')
        labels_path = os.path.join(train_data_path, 'points_label')
        if not os.path.exists(train_data_path):
            os.makedirs(points_path)
            os.makedirs(labels_path)
        files = os.listdir(path=os.path.join(saving_root, 'PC/scene'))
        for i, file in enumerate(files):
            if file.endswith(".ply"):
                filename, ext = os.path.splitext(file)
                # point_cloud = load_ply_as_object(file_path=os.path.join(saving_root, 'PC/scene', file))
                # points = np.asarray(point_cloud.points)
                # colors = np.asarray(point_cloud.colors)
                # visualize_point_cloud_object(point_cloud)

                # Load the .ply file and write .pts and .seg files simultaneously
                line_idx = 0
                with open(os.path.join(saving_root, 'PC/scene', file)) as file_ply: # open the .ply file to read
                    with open(os.path.join(points_path, filename + '.pts'), 'w') as file_pts: # open .pts file to write
                        with open(os.path.join(labels_path, filename + '.seg'), 'w') as file_seg: # open .seg file to write
                            for line in file_ply:
                                if line_idx > 10:
                                    numbers = [float(num_str) for num_str in line.split(' ')]
                                    X, Y, Z = np.asarray(numbers[:3]).astype(np.float32)
                                    Z = Z - 0.677 # adjust Z value to move center of working space from camera to table surface
                                    R, G, B = np.asarray(numbers[3:6]).astype(np.int)
                                    label = int(numbers[6])

                                    file_pts.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(X, Y, Z, R, G, B))
                                    file_seg.write("{:d}\n".format(label))
                                line_idx += 1
                print("Data {:s} completed.".format(filename))


if __name__ == '__main__':
    main()
