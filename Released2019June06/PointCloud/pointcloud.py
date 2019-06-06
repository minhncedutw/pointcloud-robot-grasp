#!pip install open3d-python

__version__ = '0.1'
__author__ = 'Cong-Minh Nguyen'

import numpy as np
import open3d as op3
# from .utils.utils import sample_indices, sample_arrays
# from point_cloud.utils.utils import sample_indices, sample_arrays

import sys
from os.path import dirname, abspath
sys.path.append(dirname(abspath(__file__)))
from utils.utils import sample_indices, sample_arrays

"""
Define some utility functions for point cloud
"""
def load_ply_as_object(file_path):
    """
    Load a point cloud from a .ply file
    :param file_path: a string, path to the .ply file
    :return: a point cloud object of Open3D
    """
    point_cloud = op3.read_point_cloud(filename=file_path)
    return point_cloud


def point_cloud_object_to_numpy(point_cloud_object:op3.PointCloud()):
    """
    Convert point cloud of Open3D object to numpy array
    :param point_cloud_object: a Open3D object, of point cloud
    :return: a numpy array of points
    """
    return np.asarray(point_cloud_object.points)


def point_cloud_numpy_to_object(points):
    """
    Convert point cloud of numpy array to Open3D object
    :param points: a numpy array of points, that describes the point cloud
    :return: a point cloud object of Open3D
    """
    point_cloud = op3.PointCloud()
    point_cloud.points = op3.Vector3dVector(points)
    return point_cloud


def load_ply_as_points(file_path):
    """
    Load a point cloud from a .ply file
    :param file_path: a string, path to the .ply file
    :return: an numpy array of points
    """
    point_cloud = load_ply_as_object(file_path=file_path)
    return point_cloud_object_to_numpy(point_cloud)


def label_to_color(labels):
    """
    Convert labels to colors(label's values must be in range [1, 6])
    :param labels: an array of label values
    :return: an array of color codes
    """
    map_label_to_rgb = {
        1: [0, 255, 0], # green
        2: [0, 0, 255], # blue
        3: [255, 0, 0], # red
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }
    colors = np.array([map_label_to_rgb[label] for label in labels])
    return colors


def visualize_point_cloud_object(*args):
    """
    Visualize of list of point clouds
    :param *args: a list of Open3D objects, that are point clouds
    """
    op3.draw_geometries([*args])


"""
Define PointCloud object
"""
class PointCloud(object):
    def __init__(self, points, labels=None, colors=None):
        """
        Initialize a pointcloud from a list of points, maybe with other attributes like labels and color codes
        :param points: a list of points
        :param labels: a list of integers, that are labels for each points
        :param colors: a list of color codes - each code is array of 3 values R-G-B, that are labels for each points
        """
        self.points = np.array(points)
        assert (labels is None) or (len(labels) == len(self.points)), "labels must have same length with points"
        self.labels = None if labels is None else np.array(labels).astype(int)
        assert (colors is None) or (len(colors) == len(self.points)), "colors must have same length with points"
        self.colors = None if colors is None else np.array(colors).astype(int)
    
    def set_attributes(self, **kwargs):
        """
        Set values for PointCloud object
        :param **kwargs: list of arguments(maybe points, labels and colors), arguments must have the same length
        """
        [self.__setattr__(key, kwargs.get(key)) for key in ['points', 'labels', 'colors']]
        assert (self.labels is None) or (len(self.labels) == len(self.points)), "labels must have same length with points"
        assert (self.colors is None) or (len(self.colors) == len(self.points)), "colors must have same length with points"

    def sample_point_cloud(self, num_samples:int):
        """
        Sample the point cloud
        :param num_samples: an integer, that is the number of points you want to sample
        """
        new_indices = sample_indices(num_samples=num_samples, length=len(self.points))
        self.points = self.points[new_indices]
        if self.labels is not None: self.labels = self.labels[new_indices]
        if self.colors is not None: self.colors = self.colors[new_indices]
    
    def measure_centroid(self):
        """
        Mesure the centroid of point cloud
        :return: a point as an array(each element in array is x, y, z, ...)
        """
        self.centroid = np.mean(self.points, axis=0)
        return self.centroid
    
    def to_object(self):
        """
        Convert internal attributes to Open3D point cloud object
        :return: an object of Open3D point cloud
        """
        point_cloud = point_cloud_numpy_to_object(self.points)
        if self.labels is not None: self.colors = label_to_color(self.labels)
        if self.colors is not None: point_cloud.colors = op3.Vector3dVector(self.colors)
        return point_cloud
    
    def visualize(self):
        """
        Visualize the point cloud.
        ! You can call .set_attributes(colors=label_to_color(labels)) to color the point cloud according to your labels
        """
        point_cloud = self.to_object()
        visualize_point_cloud_object(point_cloud)

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is pointcloud Program')

    from pathlib import Path
    # pipe_path = 'E:/CLOUD/GDrive(t105ag8409)/Projects/ARLab/Point Cloud Robot Grasping/camera/orbbec_astra_s/recorded_data/standard_model_May07/PC/pipe/00011.ply'
    pipe_path = 'E:/CLOUD/GDrive(t105ag8409)/Projects/ARLab/Point Cloud Robot Grasping/camera/orbbec_astra_s/recorded_data/00094.ply'
    points = load_ply_as_points(file_path=pipe_path)
    point_cloud = PointCloud(points=points, labels=np.ones(len(points))*2)
    point_cloud.visualize()
    points_obj = load_ply_as_object(file_path=pipe_path)
    point_cloud = PointCloud(points=np.asarray(points_obj.points), colors=np.asarray(points_obj.colors))
    point_cloud.visualize()
    visualize_point_cloud_object(points_obj)

    pipe_file_path = 'E:/CLOUD/GDrive(t105ag8409)/Projects/ARLab/Point Cloud Robot Grasping/data/72_scenes_of_pipe/pipe/1001.ply'
    pipe_file_path2 = 'E:/CLOUD/GDrive(t105ag8409)/Projects/ARLab/Point Cloud Robot Grasping/data/72_scenes_of_pipe/pipe/1002.ply'

    # illustrate how to init point cloud from loading ply
    point_cloud = PointCloud(points=load_ply_as_points(file_path=pipe_file_path))
    point_cloud2 = PointCloud(points=load_ply_as_points(file_path=pipe_file_path2))

    # illustrate how to visualize
    point_cloud.visualize()

    visualize_point_cloud_object(point_cloud.to_object(), point_cloud2.to_object())


if __name__ == '__main__':
    main()
