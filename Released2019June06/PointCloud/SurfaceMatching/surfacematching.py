import sys

import numpy as np

sys.path.append('./../')
from pointcloud import PointCloud, load_ply_as_points, point_cloud_numpy_to_object, visualize_point_cloud_object, op3


import copy
def visualize_registration(source:op3.PointCloud(), target:op3.PointCloud(), transformation=np.eye(4)):
    """
    Visualize the matching performance of source on target through transformation
    Default transformation is I(it means no transform)
    :param num_samples: an integer, that is the number of points you want to sample
    :return: a point as an array(each element in array is x, y, z, ...)
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0]) # source point cloud: red
    target_temp.paint_uniform_color([0, 0, 1]) # target point cloud: blue
    source_temp.transform(transformation)
    visualize_point_cloud_object(source_temp, target_temp)
    
def sample_point_cloud_feature(point_cloud:op3.PointCloud(), voxel_size, verbose=False):
    """
    Down sample and sample the point cloud feature
    :param point_cloud: an object of Open3D
    :param voxel_size: a float value, that is how sparse you want the sample points is
    :param verbose: a boolean value, display notification and visualization when True and no notification when False
    :return: 2 objects of Open3D, that are down-sampled point cloud and point cloud feature fpfh
    """
    if verbose: print(":: Downsample with a voxel size %.3f." % voxel_size)
    point_cloud_down_sample = op3.voxel_down_sample(point_cloud, voxel_size)

    radius_normal = voxel_size * 1
    if verbose: print(":: Estimate normal with search radius %.3f." % radius_normal)
    op3.estimate_normals(point_cloud_down_sample, op3.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    if verbose: print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    point_cloud_fpfh = op3.compute_fpfh_feature(point_cloud_down_sample, op3.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return point_cloud_down_sample, point_cloud_fpfh

def execute_global_registration(source_down:op3.PointCloud(), target_down:op3.PointCloud(), 
                                source_fpfh:op3.PointCloud(), target_fpfh:op3.PointCloud(), 
                                voxel_size, verbose=False):
    """
    Do global matching, find gross transformation form source to target
    :param source_down, target_down: 2 objects of Open3D, that are point clouds of source and target after down-sampling
    :param source_fpfh, target_fpfh: 2 objects of Open3D, that are point cloud features of source and target
    :param voxel_size: a float value, that is how sparse you want the sample points is
    :param verbose: a boolean value, display notification and visualization when True and no notification when False
    :return: a transformation object
    """
    distance_threshold = voxel_size * 1.5
    if verbose:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = op3.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            op3.TransformationEstimationPointToPoint(False), 4,
            [op3.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            op3.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            op3.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source:op3.PointCloud(), target:op3.PointCloud(), voxel_size, gross_matching, verbose=False):
    """
    Refine the matching
    :param source, target: 2 objects of Open3D, that are point clouds of source and target
    :param voxel_size: a float value, that is how sparse you want the sample points is
    :param gross_matching: a transformation matrix, that grossly matches source to target
    :param verbose: a boolean value, display notification and visualization when True and no notification when False
    :return: a transformation object
    """
    distance_threshold = voxel_size * 1
    if verbose:
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
    result = op3.registration_icp(source, target, distance_threshold,
                                  gross_matching.transformation,
                                  op3.TransformationEstimationPointToPlane(),
                                  op3.ICPConvergenceCriteria(max_iteration=2000))
    return result

def match_surface(source:op3.PointCloud(), target:op3.PointCloud(), voxel_size = 0.005, verbose=False):
    """
    Find registertration to transform source point cloud to target point cloud
    :param source, target: 2 objects of Open3D, that are point clouds of source and target
    :param voxel_size: a float value, that is how sparse you want the sample points is
    :param verbose: a boolean value, display notification and visualization when True and no notification when False
    :return: a transformation object
    """
    if verbose: visualize_registration(source=source, target=target, transformation=np.identity(4))  # visualize point cloud
    
    # downsample data
    source_down, source_fpfh = sample_point_cloud_feature(point_cloud=source, voxel_size=voxel_size, verbose=verbose)
    target_down, target_fpfh = sample_point_cloud_feature(point_cloud=target, voxel_size=voxel_size, verbose=verbose)

    # 1st: gross matching(RANSAC)
    result_ransac = execute_global_registration(source_down=source_down, target_down=target_down, 
                                                source_fpfh=source_fpfh, target_fpfh=target_fpfh, 
                                                voxel_size=voxel_size, verbose=verbose)
    if verbose: visualize_registration(source=source_down, target=target_down, transformation=result_ransac.transformation)

    # 2nd: fine-tune matching(ICP)
    result_icp = refine_registration(source=source_down, target=target_down, voxel_size=voxel_size, gross_matching=result_ransac)
    if verbose: visualize_registration(source=source_down, target=target_down, transformation=result_icp.transformation)
    return result_icp


import math
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    """
    Check???
    :param R: a matrix
    :return: a boolean, ???
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """
    Measure rotations around x, y and z from transformation matrix
    :param R: a rotation matrix
    :return: an array of 3 values that describe rotations around x, y and z axis, unit is "radian"
    """
    assert (isRotationMatrix(R))

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
