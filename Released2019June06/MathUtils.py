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
__date__ = "5/21/2019"
__maintainer__ = "CONG-MINH NGUYEN"
__email__ = "minhnc.edu.tw@gmail.com"
__status__ = "Development"  # ["Prototype", "Development", or "Production"]
# Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
# Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

#==============================================================================
# Imported Modules
#==============================================================================
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Union, Any


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # The GPU id to use, usually either "0" or "1"

import numpy as np
import cv2

#==============================================================================
# Constant Definitions
#==============================================================================
PI = np.pi
TWOPI = 2 * PI
ONE_DIV_PI = 1. / np.pi
ONE_DIV_180 = 1. / 180

#==============================================================================
# Function Definitions
#==============================================================================

def rad2deg(angle: float):
    """Convert radian unit to degree unit

    Args:
        angle (float): input angle.

    Returns:
        float: output angle.

    Usage:
        deg_angle = rad2deg(rad_angle)
    """
    return angle * ONE_DIV_PI * 180


def deg2rad(angle: float):
    """Convert degree unit to radian unit

    Args:
        angle (float): input angle.

    Returns:
        float: output angle.

    Usage:
        rad_angle = deg2rad(deg_angle)
    """
    return angle * ONE_DIV_180 * PI


def m2mm(value: float):
    """Convert meter unit to milimeter unit

    Args:
        value (float): input value.

    Returns:
        float: output value.

    Usage:
        mm_value = m2mm(m_value)
    """
    return value * 1000


def mm2m(value: float):
    """Convert meter unit to milimeter unit

    Args:
        value (float): input value.

    Returns:
        float: output value.

    Usage:
        m_value = mm2m(m_value)
    """
    return value * 0.001


def create_tranl_matrix(vector: Tuple[float, float, float]):
    """Create 3D matrix of translating along vector

    Args:
        vector ([float, float, float]): 3D vector.

    Returns:
        float(4x4): translation matrix.

    Usage:
        rotx = create_rotx_matrix(theta)
    """
    matrix = np.eye(4)
    matrix[:3, 3] += vector
    return matrix


def create_rotz_matrix(theta: float):
    """Create 3D matrix of rotating around z transformation

    Args:
        theta (float): angle in radian unit.

    Returns:
        float(4x4): rotation matrix around z.

    Usage:
        rotz = create_rotz_matrix(theta)
    """
    matrix = np.eye(4)
    matrix[0, 0] = np.cos(theta)
    matrix[0, 1] = -np.sin(theta)
    matrix[1, 0] = np.sin(theta)
    matrix[1, 1] = np.cos(theta)
    return matrix


def create_rotx_matrix(theta: float):
    """Create 3D matrix of rotating around x transformation

    Args:
        theta (float): angle in radian unit.

    Returns:
        float(4x4): rotation matrix around x.

    Usage:
        rotx = create_rotx_matrix(theta)
    """
    matrix = np.eye(4)
    matrix[1, 1] = np.cos(theta)
    matrix[1, 2] = -np.sin(theta)
    matrix[2, 1] = np.sin(theta)
    matrix[2, 2] = np.cos(theta)
    return matrix


def create_roty_matrix(theta: float):
    """Create 3D matrix of rotating around y transformation

    Args:
        theta (float): angle in radian unit.

    Returns:
        float(4x4): rotation matrix around y.

    Usage:
        roty = create_roty_matrix(theta)
    """
    matrix = np.eye(4)
    matrix[2, 2] = np.cos(theta)
    matrix[2, 0] = -np.sin(theta)
    matrix[0, 2] = np.sin(theta)
    matrix[0, 0] = np.cos(theta)
    return matrix

#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    # config_path = args.conf
    # with open(config_path) as config_buffer:    
    #     config = json.loads(config_buffer.read())

    '''**************************************************************
    I. Set parameters
    '''




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
