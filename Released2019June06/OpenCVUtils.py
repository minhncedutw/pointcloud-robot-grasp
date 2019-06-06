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

from GeneralUtils import Tuple
from MathUtils import deg2rad, create_rotz_matrix

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def rgb2bgr(img):
    return img[:, :, ::-1]

def bgr2rgb(img):
    return img[:, :, ::-1]

def draw_xy_axes(img, origin: Tuple[int, int]=(0, 0), angle: float=0, length: int=50, thickness=1):
    """Draw
    Args:
        img (2L ndarray): image.
        origin ([int, int]): origin point. Defaults to (0, 0).
        angle (float): DESCRIPTION. Defaults to 0.
        length (int): DESCRIPTION. Defaults to 50.
        thickness (int): DESCRIPTION. Defaults to 1.

    Usage:
        draw_xy_axes(img=img, origin=(50, 100), angle=45)
    """
    rot_matrix = create_rotz_matrix(theta=deg2rad(angle=angle))
    vect_x = np.append(np.array([length, 0]), [1, 1])
    vect_y = np.append(np.array([0, length]), [1, 1])
    vect_x1 = np.dot(rot_matrix, vect_x)
    vect_y1 = np.dot(rot_matrix, vect_y)

    point_o = origin
    point_x = tuple((np.array(origin) + vect_x1[:2]).astype(int))
    point_y = tuple((np.array(origin) + vect_y1[:2]).astype(int))

    cv2.arrowedLine(img=img, pt1=point_o, pt2=point_x, color=(0, 0, 255), thickness=1)
    cv2.arrowedLine(img=img, pt1=point_o, pt2=point_y, color=(0, 255, 0), thickness=1)
    draw_text(img=img, text="X", pos=point_x, color=(0, 0, 255), thickness=2)
    draw_text(img=img, text="Y", pos=point_y, color=(0, 255, 0), thickness=2)


def draw_text(img, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]=(255, 255, 255), thickness: int=1):
    """Draw
    Args:
        img (2L ndarray): image.
        text (str): text.
        pos ([int, int]): position of text.
        color ([int, int, int]): color of text. Defaults to (255, 255, 255).
        thickness (int): DESCRIPTION. Defaults to 1.

    Usage:
        draw_text(img=img, text="something", pos=(100, 200), color=(0, 0, 255), thickness=2)
    """
    cv2.putText(img=img, text=text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=thickness, lineType=cv2.LINE_AA)

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
