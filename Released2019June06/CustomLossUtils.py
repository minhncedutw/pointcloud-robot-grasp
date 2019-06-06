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
__date__ = "5/12/2019"
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
from KerasUtils import K, kr, tf

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References: https://arxiv.org/pdf/1708.02002.pdf

    Usage: model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        pt1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) # Return the elements, either from `y_pred` or 1, depending on the condition True of False
        pt0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred)) # Return the elements, either from `y_pred` or 1, depending on the condition True of False

        # clip to prevent NaN's and Inf's
        epsilon = K.epsilon()
        pt1 = K.clip(pt1, epsilon, 1. - epsilon)
        pt0 = K.clip(pt0, epsilon, 1. - epsilon)

        return - K.sum(alpha * K.pow(1. - pt1, gamma) * K.log(pt1)) \
               - K.sum((1 - alpha) * K.pow(pt0, gamma) * K.log(1. - pt0))

    return binary_focal_loss_fixed

def categorical_focal_loss(gamma=2., alpha=0.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage: model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

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

    '''**************************************************************
    II. Prepare the data
    '''
    # 1: Instantiate two `DataGenerator`

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
    # 1: Instantiate generator for the test/evaluation.

    # 2: Generate samples.

    # 3: Make predictions.

    # 4: Decode the raw predictions into labels.

    # 5: Display predicted results

    # 6: Run evaluation.


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
