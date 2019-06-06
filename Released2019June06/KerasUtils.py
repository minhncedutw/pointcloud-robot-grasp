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
__date__ = "5/11/2019"
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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

from typing import Union, Any, List, Optional

from GeneralUtils import np, Path
from GeneralUtils import path2str, makedir

import tensorflow as tf
import keras as kr
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
# x1 = MatMul()([inputs, transformation1])
class MatMul(kr.layers.Layer):
    """~tf.matmul
    Do tf.matmul 2 tensors
    :param inputs (list of 2 tensors): the path to save model checkpoints
    :return . (tensor): result of 1st tensor matmul with 2nd tensor
    """
    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)

# how to use matmul+matmul_out_shape: x1 = Lambda(matmul, output_shape=matmul_out_shape)([inputs, transformation1])
def matmul(inputs):
    A, B = inputs
    return tf.matmul(A, B)

def matmul_out_shape(input_shape):
    shapeA, shapeB = input_shape
    assert shapeA[2] == shapeB[1]
    return tuple([shapeA[0], shapeA[1], shapeB[2]])


def expand_dim(global_feature, axis):
    return K.expand_dims(global_feature, axis)


def create_checkpoint_cb(directory: Union[str, Path]='checkpoints', filename: str='model.loss.{epoch:03d}.{val_loss:.4f}', monitor: str='val_loss', mode: str='auto'):
    makedir(path=directory, exist_ok=True)
    callback = ModelCheckpoint(filepath=path2str(directory) + '/' + filename + '.hdf5',  # string, path to save the model file.
                               monitor=monitor,  # quantity to monitor.
                               save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                               mode=mode, # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                               save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                               period=1,  # Interval (number of epochs) between checkpoints.
                               verbose=1),  # verbosity mode, 0 or 1.
    return callback[0]

def create_graph_cb(directory: Union[str, Path]='logs'):
    makedir(path=directory, exist_ok=True)
    callback = TensorBoard(log_dir=directory, # the path of the directory where to save the log files to be parsed by TensorBoard.
                           histogram_freq=0, # frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
                           # batch_size=batch_size,
                           write_graph=True, # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
                           write_grads=False, # whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0.
                           write_images=True,  # whether to write model weights to visualize as image in TensorBoard.
                           embeddings_freq=0),  # frequency (in epochs) at which selected embedding layers will be saved. If set to 0, embeddings won't be computed. Data to be visualized in TensorBoard's Embedding tab must be passed as embeddings_data.
    return callback[0]

import pickle
def save_trn_history(history, saving_path: Union[str, Path]='/history.pickle'):
    with open(saving_path, 'wb') as handle:
        pickle.dump(obj=history.history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_trn_history(history_path: Union[str, Path]='/history.pickle'):
    with open(history_path, 'rb') as handle:
        obj = pickle.load(file=handle)
    return obj


#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    # config_path = args.conf
    # with open(config_path) as config_buffer:    
    #     config = json.loads(config_buffer.read())

    '''**************************************************************
    I. Sample create callbacks
    '''
    def lr_schedule(epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        :param epoch (int): The number of epochs
        :return lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def create_callbacks(checkpoint_path, tensorboard_path):
        """List of callbacks
        Define a list of used callbacks.
        :param checkpoint_path (str): the path to save model checkpoints
        :param tensorboard_path (str): the path to save model training curve
        :return lr (float32): learning rate
        """
        callback_list = [
            create_checkpoint_cb(directory=checkpoint_path, filename='model.loss.{epoch:03d}.{val_loss:.4f}', monitor='val_loss'),
            create_checkpoint_cb(directory=checkpoint_path, filename='model.acc.{epoch:03d}.{val_acc:.4f}', monitor='val_acc'),
            create_graph_cb(directory=tensorboard_path),
            LearningRateScheduler(lr_schedule),
            ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
            EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
        ]
        return callback_list




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
