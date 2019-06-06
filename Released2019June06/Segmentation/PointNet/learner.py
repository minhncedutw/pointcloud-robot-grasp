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
import os
import sys
import time
import datetime
import argparse
from pathlib import Path
from typing import List, Tuple, Union, Any

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

from enum import Enum
class Platform(Enum):
    WINDOWS = 1
    UBUNTU = 2
    COLAB = 3
    DGX1 = 4
    none = 5

import numpy as np

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import load_model
from KerasUtils import save_trn_history

from PointCloudUtils import visualize_pc, coords_labels_to_pc, points_to_pc
from CustomLossUtils import categorical_focal_loss

from segmentation.PointNet.databunch import DataBunch, normalize_points
from segmentation.PointNet.generator import Generator
from segmentation.PointNet.model import PointNet, lr_schedule, create_callbacks, pointnet_loss

#==============================================================================
# Constant Definitions
#==============================================================================
_PLATFORM = Platform.WINDOWS
if _PLATFORM is Platform.WINDOWS:
    _DATA_DIRECTORY = 'E:/CLOUD/GDrive(t105ag8409)/data/shapenetcore_partanno_segmentation_benchmark_v0'
elif _PLATFORM is Platform.UBUNTU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"
    _DATA_DIRECTORY = '/home/minhnc-lab/gdrive/data/shapenetcore_partanno_segmentation_benchmark_v0'
elif _PLATFORM is Platform.COLAB:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"
    _DATA_DIRECTORY = '/content/drive/My Drive/data/shapenetcore_partanno_segmentation_benchmark_v0'
    from google.colab import drive
    drive.mount('/content/drive')
elif _PLATFORM is Platform.DGX1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"
    _DATA_DIRECTORY = '/home/sofin/Workspace/gdrive/data/shapenetcore_partanno_segmentation_benchmark_v0'
else:
    raise Exception('Platform not recognized! Only acknowlege WINDOWS, UBUNTU, COLAB, and DGX1 at the moment!')


class BaseLearner(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_shape, labels, verbose=True):
        raise NotImplementedError("error message")

    def save_weight(self, weight_path: Union[str, Path]):
        success = self.model.save_weights(filepath=weight_path)
        print('Save weight {}: {}'.format(weight_path, success))

    def load_weight(self, weight_path: Union[str, Path]):
        success = self.model.load_weights(weight_path)
        self.model._make_predict_function() # safe thread
        self.graph = tf.get_default_graph() # save graph of current model
        self.predict(np.ones(self.input_shape)) # initialize the network in the 1st time
        print('Load weight {}: {}'.format(weight_path, success))

    def save_model(self, model_path: Union[str, Path]):
        success = self.model.save(filepath=model_path)
        print('Save model {}: {}'.format(model_path, success))

    def load_model(self, model_path: Union[str, Path]):
        try:
            self.model = load_model(filepath=model_path)
            self.model._make_predict_function()  # safe thread
            self.graph = tf.get_default_graph()  # save graph of current model
            self.predict(np.ones(self.input_shape))  # initialize the network in the 1st time
            print('Load model {}: successful'.format(model_path))
        except IOError:
            print('An error occured trying to read the file.')

    def save_history(self, history_path):
        save_trn_history(history=self.model.trn_his, saving_path=history_path)


class PointNetLearner(BaseLearner):
    def __init__(self, input_shape, labels, verbose=True):
        """
        Description:
        :param NAME: TYPE, MEAN
        :return: TYPE, MEAN
        """
        '''**************************************************************
        I. Set parameters
        '''
        self.input_shape = input_shape

        self.labels = list(labels)
        self.n_classes = len(self.labels)

        '''**************************************************************
        II. Make the models
        '''
        self.model = PointNet(input_shape=input_shape, n_classes=self.n_classes)
        self.model._make_predict_function()  # have to initialize before threading
        self.graph = None

        if verbose: self.model.summary()

    def train(self, data_bunch, n_epochs, loss, optimizer, callbacks, metrics=['accuracy'],
              verbose=1):
        ############################################
        # Make train and validation generators
        ############################################

        ############################################
        # Compile the model
        ############################################
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics)

        ############################################
        # Start the training process
        ############################################
        from segmentation.PointNet.lr_finder import LRFinder
        lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(data_bunch.trn_generator.__len__()), epochs=3)
        self.model.fit_generator(generator=data_bunch.trn_generator, callbacks=[lr_finder])

        lr_finder.plot_loss()

        self.trn_his = self.model.fit_generator(generator=data_bunch.trn_generator,
                                                validation_data=data_bunch.val_generator,
                                                epochs=n_epochs, callbacks=callbacks,
                                                workers=1, verbose=verbose)

        ############################################
        # Compute mAP on the validation set
        ############################################

    def predict(self, x, verbose=0):
        start_time_pred = time.time()

        # Normalize data before feed to PointNet
        # x = self.normalizer(x)

        # Predict labels
        with self.graph.as_default():
            prob = self.model.predict(x=np.expand_dims(a=x, axis=0))

        # Decode probabilities to labels
        prob = np.squeeze(prob)  # remove dimension of batch
        pred_labels = prob.argmax(axis=1)

        stop_time_pred = time.time()
        if verbose: print('Execution Time: ', stop_time_pred - start_time_pred)

        return pred_labels

#==============================================================================
# Function Definitions
#==============================================================================

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
    data_dir= _DATA_DIRECTORY
    cat_choices = ['pipe_wrench']
    labels = ['0', 'background', 'wrench', 'pipe']
    n_points = 2048
    n_channels = 6
    n_classes = len(labels)
    n_epochs = 50
    bs = 2
    pretrained_model = './checkpointsmodel.acc.030.0.9973.hdf5'
    input_shape = [n_points, n_channels]

    current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
    checkpoint_path = os.path.join(_DATA_DIRECTORY, 'outputs/keras_{}/checkpoints'.format(current_time))
    tensorboard_path = os.path.join(_DATA_DIRECTORY, 'outputs/keras_{}/graphs'.format(current_time))

    '''**************************************************************
    II. Prepare the data
    '''
    # 1: Instantiate DataBunch(which contains generators of train and test set)
    data_bunch = DataBunch(data_dir=data_dir, categories=cat_choices, n_points=n_points, n_classes=n_classes,
                           n_channels=n_channels, bs=bs, normalizer=normalize_points, split_ratio=0.8,
                           balanced=True, shuffle=True, seed=0)

    # 2: Explore data
    batch_x, batch_y = data_bunch.val_generator.__getitem__(0) # get data
    print(batch_x.shape)

    visualize_pc(points_to_pc(batch_x[0, :, :])) # visualize data

    '''**************************************************************
    III. Create the model
    '''
    # 1: Build the model architecture.
    learner = PointNetLearner(input_shape=input_shape, labels=labels, verbose=True)

    # 2: Load some weights into the model.
    # learner.load_model(model_path=pretrained_model)
    # learner.load_weight(weight_path=pretrained_model)

    '''**************************************************************
    IV. Kick off the training
    '''
    # 1: Instantiate training arguments.
    optimizer = Adam(lr=lr_schedule(0))
    # loss = categorical_focal_loss(gamma=2, alpha=0.25)
    loss = pointnet_loss(xtran=learner.model.get_layer(name='xtran2'), reg_weight=0.001)
    metrics = ['accuracy']
    callbacks = create_callbacks(checkpoint_path=checkpoint_path, tensorboard_path=tensorboard_path)

    # 2: Train model
    learner.train(data_bunch=data_bunch,
                  n_epochs=n_epochs,
                  loss=loss, optimizer=optimizer, metrics=metrics,
                  callbacks=callbacks, verbose=1)

    '''**************************************************************
    V. Test
    '''
    # 1: Take data
    batch_x, batch_y = data_bunch.val_generator.__getitem__(1)  # get data
    print(batch_x.shape)

    # 2: Predict
    pred_labels = learner.predict(batch_x[0])

    # 3: Visualize output
    # visualize_pc(coords_labels_to_pc(coords=batch_x[0, :, :3], labels=pred_labels))
    visualize_pc(points_to_pc(points=batch_x[0, :, :6]))

    '''**************************************************************
    VI. Evaluate
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
