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
__date__ = "5/16/2019"
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

import matplotlib.pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # The GPU id to use, usually either "0" or "1"
from KerasUtils import K, kr

#==============================================================================
# Constant Definitions
#==============================================================================
class LRFinder(kr.callbacks.Callback):
    def __init__(self, steps_per_epoch, epochs, min_lr=1e-5, max_lr=1e-1):
        """
        Description: A simple callback for finding the optimal learning rate range for your model + dataset.
        :param min_lr           : float, The lower bound of the learning rate range for the experiment.
        :param max_lr           : float, The upper bound of the learning rate range for the experiment.
        :param steps_per_epoch  : uint, Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        :param epochs           : uint, Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

        Usage
            ```python
                lr_finder = LRFinder(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     epochs=3)
                model.fit(X_train, Y_train, callbacks=[lr_finder])

                lr_finder.plot_loss()
            ```

        # References
            Blog post: jeremyjordan.me/nn-learning-rate
            Original paper: https://arxiv.org/abs/1506.01186
        """
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iters = steps_per_epoch * epochs # count number of iterations
        self.iter = 0 # iteration
        self.his = {} # history

    def cal_lr(self):
        '''Calculate the learning rate.'''
        x = self.iter / self.num_iters
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(x=self.model.optimizer.lr, value=self.min_lr)

    def on_batch_end(self, batch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iter += 1

        self.his.setdefault('lr', []).append(K.get_value(x=self.model.optimizer.lr))
        self.his.setdefault('iters', []).append(self.iter)

        for k, v in logs.items():
            self.his.setdefault(k, []).append(v)

        K.set_value(x=self.model.optimizer.lr, value=self.cal_lr())

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.his['iters'], self.his['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.his['lr'], self.his['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

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
