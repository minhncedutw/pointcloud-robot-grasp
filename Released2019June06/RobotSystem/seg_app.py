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
__date__ = "5/24/2019"
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
import json

from flask import Flask, request, jsonify, redirect, url_for, flash
# from flask_restful import reqparse, abort, Api, Resource

import numpy as np
from segmentation.PointNet.learner import PointNetLearner

#==============================================================================
# APIs
#==============================================================================
app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def predict_segment():
    data = request.get_json()
    # print(data)
    points = np.array(data['points'])

    labels = learner.predict(x=points)

    return jsonify(labels.tolist())

#==============================================================================
# Main function
#==============================================================================
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    argparser.add_argument('-c', '--conf', default='./config.json', help='path to configuration file')

    args = argparser.parse_args()

    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ## Load config to variables
    lbl_names = config['seg_net']['labels']
    n_points = config['seg_net']['n_points']
    n_channels = config['seg_net']['n_channels']
    ptn_model = config['seg_net']['ptn_model']

    ## Interpolate some other variables
    n_classes = len(lbl_names)
    input_shape = [n_points, n_channels]

    learner = PointNetLearner(input_shape=input_shape, labels=lbl_names)
    learner.load_weight(weight_path=ptn_model)

    app.run(debug=False, host='localhost', port=5000)
