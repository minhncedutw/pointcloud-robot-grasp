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
__date__ = "5/10/2019"
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

import numpy as np
import pickle

from typing import List, Tuple, Dict, Union, Optional, Any, Generic, TypeVar
Vector2D = Tuple[float, float]
Vector3D = Tuple[float, float, float]
Vector = List[float]

# Source: https://docs.python.org/3/library/typing.html
# Source: https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
'''**************************************************************
Python's basic lib
'''

# ─── UNZIP LIST OF TUPLE ────────────────────────────────────────────────────────
def unzip_list_of_tuple(l: list):
    return list(zip(*l))
    # return map(list, zip(*l)) #! just another case
    # return [list(t) for t in zip(*l)] #! just another case
# ────────────────────────────────────────────────────────────────────────────────


# ─── MANIPULATION WITH PATH ─────────────────────────────────────────────────────
# Source: https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd
def str2path(string: str) -> Path:
    return Path(string)


def path2str(path: Path) -> str:
    if isinstance(path, Path): string = str(path.resolve())
    else: string = path
    return string


def get_cur_file_path() -> Path:
    """
    Description: get path of current file
    :param NAME: TYPE, MEAN
    :return: TYPE, MEAN
    """
    return Path(__file__).resolve()


def get_cur_parent_dir() -> Path:
    """
    Description: get parent directory of current file
    :param NAME: TYPE, MEAN
    :return: TYPE, MEAN
    """
    return Path(__file__).resolve().parent


def get_cur_exe_dir() -> Path:
    """
    Description: get current execution directory
    :param NAME: TYPE, MEAN
    :return: TYPE, MEAN
    """
    return Path(__file__).cwd()


def makedir(path: Union[str, Path], mode=0o777, parents: bool=True, exist_ok: bool=False, verbose=False):
    """
    Description:
    :param path: [Path, str], path
    :param mode: [0o777, 0o444, ...], chmod(refer: https://help.ubuntu.com/community/FilePermissions)
    :param parents: [Path, str], True: if path is relative & False: if path is absolute
    :param exist_ok: boolean, if path already exists, True: force overwrite & False: raise error
    :return: TYPE, MEAN
    """
    if verbose:
        print('Create directory: \n\t{} \n\t{}'.format(Path(path).resolve(), Path(path).absolute()))
    Path(path).mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
# ────────────────────────────────────────────────────────────────────────────────


'''**************************************************************
Numpy lib
'''
# ─── ENCODE/DECODE LABELS-ONEHOT FOR MACHINE LEARNING ───────────────────────────
def onehot_encoding(labels, n_classes) -> np.ndarray:
    """
    Description: convert integer labels to one hot
    :param labels: an [int, ndarray], a label array of shape (d0, d1, d2, ...dn)
    :param n_classes: an int, number of classes
    :return: [ndarray], an one hot array of shape (d0, d1, ...dn, n_classes)
    """
    onehot = np.identity(n_classes)[labels]
    return onehot


def onehot_decoding(probs, class_axis) -> np.ndarray:
    """
    Description: convert one-hot encoding to labels
    :param probs: [ndarray], an probability array, one-hot-encoding type of shape (d0, d1, ...dn)
    :param class_axis: int, axis of classes in 'probs' array(0 <= class_axis <= n)
    :return: [int, ndarray], an label array of shape (d0, d1, ...dn-1)
    """
    labels = np.argmax(np.asarray(probs), axis=class_axis)
    return labels
# ────────────────────────────────────────────────────────────────────────────────

# ─── RANDOM ─────────────────────────────────────────────────────────────────────
def rand_int(low: int, high: int) -> int:
    return np.random.random_integers(low=low, high=high)

def rand_int_arr(low: int, high: int, size: List[np.uint]) -> np.ndarray:
    return np.random.random_integers(low=low, high=high, size=size)

def rand_float_arr(size: List[np.uint]) -> np.ndarray:
    return np.random.random(size=size)


def sample_indices(n_samples: int, max_index: int, replace: bool=None) -> np.ndarray:
    """
    Get a list indice sample for an array
    :param n_samples: an integer, number of expected samples
    :param length: an integer, length of array
    :return: an array of numpy, is a list of indices
    """
    if replace is None:
        replace = n_samples > max_index
    new_indices = np.random.choice(a=max_index, size=n_samples, replace=replace)
    return new_indices


def sample_arrays(arrs: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]], n_samples: int) -> List[np.ndarray]:
    """
    Sample a list of arrays
    :param arrs: List or Tuple of ndarray, the arrays that need to be sampled
    :param n_samples: an integer, number of expected samples
    :return: a list of numpy array, that are synchronically-sampled arrays
    """
    if isinstance(arrs, List) or isinstance(arrs, Tuple):
        list_or_tuple = True
    else:
        list_or_tuple = False
        arrs = [arrs]

    lengths = [len(arr) for arr in arrs]
    assert len(np.unique(lengths)) == 1, "Input arrss must have SAME length!"

    # random selecting sample indices
    new_indices = sample_indices(n_samples=n_samples, max_index=lengths[0])

    # shuffle all arrays by the same order
    results = [arr[new_indices] for arr in arrs]

    return results if list_or_tuple else results[0] # Return list if input arrs is list or arrays, else return array
# ────────────────────────────────────────────────────────────────────────────────

# ─── STACK ARRAYS ───────────────────────────────────────────────────────────────
def stack_list_vertical(arrs: List[np.ndarray]):
    return np.vstack(arrs)

def stack_list_horizontal(arrs: List[np.ndarray]):
    return np.hstack(arrs)
# ────────────────────────────────────────────────────────────────────────────────


'''**************************************************************
Pickle lib
'''
def save_pickle(obj, name: str='name', protocol: Optional[int]=pickle.HIGHEST_PROTOCOL):
    try:
        handle = open(name + ".pickle", 'wb')
        pickle.dump(obj=obj, file=handle, protocol=protocol)
        handle.close()
        return True
    except:
        return False

def load_pickle(name: str):
    """
    Description:
    :param name: str, file name without file extension
    :return: obj
    """
    handle = open(name + ".pickle", 'rb')
    obj = pickle.load(file=handle)
    handle.close()
    return obj


#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    # config_path = args.conf
    # with open(config_path) as config_buffer:    
    #     config = json.loads(config_buffer.read())
    path = 'gdgergegd'
    makedir(path=path, exist_ok=True, verbose=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
