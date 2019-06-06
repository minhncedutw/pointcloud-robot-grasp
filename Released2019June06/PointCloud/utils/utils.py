
__version__ = '0.1'
__author__ = 'Cong-Minh Nguyen'

import numpy as np


def sample_indices(num_samples, length):
    """
    Get a list indice sample for an array
    :param num_samples: an integer, number of expected samples
    :param length: an integer, length of array
    :return: an array of numpy, is a list of indices
    """
    replace = num_samples > length
    new_indices = np.random.choice(a=length, size=num_samples, replace=replace)
    return new_indices


def sample_arrays(num_samples, return_indices=False, **kwargs):
    """
    Sample a list of arrays
    :param num_samples: an integer, number of expected samples
    :param return_indices: a boolean, that you want to return indices with your sampled arrays or not
    :param **kwargs: a list of arguments, is a list of arrays that you want to synchronically sample(they must have the same length)
    :return: a list of numpy array, that are synchronically-sampled arrays(include indices if toggle on 'return_indices')
    """
    # check whether input arrays have the same length
    length = -1
    for key,value in kwargs.items():
        if length == -1: length = len(value)
        assert len(value) == length, "input arguments must have same length"
    
    # randome selecting sample indices
    new_indices = sample_indices(num_samples, length)
    
    results = []
    for key,value in kwargs.items():
        value = np.array(value)[new_indices] # sample the array
        results.append(value)
    if return_indices: results.append(new_indices)
    return results


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is utils Program')


if __name__ == '__main__':
    main()

