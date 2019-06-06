import os
import sys
import math

import random
import copy

import numpy as np

from keras.utils import Sequence, to_categorical

from GeneralUtils import onehot_decoding
from PointCloudUtils import visualize_pc, coords_labels_to_pc

class Generator(Sequence):
    def __init__(self, directory, n_points=2048, cat_choices=None, n_classes: int=None, n_channels: int=3,
                 bs=32, train=True, balanced=True, shuffle=False, seed=0):
        self.dir = directory
        self.n_points = n_points
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.cat_choices = cat_choices
        self.bs = bs
        self.train = train
        self.balanced = balanced
        self.seed = seed

        cat_list_file = os.path.join(self.dir, 'synsetoffset2category.txt') # category file
        self.cat_dict = {} # category dictionary
        # create dict of category-path
        with open(cat_list_file, 'r') as f: # get list of category/path in the cat_list_file
            for line in f:
                [category, path] = line.strip().split()
                self.cat_dict[category] = path # 'category' information is saved in 'folder'
        if not cat_choices is None: # exclude category/path that are not chosen
            self.cat_dict = {category: path for category, path in self.cat_dict.items() if category in cat_choices}

        self.datapath = []
        for item in self.cat_dict:
            points_path = os.path.join(self.dir, self.cat_dict[item], "points") # path to points folder
            labels_path = os.path.join(self.dir, self.cat_dict[item], "points_label") # path to labels folder

            self.points_filenames = [file for file in sorted(os.listdir(points_path))]
            if shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(self.points_filenames)
            if self.train:
                self.points_filenames = self.points_filenames[:int(len(self.points_filenames) * 0.9)]
            else:
                self.points_filenames = self.points_filenames[int(len(self.points_filenames) * 0.9):]

            for fn in self.points_filenames:
                token = (os.path.splitext(os.path.basename(fn))[0])
                pts_file = os.path.join(points_path, token + '.pts')
                seg_file = os.path.join(labels_path, token + '.seg')
                self.datapath.append((item, pts_file, seg_file))

        # count_classes = 0
        # for i in range(math.ceil(len(self.datapath) / 50)):
        #     biggest_label = np.max(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
        #     if biggest_label > count_classes:
        #         count_classes = biggest_label
        # if (self.n_classes is None) or (self.n_classes < count_classes + 1):
        #     self.n_classes = count_classes + 1

    @staticmethod
    def rotate_point_cloud(data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        rotation_angle = np.random.uniform() * 0.05 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = copy.deepcopy(data)
        rotated_data[:, :3] = np.dot(rotated_data[:, :3], rotation_matrix) # only rotate the X, Y, Z element. R, G, B ... keep original
        return rotated_data

    @staticmethod
    def jitter_point_cloud(data, sigma=0.0001, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    def __len__(self):
        return int(np.ceil(len(self.datapath) / float(self.bs)))

    def __getitem__(self, idx):
        left_bound = idx * self.bs
        right_bound = (idx + 1) * self.bs

        if right_bound > len(self.datapath):
            right_bound = len(self.datapath)

        batch_x = []
        batch_y = []
        for i in range(right_bound - left_bound):
            points = np.loadtxt(self.datapath[left_bound + i][1]).astype('float32')
            labels = np.loadtxt(self.datapath[left_bound + i][2]).astype('int')

            if (points.shape[1] > 5): # convert uint8 color to float color
                points[:, 3:6] /= 255.0

            # if data is more than required number of points, we should select different points when sample
            # else, we have to duplicate some data to get enough input points
            if len(points) > self.n_points: replace = False
            else: replace = True

            if self.balanced:
                # Balance loading points
                obj_idxs = np.argwhere(labels > 1).ravel()
                grd_idxs = np.argwhere(labels == 1).ravel()
                choice = np.array([])
                for loop in range(100):
                    taken_grd_idx = np.random.choice(a=grd_idxs, size=len(obj_idxs), replace=replace)
                    choice = np.hstack((choice, obj_idxs))
                    choice = np.hstack((choice, taken_grd_idx))
                    if len(choice) >= self.n_points:
                        break
                np.random.shuffle(choice)
                choice = choice[:self.n_points].astype(np.int)
            else:
                choice = np.random.choice(a=len(points), size=self.n_points, replace=replace)

            points = points[choice, :self.n_channels]
            labels = labels[choice]

            # if self.train:
            #     is_rotate = random.randint(0, 1)
            #     is_jitter = random.randint(0, 1)
            #     if is_rotate == 1:
            #         points = self.rotate_point_cloud(points)
            #     if is_jitter == 1:
            #         points = self.jitter_point_cloud(points)

            onehot_labels = to_categorical(y=labels, num_classes=self.n_classes)

            batch_x.append(points)
            batch_y.append(onehot_labels)

            if len(np.array(batch_x).shape) < 3:
                print("error")

        return np.array(batch_x), np.array(batch_y)

'''
Illustrate how to use generator
'''
def main(argv=None):
    # Declare necessary arguments
    data_directory = 'E:/CLOUD/GDrive(t105ag8409)/data/shapenetcore_partanno_segmentation_benchmark_v0'
    n_points = 1024
    num_epoches = 10
    bs = 2
    cat_choices = ['pipe_wrench']
    pretrained_model = None
    n_classes = 4

    # Declare generator
    trn_generator = Generator(directory=data_directory, cat_choices=cat_choices, n_channels=3,
                                   n_points=n_points, n_classes=n_classes, bs=bs,
                                   train=True, balanced=True, shuffle=True, seed=10)
    
    # Take the data from generator
    batch_x, batch_y = trn_generator.__getitem__(3)

    # Visualize data
    visualize_pc(coords_labels_to_pc(coords=batch_x[0], labels=onehot_decoding(probs=batch_y[0], class_axis=1)))
    
if __name__ == '__main__':
    main()