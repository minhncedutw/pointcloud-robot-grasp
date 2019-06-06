import requests
import json
import numpy as np
from Segmentation.PointNet.generator import Generator

# 0: Prepare main url of APIs
host = 'localhost'
port = 5000
version = 'v1'
task = 'state'
url = 'http://{:s}:{:d}/{:s}/tx40/{:s}/'.format(host, port, version, task)

# Prepare before loading data
DATA_DIRECTORY = 'E:/CLOUD/GDrive(t105ag8409)/data/shapenetcore_partanno_segmentation_benchmark_v0'
data_dir= DATA_DIRECTORY
cat_choices = ['pipe_wrench']
labels = ['0', 'background', 'wrench', 'pipe']
n_points = 2048*8
n_channels = 6
n_classes = len(labels)
n_epochs = 50
bs = 2
pretrained_model = './checkpointsmodel.acc.030.0.9973.hdf5'
input_shape = [n_points, n_channels]

val_generator = Generator(directory=data_dir, cat_choices=cat_choices, n_channels=6,
                          n_points=n_points, n_classes=n_classes, bs=bs,
                          train=False, balanced=False, shuffle=True, seed=10)

# 1: Take data
batch_x, batch_y = val_generator.__getitem__(1)  # get data

# 2: Request prediction for taken data
data = {'points': batch_x[0].tolist()} # convert ndarray data to dict of list
j_data = json.dumps(data) # convert dict data to record of json
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'} # config header of REST API
res = requests.post(url, data=j_data, headers=headers) # send request
pred_labels = np.asarray(json.loads(res.text)) # convert response json into array of values
print(res)
print(res.text)
print(json.loads(res.text))
