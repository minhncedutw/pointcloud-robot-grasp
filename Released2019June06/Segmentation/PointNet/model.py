import os

from typing import List, Tuple, Union

import numpy as np

from KerasUtils import K, kr, tf, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from KerasUtils import MatMul, create_checkpoint_cb, create_graph_cb
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation, Flatten, Dropout
from keras.layers import Lambda, concatenate

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    :param epoch (int): The number of epochs
    :return lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 45:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 20:
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
        create_checkpoint_cb(directory=checkpoint_path, filename='model.loss.{epoch:03d}.{val_loss:.4f}',
                             monitor='val_loss'),
        create_checkpoint_cb(directory=checkpoint_path, filename='model.acc.{epoch:03d}.{val_acc:.4f}',
                             monitor='val_acc'),
        create_graph_cb(directory=tensorboard_path),
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    ]
    return callback_list

from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
def create_callback_list(checkpoint_path, tensorboard_path):
    """List of callbacks
    Define a list of used callbacks.
    :param checkpoint_path (str): the path to save model checkpoints
    :param tensorboard_path (str): the path to save model training curve
    :return lr (float32): learning rate
    """
    if not os.path.exists(path=checkpoint_path):
        os.makedirs(name=checkpoint_path)
    if not os.path.exists(path=tensorboard_path):
        os.makedirs(name=tensorboard_path)

    callback_list = [
        ModelCheckpoint(
                        filepath=checkpoint_path + '/model.acc.{epoch:03d}.{val_acc:.4f}.hdf5', # string, path to save the model file.
                        monitor='val_acc', # quantity to monitor.
                        save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto', # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period= 1, # Interval (number of epochs) between checkpoints.
                        verbose=1), # verbosity mode, 0 or 1.
        ModelCheckpoint(
                        filepath=checkpoint_path + '/model.loss.{epoch:03d}.{val_acc:.4f}.hdf5',  # string, path to save the model file.
                        monitor='val_loss',  # quantity to monitor.
                        save_best_only=True, # if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                        mode='auto', # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
                        save_weights_only='false', # if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                        period=1,  # Interval (number of epochs) between checkpoints.
                        verbose=1),  # verbosity mode, 0 or 1.
        TensorBoard(log_dir=tensorboard_path, # the path of the directory where to save the log files to be parsed by TensorBoard.
                    histogram_freq=0, # frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
                    # batch_size=batch_size,
                    write_graph=True, # whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
                    write_grads=False, # whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0.
                    write_images=True, # whether to write model weights to visualize as image in TensorBoard.
                    embeddings_freq=0), # frequency (in epochs) at which selected embedding layers will be saved. If set to 0, embeddings won't be computed. Data to be visualized in TensorBoard's Embedding tab must be passed as embeddings_data.
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(factor=np.sqrt(0.1),
                          cooldown=0,
                          patience=5,
                          min_lr=0.5e-6),
    ]
    return callback_list

def expand_dim(global_feature, axis):
    return K.expand_dims(global_feature, axis)

def duplicate(global_feature, num_points):
    return K.tile(global_feature, [1, num_points, 1])

def conv_stack(x, num_hiddens:np.ndarray, kernel_size=1):
    assert len(num_hiddens) > 0, "number of hidden layers must >= 1"
    outputs = []
    for i, num_hidden_per_layer in enumerate(num_hiddens):
        x = Convolution1D(filters=num_hidden_per_layer, kernel_size=kernel_size, strides=1, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        outputs.append(x)
    return outputs

def dense_stack(x, num_hiddens:np.ndarray):
    assert len(num_hiddens) > 0, "number of hidden layers must >= 1"
    outputs = []
    for i, num_hidden_per_layer in enumerate(num_hiddens):
        x = Dense(units=num_hidden_per_layer, activation='relu')(x)
        x = BatchNormalization()(x)
        outputs.append(x)
    return outputs

def T_net(x, num_hiddens1:np.ndarray, num_hidden2:np.ndarray, num_output_channels:int, name='xtran'):
    x = conv_stack(x, num_hiddens=num_hiddens1)[-1]
    x = GlobalMaxPooling1D()(x)
    x = dense_stack(x, num_hiddens=num_hidden2)[-1]
    x = Dense(units=num_output_channels * num_output_channels,
              weights=[np.zeros([num_hidden2[-1], num_output_channels * num_output_channels]),
                       np.eye(num_output_channels).flatten().astype(np.float32)])(x)
    transformation = Reshape((num_output_channels, num_output_channels), name=name)(x)
    return transformation

def PointNet(input_shape: Tuple[int, int]=(2048, 3), n_classes: int=10):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    # Begin defining Pointnet Architecture
    n_points, n_channel = input_shape
    inputs = Input(shape=input_shape)

    transformation1 = T_net(inputs, num_hiddens1=[64, 128, 1024], num_hidden2=[512, 256], num_output_channels=n_channel, name='xtran1')
    x1 = MatMul()([inputs, transformation1])

    features1 = conv_stack(x1, num_hiddens=[64, 64])

    transformation2 = T_net(features1[-1], num_hiddens1=[64, 128, 1024], num_hidden2=[512, 256], num_output_channels=64, name='xtran2')
    x2 = MatMul()([features1[-1], transformation2])

    features2 = conv_stack(x2, num_hiddens=[64, 128, 1024])

    global_feature = GlobalMaxPooling1D()(features2[-1])
    global_feature = Lambda(expand_dim, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(duplicate, arguments={'num_points': n_points})(global_feature)

    c = concatenate([x2, global_feature])
    c = conv_stack(c, num_hiddens=[512, 256, 128, 128])
    outputs = Convolution1D(n_classes, 1, activation='softmax')(c[-1])
    # End defining Pointnet Architecture

    ''' 
    Define Model
    '''
    model = Model(inputs=inputs, outputs=outputs)

    return model

from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
def pointnet_loss(xtran, reg_weight=0.001):
    """
    Description:
    :param NAME: TYPE, MEAN
    :return: TYPE, MEAN
    Implemented from: https://github.com/charlesq34/pointnet/blob/master/models/pointnet_seg.py
        Source to reference more: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py; https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_segmentation.py
    Usage: model.compile(loss=[pointnet_loss(xtran=learner.model.get_layer(name='xtran2'), reg_weight=0.001)], metrics=["accuracy"], optimizer=adam)
    """
    def pointnet_loss_fixed(y_true, y_pred):
        loss = categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        seg_loss = tf.reduce_mean(loss)

        xtran_shape = xtran.output_shape[1]
        # mat_diff = Lambda(matmul, output_shape=matmul_out_shape)([xtran.output, K.permute_dimensions(xtran.output, pattern=(0, 2, 1))])
        # mat_diff -= K.constant(np.eye(xtran_shape), dtype='float32')
        mat_diff = tf.matmul(xtran.output, tf.transpose(xtran.output, perm=[0, 2, 1])) - tf.constant(np.eye(xtran_shape), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)

        return seg_loss + mat_diff_loss * reg_weight

    return pointnet_loss_fixed

'''
Illustrate how to use model
'''
def main(argv=None):
    # Declare model
    model = PointNet(num_points=1024, num_classes=5)
    model.compile(optimizer=kr.optimizers.Adam(lr=lr_schedule(0)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

if __name__ == '__main__':
    main()

