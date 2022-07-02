import warnings
from enum import Enum
from itertools import permutations
from queue import Queue

import numpy as np
from tensorflow.keras.callbacks import Callback

REF_AXES = 'TSZYXC'

PREDICT = '_denoised'
DENOISING = 'Denoised'


class State(Enum):
    IDLE = 0
    RUNNING = 1


class Updates(Enum):
    EPOCH = 'epoch'
    BATCH = 'batch'
    LOSS = 'loss'
    PRED = 'prediction'
    N_IMAGES = 'number of images'
    IMAGE = 'image'
    DONE = 'done'


class SaveMode(Enum):
    MODELZOO = 'Bioimage.io'
    TF = 'TensorFlow'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def prepare_data(img_train, img_val, patch_shape=(64, 64)):
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    # get images
    if len(patch_shape) == 2:
        X_train = img_train[..., np.newaxis]  # (1, S, Y, X, 1)
    else:
        X_train = img_train[np.newaxis, ..., np.newaxis]  # (1, S, Z, Y, X, 1)

    # TODO: what if Time dimension
    # create data generator
    data_gen = N2V_DataGenerator()

    # generate train patches
    print(f'Patch {patch_shape}')
    print(f'X train {X_train.shape}')
    X_train_patches = data_gen.generate_patches_from_list([X_train], shape=patch_shape, shuffle=True)

    if img_val is None:  # TODO: how to choose number of validation patches?
        X_val_patches = X_train_patches[-5:]
        X_train_patches = X_train_patches[:-5]
    else:
        if len(patch_shape) == 2:
            X_val = img_val[..., np.newaxis]
        else:
            X_val = img_val[np.newaxis, ..., np.newaxis]

        print(f'X val {X_val.shape}')
        X_val_patches = data_gen.generate_patches_from_list([X_val], shape=patch_shape, shuffle=True)

    print(f'Train patches: {X_train_patches.shape}')
    print(f'Val patches: {X_val_patches.shape}')

    return X_train_patches, X_val_patches


class Updater(Callback):
    def __init__(self):
        self.queue = Queue(10)
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.queue.put({Updates.EPOCH: self.epoch + 1})

    def on_epoch_end(self, epoch, logs=None):
        self.queue.put({Updates.LOSS: (self.epoch, logs['loss'], logs['val_loss'])})

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.queue.put({Updates.BATCH: self.batch + 1})

    def on_train_end(self, logs=None):
        self.queue.put(Updates.DONE)

    def stop_training(self):
        self.model.stop_training = True


def create_model(X_patches,
                 n_epochs=100,
                 n_steps=400,
                 batch_size=16,
                 model_name='n2v',
                 basedir='models',
                 updater=None):
    from n2v.models import N2VConfig, N2V

    # create config
    # config = N2VConfig(X_patches, unet_kern_size=3,
    #                  train_steps_per_epoch=n_steps, train_epochs=n_epochs, train_loss='mse',
    #                 batch_norm=True, train_batch_size=batch_size, n2v_perc_pix=0.198,
    #                n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=neighborhood_radius)
    n2v_patch_shape = X_patches.shape[1:-1]
    config = N2VConfig(X_patches, unet_kern_size=3, train_steps_per_epoch=n_steps, train_epochs=n_epochs,
                       train_loss='mse', batch_norm=True, train_batch_size=batch_size, n2v_perc_pix=0.198,
                       n2v_patch_shape=n2v_patch_shape, unet_n_first=96, unet_residual=True,
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=2, single_net_per_channel=False)

    # create network
    model = N2V(config, model_name, basedir=basedir)

    # add updater
    model.prepare_for_training(metrics=())
    model.callbacks.append(updater)

    return model


def filter_dimensions(shape_length, is_3D):
    """
    """
    axes = list(REF_AXES)
    axes.remove('Y')  # skip YX, constraint
    axes.remove('X')
    n = shape_length - 2

    if not is_3D:  # if not 3D, remove it from the
        axes.remove('Z')

    if n > len(axes):
        warnings.warn('Data shape length is too large.')
        return []
    else:
        all_permutations = [''.join(p) + 'YX' for p in permutations(axes, n)]

        if is_3D:
            all_permutations = [p for p in all_permutations if 'Z' in p]

        if len(all_permutations) == 0 and not is_3D:
            all_permutations = ['YX']

        return all_permutations


def are_axes_valid(axes: str):
    _axes = axes.upper()

    # length 0 and >6 are not accepted
    if 0 > len(_axes) > 6:
        return False

    # all characters must be in REF_AXES = 'STZYXC'
    if not all([s in REF_AXES for s in _axes]):
        return False

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            return False

    return True
