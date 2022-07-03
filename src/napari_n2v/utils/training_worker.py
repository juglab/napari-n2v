import os
import numpy as np
from queue import Queue

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from napari.qt.threading import thread_worker
from napari_n2v.utils import Updates, State, create_model, reshape_data


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


@thread_worker(start_thread=False)
def train_worker(widget, pretrained_model=None):
    import threading

    # create updater
    updater = Updater()

    # get other parameters
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()
    patch_XY = widget.patch_XY_spin.value()
    patch_Z = widget.patch_Z_spin.value()

    # patch shape
    if widget.is_3D:
        patch_shape = (patch_Z, patch_XY, patch_XY)
    else:
        patch_shape = (patch_XY, patch_XY)

    # get data
    _x_train, _x_val, new_axes = load_images(widget)
    widget.x_train, widget.y_train = _x_train, _x_val  # save for prediction

    # prepare data
    X_train, X_val = prepare_data(_x_train, _x_val, patch_shape)

    # create model
    model_name = 'n2v_3D' if widget.is_3D else 'n2v_2D'
    base_dir = 'models'
    model = create_model(X_train, n_epochs, n_steps, batch_size, model_name, base_dir, updater)
    widget.weights_path = os.path.join(base_dir, model_name, 'weights_best.h5')

    # if we use a pretrained model (just trained or loaded)
    if pretrained_model:
        model.keras_model.set_weights(pretrained_model.keras_model.get_weights())

    training = threading.Thread(target=train, args=(model, X_train, X_val))
    training.start()

    # loop looking for update events
    while True:
        update = updater.queue.get(True)

        if Updates.DONE == update:
            break
        elif widget.state != State.RUNNING:
            updater.stop_training()
            yield Updates.DONE
            break
        else:
            yield update

    widget.model = model
    widget.tf_version = tf.__version__

    # save input/output for bioimage.io
    example = X_val[np.newaxis, 0, ...].astype(np.float32)
    widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
    widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
    np.save(widget.inputs, example)
    np.save(widget.outputs, model.predict(example, new_axes, tta=False))


def load_data_from_disk(source, axes):
    """

    """
    from napari_n2v.utils import load_from_disk

    # load data
    _x = load_from_disk(source, axes)

    # reshape data
    # TODO test what happens when x is a list
    if type(_x) == list:
        for i, e in enumerate(_x):
            _x[i], new_axes = reshape_data(e, axes)
    else:
        if 'S' not in axes and _x.shape[0] > 1:
            new_axes = 'S' + axes

        _x, new_axes = reshape_data(_x, new_axes)

    return _x, new_axes


def check_napari_data(x_train, x_val, axes: str):
    """

    :param x_train:
    :param x_val:
    :param axes:
    :return:
    """

    if axes[-2:] != 'YX':
        raise ValueError('X and Y axes are in the wrong order.')

    if len(axes) != len(x_train.shape):
        raise ValueError('Train images dimensions and axes are incompatible.')

    if x_val is not None and len(axes) != len(x_val.shape):
        raise ValueError('Val images dimensions and axes are incompatible.')

    if x_val is not None and  len(x_train.shape) != len(x_val.shape):
        raise ValueError('Train and val images dimensions are incompatible.')


def load_data_layers(x_train, x_val, axes):
    """
    :param x_train:
    :param x_val:
    :param axes
    :return:
    """

    # sanity check on the data
    check_napari_data(x_train, x_val, axes)

    # reshape data
    _x_train, new_axes = reshape_data(x_train, axes)

    if x_val is not None:
        _x_val, _ = reshape_data(x_val, axes)
    else:
        _x_val = None

    return _x_train, _x_val, new_axes


def load_images(widget):
    """

    :param widget:
    :return:
    """
    # TODO make clearer what objects are returned

    # get axes
    axes = widget.axes_widget.get_axes()

    # get images and labels
    if widget.load_from_disk:  # from local folders
        path_train_X = widget.train_images_folder.get_folder()
        path_val_X = widget.val_images_folder.get_folder()

        # load train data
        _x_train, new_axes = load_data_from_disk(path_train_X, axes)

        if path_val_X is None and path_val_X != path_train_X:
            _x_val = None
        else:
            _x_val, _ = load_data_from_disk(path_val_X, axes)

        return _x_train, _x_val, new_axes

    else:  # from layers
        x_train = widget.img_train.value.data
        x_val = widget.img_val.value.data

        if widget.img_train.value == widget.img_val.value:
            x_val = None

        _x_train, _x_val, new_axes = load_data_layers(x_train, x_val, axes)

        return _x_train, _x_val, new_axes


def prepare_data(x_train, x_val, patch_shape=(64, 64)):
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    # create data generator
    data_gen = N2V_DataGenerator()

    # generate train patches
    _x_train = [x_train] if type(x_train) != list else x_train

    X_train_patches = data_gen.generate_patches_from_list(_x_train, shape=patch_shape, shuffle=True)

    if x_val is None:  # TODO: how to choose number of validation patches?
        X_val_patches = X_train_patches[-5:]
        X_train_patches = X_train_patches[:-5]
    else:
        _x_val = [x_val] if type(x_val) != list else x_val
        X_val_patches = data_gen.generate_patches_from_list(_x_val, shape=patch_shape, shuffle=True)

    print(f'Train patches: {X_train_patches.shape}')
    print(f'Val patches: {X_val_patches.shape}')

    return X_train_patches, X_val_patches


def train(model, X_patches, X_val_patches):
    model.train(X_patches, X_val_patches)



