import os
import warnings
from pathlib import Path

import numpy as np
from queue import Queue

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from napari.qt.threading import thread_worker

import napari.utils.notifications as ntf
from tensorflow.python.framework.errors_impl import (
    ResourceExhaustedError,
    NotFoundError,
    UnknownError,
    InternalError,
    InvalidArgumentError
)

from napari_n2v.utils import (
    cwd,
    get_default_path,
    UpdateType,
    State,
    create_model,
    reshape_data,
    load_model,
    load_and_reshape
)


class Updater(Callback):
    def __init__(self):
        super().__init__()
        self.queue = Queue(10)
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.queue.put({UpdateType.EPOCH: self.epoch + 1})

    def on_epoch_end(self, epoch, logs=None):
        self.queue.put({UpdateType.LOSS: (self.epoch, logs['loss'], logs['val_loss'])})

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        self.queue.put({UpdateType.BATCH: self.batch + 1})

    def on_train_end(self, logs=None):
        self.queue.put(UpdateType.DONE)

    def on_train_crashed(self):
        self.queue.put(UpdateType.CRASHED)

    def stop_training(self):
        self.model.stop_training = True


# TODO this method has become confusing, we need more visibility for the events (crashing, failing to start, done..etc.)
@thread_worker(start_thread=False)
def train_worker(widget, pretrained_model=None, expert_settings=None):
    import threading

    # create updater
    updater = Updater()

    # get other parameters
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.get_batch_size()
    patch_XY = widget.get_patch_XY()
    patch_Z = widget.get_patch_Z()

    # patch shape
    if widget.is_3D:
        patch_shape = (patch_Z, patch_XY, patch_XY)
    else:
        patch_shape = (patch_XY, patch_XY)

    # get data
    ntf.show_info('Loading data')
    _x_train, _x_val, new_axes = load_images(widget)  # images are reshaped
    widget.x_train, widget.x_val, widget.new_axes = _x_train, _x_val, new_axes  # save for prediction

    # prepare data
    ntf.show_info('Shaping data')
    if expert_settings is None:
        X_train, X_val = prepare_data(_x_train, _x_val, patch_shape)
    else:
        # if structN2V we should augment only by flip along the right directions, currently no augmentation
        # augment if there is not structN2V mask and if augmentations are required
        X_train, X_val = prepare_data(_x_train,
                                      _x_val,
                                      patch_shape,
                                      augment=(not expert_settings.has_mask()) and expert_settings.use_augmentation(),
                                      n_val=expert_settings.get_val_size())

    # create model
    ntf.show_info('Creating model')
    with cwd(get_default_path()):
        model_name = 'n2v_3D' if widget.is_3D else 'n2v_2D'
        base_dir = Path('models')

        # save model weights path for later prediction
        widget.weights_path = Path(base_dir, model_name, 'weights_best.h5').absolute()

        try:
            model = create_model(X_train,
                                 n_epochs,
                                 n_steps,
                                 batch_size,
                                 model_name,
                                 base_dir.absolute(),
                                 updater,
                                 expert_settings=expert_settings)
        except InternalError as e:
            print(e.message)
            # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
            # ntf.show_error(e.message)
            ntf.show_info(e.message)
            warnings.warn('InternalError could be caused by the GPU already being used by another process.')

            # stop the training process gracefully
            yield {UpdateType.FAILED: ''}  # todo silly to return a dict just for a key
            return

        # if we use a pretrained model (just trained or loaded)
        try:
            if pretrained_model:
                model.keras_model.set_weights(pretrained_model.keras_model.get_weights())
            elif expert_settings is not None and expert_settings.has_model():
                # TODO check if models are compatible
                new_model = load_model(expert_settings.get_model_path())
                model.keras_model.set_weights(new_model.keras_model.get_weights())
        except ValueError as e:
            print(str(e))
            # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
            # ntf.show_error(str(e))
            ntf.show_info(str(e))
            warnings.warn('ValueError could be caused by incompatible weights and model.')

            # stop the training process gracefully
            yield {UpdateType.FAILED: ''}  # todo silly to return a dict just for a key
            return

        ntf.show_info('Start training')
        training = threading.Thread(target=train, args=(model, X_train, X_val, updater))
        training.start()

    # loop looking for update events
    while True:
        update = updater.queue.get(True)

        if update == UpdateType.DONE or update == UpdateType.CRASHED:
            break
        elif widget.state != State.RUNNING:
            updater.stop_training()
            break
        else:
            yield update

    widget.model = model
    widget.tf_version = tf.__version__

    # save input/output for bioimage.io
    # TODO here TF will throw an error if the GPU is busy (UnknownError). Is there a way to gracefully escape it?
    example = X_val[np.newaxis, 0, ...].astype(np.float32)
    with cwd(get_default_path()):
        widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
        widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
        np.save(widget.inputs, example)

        try:
            np.save(widget.outputs, model.predict(example, new_axes, tta=False))
        except (NotFoundError, UnknownError) as e:
            msg = 'NotFoundError or UnknownError can be caused by an improper loading of cudnn, try restarting.'
            train_error(updater, e.message, msg)

    ntf.show_info('Done')


def check_napari_data(x_train, x_val, axes: str):
    """

    :param x_train:
    :param x_val:
    :param axes:
    :return:
    """

    if len(axes) != len(x_train.shape):
        raise ValueError('Train images dimensions and axes are incompatible.')

    if x_val is not None and len(axes) != len(x_val.shape):
        raise ValueError('Val images dimensions and axes are incompatible.')

    if x_val is not None and len(x_train.shape) != len(x_val.shape):
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
        _x_train, new_axes = load_and_reshape(path_train_X, axes)

        # TODO if this is a path, use pathlib
        if path_val_X == '' or path_val_X == path_train_X:
            _x_val = None
        else:
            _x_val, _ = load_and_reshape(path_val_X, axes)

        return _x_train, _x_val, new_axes

    else:  # from layers
        x_train = widget.img_train.value.data

        # remember scale for potentially showing the prediction
        widget.scale = widget.img_train.value.scale

        # if the val combobox is not empty and train != val
        if widget.img_val.value is not None and\
                widget.img_train.value.name != widget.img_val.value.name:
            x_val = widget.img_val.value.data
        else:
            x_val = None

        _x_train, _x_val, new_axes = load_data_layers(x_train, x_val, axes)

        return _x_train, _x_val, new_axes


def prepare_data(x_train, x_val, patch_shape=(64, 64), augment=True, n_val=5):
    """
    `x_train` and `x_val` can be np.arrays or tuple(list[np.arrays], list[str])
    """
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    # create data generator
    data_gen = N2V_DataGenerator()

    # train patches
    _x_train = [x_train] if type(x_train) != tuple else x_train[0]

    # sanity check
    if type(_x_train) is list:
        if patch_shape[0] > _x_train[0].shape[1]:
            raise ValueError('Patch size too large for data size.')
    else:
        if patch_shape[0] > _x_train.shape[1]:
            raise ValueError('Patch size too large for data size.')

    X_train_patches = data_gen.generate_patches_from_list(_x_train, shape=patch_shape, shuffle=True, augment=augment)

    if x_val is None:
        X_val_patches = X_train_patches[-n_val:]
        X_train_patches = X_train_patches[:-n_val]
    else:
        _x_val = [x_val] if type(x_val) != tuple else x_val[0]
        X_val_patches = data_gen.generate_patches_from_list(_x_val, shape=patch_shape, augment=augment)

    print(f'Train patches: {X_train_patches.shape}')
    print(f'Val patches: {X_val_patches.shape}')

    return X_train_patches, X_val_patches


def train_error(updater, args, msg: str):
    # TODO all necessary?
    # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
    # ntf.show_error(msg)
    ntf.show_info(msg)
    warnings.warn(msg)
    print(args)
    updater.on_train_crashed()


def train(model, X_patches, X_val_patches, updater):
    try:
        model.train(X_patches, X_val_patches)

    except AssertionError as e:
        # TODO there's probably a lot more than that
        msg = 'AssertionError can be caused by n2v masked pixel % being too low'
        train_error(updater, e.args, msg)
    except (MemoryError, InternalError) as e:
        msg = 'MemoryError or InternalError can be an OOM error on the GPU (reduce batch and/or patch size, ' \
              'close other processes). '
        train_error(updater, str(e), msg)
    except InvalidArgumentError as e:
        msg = 'InvalidArgumentError can be the result of a mismatch between shapes in the model, check input dims.'
        train_error(updater, e.message, msg)
    except ResourceExhaustedError as e:
        msg = 'ResourceExhaustedError can be an OOM error on the GPU (reduce batch and/or patch size)'
        train_error(updater, e.message, msg)
    except (NotFoundError, UnknownError) as e:
        msg = 'NotFoundError or UnknownError can be caused by an improper loading of cudnn, try restarting.'
        train_error(updater, e.message, msg)

    # TODO add other possible errors and general error catching
