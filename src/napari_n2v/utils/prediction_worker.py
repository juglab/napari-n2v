import numpy as np
from napari.qt.threading import thread_worker
from napari_n2v.utils import Updates, create_model


@thread_worker(start_thread=False)
def prediction_after_training_worker(widget):
    # TODO probably doesn't work if images are lists from the disk
    model = widget.model

    # get images
    _x_train = widget.x_train

    # get axes
    axes = widget.axes_widget.get_axes()

    # denoise training images
    counter = 0
    for i in range(_x_train.shape[0]):
        widget.pred_train[i, ...] = model.predict(_x_train[i, ...].astype(np.float32), axes=axes[1:], n_tiles=10)
        counter += 1
        yield {Updates.PRED: counter}

    # check if there is validation data
    if widget.x_val is not None:
        _x_val = widget.x_val

        # denoised val images
        for i in range(_x_val.shape[0]):
            widget.pred_val[i, ...] = model.predict(_x_val[i, ...].astype(np.float32), axes=axes[1:], n_tiles=10)
            counter += 1
            yield {Updates.PRED: counter}

    yield Updates.DONE


@thread_worker(start_thread=False)
def prediction_worker(widget):
    import os
    import threading

    # TODO lazy load

    # TODO remove (just used because I currently cannot use the GPU)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    # get images
    images = widget.images.value.data

    # get other parameters
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()
    patch_XY = widget.patch_XY_spin.value()
    patch_Z = widget.patch_Z_spin.value()

    # patch shape
    is_3d = widget.checkbox_3d.isChecked()
    if is_3d:
        patch_shape = (patch_Z, patch_XY, patch_XY)
    else:
        patch_shape = (patch_XY, patch_XY)

    # prepare data
    X_train, X_val = None, None #prepare_data(images, None, patch_shape)

    # create model
    if is_3d:
        model_name = 'n2v_3D'
    else:
        model_name = 'n2v_2D'
    base_dir = 'models'
    model = create_model(X_train, n_epochs, n_steps, batch_size, model_name, base_dir, None)
    widget.weights_path = os.path.join(base_dir, model_name, 'weights_best.h5')

    # TODO: predict images and yield progress to the progress bar
