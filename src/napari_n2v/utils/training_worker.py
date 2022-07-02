import os
import numpy as np
from napari.qt.threading import thread_worker
from napari_n2v.utils import Updater, Updates, State, prepare_data


@thread_worker(start_thread=False)
def train_worker(widget, pretrained_model=None):
    import threading

    # TODO remove (just used because I currently cannot use the GPU)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    # get images
    if widget.img_train.value == widget.img_val.value:
        train_image = widget.img_train.value.data
        validation_image = None
    else:
        train_image = widget.img_train.value.data
        validation_image = widget.img_val.value.data

    # create updated
    updater = Updater()

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
    X_train, X_val = prepare_data(train_image, validation_image, patch_shape)

    # create model
    if is_3d:
        model_name = 'n2v_3D'
    else:
        model_name = 'n2v_2D'
    base_dir = 'models'
    model = create_model(X_train, n_epochs, n_steps, batch_size, model_name, base_dir, updater)
    widget.weights_path = os.path.join(base_dir, model_name, 'weights_best.h5')

    # if we use a pretrained model (just trained or loaded)
    if pretrained_model:
        # TODO: how to make sure the two are compatible? For instance, unchecking the 3D leads to different models
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
    print(example.shape)
    widget.inputs = os.path.join(widget.model.basedir, 'inputs.npy')
    widget.outputs = os.path.join(widget.model.basedir, 'outputs.npy')
    np.save(widget.inputs, example)

    if is_3d:
        example_dims = 'SZYXC'
        print('3D')
    else:
        example_dims = 'SYXC'
    print(example_dims)
    np.save(widget.outputs, model.predict(example, example_dims, tta=False))


def train(model, X_patches, X_val_patches):
    model.train(X_patches, X_val_patches)



