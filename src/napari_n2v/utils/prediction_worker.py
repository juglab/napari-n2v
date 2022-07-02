



@thread_worker(start_thread=False)
def predict_worker(widget: TrainWidget):
    model = widget.model

    # check if it is 3D
    if widget.checkbox_3d.isChecked():
        im_dims = 'ZYX'
    else:
        im_dims = 'YX'

    # get train images
    train_image = widget.img_train.value.data

    # denoise training images
    counter = 0
    if im_dims == 'YX':
        for i in range(train_image.shape[0]):
            widget.pred_train[i, ...] = model.predict(train_image[i, ...].astype(np.float32), im_dims)
            counter += 1
            yield {Updates.PRED: counter}
    else:
        widget.pred_train = model.predict(train_image.astype(np.float32), im_dims)
        yield {Updates.PRED: 1}

    # check if there is validation data
    if widget.img_train.value != widget.img_val.value:
        val_image = widget.img_val.value.data

        # denoised val images
        if im_dims == 'YX':
            for i in range(val_image.shape[0]):
                widget.pred_val[i, ...] = model.predict(val_image[i, ...].astype(np.float32), im_dims)
                counter += 1
                yield {Updates.PRED: counter}
        else:
            widget.pred_val = model.predict(val_image.astype(np.float32), im_dims)
            yield {Updates.PRED: 2}
    yield Updates.DONE



@thread_worker(start_thread=False)
def prediction_worker(widget: PredictWidget):
    import os
    import threading

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
    X_train, X_val = prepare_data(images, None, patch_shape)

    # create model
    if is_3d:
        model_name = 'n2v_3D'
    else:
        model_name = 'n2v_2D'
    base_dir = 'models'
    model = create_model(X_train, n_epochs, n_steps, batch_size, model_name, base_dir, None)
    widget.weights_path = os.path.join(base_dir, model_name, 'weights_best.h5')

    # TODO: predict images and yield progress to the progress bar
