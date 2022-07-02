




@thread_worker(start_thread=False)
def train_worker(widget: TrainWidget, pretrained_model=None):
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


def train(model, X_patches, X_val_patches):
    model.train(X_patches, X_val_patches)



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