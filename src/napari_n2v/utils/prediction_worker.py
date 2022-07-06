from pathlib import Path

import numpy as np

from napari.qt.threading import thread_worker
from tifffile import imwrite

from napari_n2v.utils import (
    UpdateType,
    create_model,
    reshape_napari,
    lazy_load_generator,
    load_from_disk,
    load_weights,
    reshape_data,
    State,
    create_config,
    get_napari_shapes
)


@thread_worker(start_thread=False)
def prediction_after_training_worker(widget):
    """

    """
    # TODO probably doesn't work if images are lists from the disk
    model = widget.model

    # get images
    _x_train = widget.x_train

    # get axes
    axes = widget.new_axes

    # denoise training images
    counter = 0
    for i in range(_x_train.shape[0]):
        _x = model.predict(_x_train[i, ...].astype(np.float32), axes=axes[1:])

        # reshape for napari
        widget.pred_train[i, ...] = reshape_napari(_x, axes[1:])[0].squeeze()

        counter += 1
        yield {UpdateType.PRED: counter}

    # check if there is validation data
    if widget.x_val is not None:
        _x_val = widget.x_val

        # denoised val images
        for i in range(_x_val.shape[0]):
            _x = model.predict(_x_val[i, ...].astype(np.float32), axes=axes[1:])

            # reshape for napari
            widget.pred_val[i, ...] = reshape_napari(_x, axes[1:])[0].squeeze()

            counter += 1
            yield {UpdateType.PRED: counter}

    yield UpdateType.DONE


@thread_worker(start_thread=False)
def prediction_worker(widget):
    # get info from widget
    is_from_disk = widget.load_from_disk
    is_lazy_loading = widget.lazy_loading.isChecked()

    # get axes
    axes = widget.axes_widget.get_axes()

    # grab images
    if is_from_disk:
        if is_lazy_loading:
            images, n_img = lazy_load_generator(widget.images_folder.get_folder())
            assert n_img > 0
        else:
            images = load_from_disk(widget.images_folder.get_folder(), axes)
            assert len(images.shape) > 0
    else:
        images = widget.images.value.data
        assert len(images.shape) > 0

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(widget, axes, images)
    elif is_from_disk and type(images) == tuple:  # load images from disk with different sizes
        yield from _run_prediction_to_disk(widget, axes, images)
    else:
        yield from _run_prediction(widget, axes, images, is_from_disk)


def _run_prediction(widget, axes, images, is_from_disk):
    # reshape data
    _data, _axes = reshape_data(images, axes)
    yield {UpdateType.N_IMAGES: _data.shape[0]}

    # if the images were loaded from disk, the layers in napari have the wrong shape
    if is_from_disk:
        shape_denoised = get_napari_shapes(_data.shape, _axes)
        widget.denoi_prediction = np.zeros(shape_denoised, dtype=np.float32)

    # create model
    model_name = 'n2v'
    base_dir = 'models'
    model = create_model(_data, 1, 1, 1, model_name, base_dir, train=False)

    # load model weights
    weight_path = widget.get_model_path()
    if not Path(weight_path).exists():
        raise ValueError('Invalid model path.')

    load_weights(model, weight_path)

    for i_slice in range(_data.shape[0]):
        _x = _data[np.newaxis, i_slice, ...]  # replace S dimension with singleton

        # update config for std and mean
        model.config = create_config(_x, 1, 1, 1)

        # yield image number + 1
        yield {UpdateType.IMAGE: i_slice + 1}

        # predict
        prediction = model.predict(_x, axes=_axes)

        # update the layer in napari
        widget.denoi_prediction[i_slice, ...] = reshape_napari(prediction, _axes)[0].squeeze()

        # check if stop requested
        if widget.state != State.RUNNING:
            break

    # update done
    yield {UpdateType.DONE}


def _run_prediction_to_disk(widget, axes, images):
    def generator(data, axes_order):
        """

        :param data:
        :param axes_order:
        :return:
        """
        yield len(data[0])
        counter = 0
        for im, f in zip(*data):
            # reshape from napari to S(Z)YXC
            _data, _axes = reshape_data(im, axes_order)
            counter += counter + 1
            yield _data, f, _axes, counter

    gen = generator(images, axes)
    n_img = next(gen)
    yield {UpdateType.N_IMAGES: n_img}

    # create model
    model_name = 'n2v'
    base_dir = 'models'
    model = create_model(reshape_data(images[0], axes)[0], 1, 1, 1, model_name, base_dir, train=False)

    # load model weights
    weight_path = widget.get_model_path()
    if not Path(weight_path).exists():
        raise ValueError('Invalid model path.')

    load_weights(model, weight_path)

    while True:
        t = next(gen)

        if t is not None:
            _x, _f, new_axes, i = t

            # update config for std and mean
            model.config = create_config(_x, 1, 1, 1)

            # yield image number + 1
            yield {UpdateType.IMAGE: i + 1}

            # predict
            prediction = model.predict(_x, axes=new_axes)

            # save to the disk
            new_file_path_denoi = Path(_f.parent, _f.stem + '_denoised' + _f.suffix)
            imwrite(new_file_path_denoi, prediction)

            # check if stop requested
            if widget.state != State.RUNNING:
                break

    # update done
    yield {UpdateType.DONE}


def _run_lazy_prediction(widget, axes, generator):
    model = None
    while True:
        next_tuple = next(generator, None)

        if next_tuple is not None:
            image, file, i = next_tuple

            yield {UpdateType.IMAGE: i}

            if i == 0:
                # create model
                model_name = 'n2v'
                base_dir = 'models'
                model = create_model(image, 1, 1, 1, model_name, base_dir, train=False)

                # load model weights
                weight_path = widget.get_model_path()
                if not Path(weight_path).exists():
                    raise ValueError('Invalid model path.')

                load_weights(model, weight_path)
            else:
                # update config for std and mean
                model.config = create_config(image, 1, 1, 1)

            # reshape data
            x, new_axes = reshape_data(image, axes)

            # run prediction
            prediction = model.predict(x, axes=new_axes)[0, ...]

            # save predictions
            new_file_path_denoi = Path(file.parent, file.stem + '_denoised' + file.suffix)
            imwrite(new_file_path_denoi, prediction)

            # check if stop requested
            if widget.state != State.RUNNING:
                break
        else:
            break

    # update done
    yield {UpdateType.DONE}
