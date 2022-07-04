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
    create_config
)


@thread_worker(start_thread=False)
def prediction_after_training_worker(widget):
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
        widget.pred_train[i, ...], _ = reshape_napari(_x, axes[1:])

        counter += 1
        yield {UpdateType.PRED: counter}

    # check if there is validation data
    if widget.x_val is not None:
        _x_val = widget.x_val

        # denoised val images
        for i in range(_x_val.shape[0]):
            _x = model.predict(_x_val[i, ...].astype(np.float32), axes=axes[1:])

            # reshape for napari
            widget.pred_val[i, ...], _ = reshape_napari(_x, axes[1:])

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
        yield from _run_prediction(widget, axes, images)
    else:
        yield from _run_lazy_prediction(widget, axes, images)


def _run_prediction(widget, model, axes, images):
    def generator(data, axes_order):
        """

        :param data:
        :param axes_order:
        :return:
        """
        if type(data) == list:
            yield len(data)
            for j, d in enumerate(data):
                _data, _axes = reshape_data(d, axes_order)
                yield _data, _axes, j
        else:
            _data, _axes = reshape_data(data, axes_order)
            yield _data.shape[0]

            for k in range(_data.shape[0]):
                yield _data[np.newaxis, k, ...], _axes, k

    gen = generator(images, axes)
    n_img = next(gen)
    yield {UpdateType.N_IMAGES: n_img}

    # create model
    model_name = 'n2v'
    base_dir = 'models'
    is_list = False
    if type(images) == list:
        is_list = True
        model = create_model(images[0], 1, 1, 1, model_name, base_dir, train=False)
    else:
        model = create_model(images, 1, 1, 1, model_name, base_dir, train=False)

    # load model weights
    weight_name = widget.load_button.Model.value
    assert len(weight_name.name) > 0, 'Model path cannot be empty.'
    load_weights(model, weight_name)

    while True:
        t = next(gen)

        if t is not None:
            _x, new_axes, i = t

            # update config for std and mean
            model.config = create_config(_x, 1, 1, 1, model_name, base_dir, train=False)

            # yield image number + 1
            yield {UpdateType.IMAGE: i + 1}

            # predict
            prediction = model.predict(_x, axes=new_axes)[0, ...]

            # update the layer in napari
            widget.denoi_prediction[i, ...] = prediction

            # check if stop requested
            if widget.state != State.RUNNING:
                break

    # update done
    yield {UpdateType.DONE}


def _run_lazy_prediction(widget, model, axes, generator):
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
                weight_name = widget.load_button.Model.value
                assert len(weight_name.name) > 0, 'Model path cannot be empty.'
                load_weights(model, weight_name)
            else:
                # update config for std and mean
                model.config = create_config(image, 1, 1, 1, model_name, base_dir, train=False)

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
