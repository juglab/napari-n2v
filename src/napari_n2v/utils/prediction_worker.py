import os
import warnings
from pathlib import Path

import numpy as np
from tensorflow.python.framework.errors_impl import UnknownError

from tifffile import imwrite
from napari.qt.threading import thread_worker
from napari.utils import notifications as ntf
from magicgui.types import PathLike

from napari_n2v.utils import (
    UpdateType,
    lazy_load_generator,
    load_from_disk,
    reshape_data,
    State,
    load_model
)

from napari_tools_menu import register_function
from napari_time_slicer import time_slicer


# TODO: setup.cfg defines the wrong entry point probably
@register_function(menu="Filtering / noise removal > Apply N2V denoiser")
@time_slicer
def apply_n2v(image: "napari.types.ImageData",
              model_filename: PathLike = "my_n2v_model",
              number_of_tiles: int = 4,
              ) -> "napari.types.ImageData":
    """
    """
    model_path = Path(model_filename)
    if not model_path.exists():
        raise Exception('Model not found')

    # load the model
    model = load_model(model_path)

    # check image shape
    if len(image.shape) == 2:
        axes = "YXC"
        tiles = (number_of_tiles, number_of_tiles, 1)
    elif len(image.shape) == 3:
        axes = "ZYXC"
        tiles = (number_of_tiles, number_of_tiles, number_of_tiles, 1)
    else:
        raise ValueError("Only 2D and 3D data supported.")

    # run prediction
    predicted_image = model.predict(image[..., np.newaxis], axes=axes, n_tiles=tiles)

    return predicted_image[..., 0]


@thread_worker(start_thread=False)
def prediction_after_training_worker(widget):
    # get model
    model = widget.model

    # get tiling
    is_tiled = widget.is_tiling_checked()
    n_tiles = widget.get_n_tiles()

    # get images (already in SXYC or SZYXC shape), np.array or list[np.array]
    _x_train = widget.x_train

    # prepare prediction with the right shape, set to none if _x_train is a tuple
    widget.pred_train = np.zeros(_x_train.shape) if type(_x_train) != tuple else None

    # get size of the training images (either first dim or length of list)
    size_x = _x_train.shape[0] if type(_x_train) != tuple else len(len(_x_train[0]))

    # get axes
    axes = widget.new_axes

    # predict training data
    yield from _predict(widget, model, _x_train, axes, widget.pred_train, is_tiled, n_tiles)

    # if we are loading from napari layers and there are channels
    if widget.pred_train is not None and widget.pred_train.shape[-1] > 1:
        # normalise to the range 0-1
        # TODO is there really no better way?
        if widget.pred_train.max() > 255:
            widget.pred_train = widget.pred_train / 65535
        else:
            widget.pred_train = widget.pred_train / 255

    if widget.pred_train is not None:
        # remove singleton dims
        widget.pred_train = widget.pred_train.squeeze()

    # check if there is validation data
    if widget.x_val is not None:
        # get data
        _x_val = widget.x_val

        # prepare prediction with the right shape, set to none if _x_train is a tuple
        widget.pred_val = np.zeros(_x_val.shape) if type(_x_val) != tuple else None

        # predict training data
        yield from _predict(widget,
                            model,
                            _x_val,
                            axes,
                            widget.pred_val,
                            is_tiled,
                            n_tiles,
                            counter_offset=size_x)

        # if we are loading from napari layers and there are channels
        if widget.pred_val is not None and widget.pred_val.shape[-1] > 1:
            # normalise to the range 0-1
            # TODO is there really no better way?
            if widget.pred_val.max() > 255:
                widget.pred_val = widget.pred_val / 65535
            else:
                widget.pred_val = widget.pred_val / 255

        if widget.pred_val is not None:
            # remove singleton dims
            widget.pred_val = widget.pred_val.squeeze()

    yield UpdateType.DONE


def _predict(widget, model, data, axes, prediction, is_tiled=False, n_tiles=4, counter_offset=0):
    # denoise training images
    if type(data) == tuple:  # data is a tuple( list[np.array], list[Path] )
        yield from _predict_list(widget, model, data, axes, is_tiled, n_tiles, counter_offset)
    else:
        yield from _predict_np(widget, model, data, axes, prediction, is_tiled, n_tiles, counter_offset)


def _predict_list(widget, model, data, axes, is_tiled=False, n_tiles=4, counter_offset=0):
    for im_ind, im in enumerate(data[0]):
        file = data[1][im_ind]

        prediction = np.zeros(im.shape)
        for i in range(im.shape[0]):
            if is_tiled:
                tiles = (len(axes) - 2) * (n_tiles,) + (1,)

                _x = model.predict(im[i, ...].astype(np.float32), axes=axes[1:], n_tiles=tiles)
            else:
                _x = model.predict(im[i, ...].astype(np.float32), axes=axes[1:])

            # add prediction
            prediction[i, ...] = _x

            if widget.state == State.IDLE:
                break

        yield {UpdateType.PRED: counter_offset + im_ind + 1}

        # save images
        parent = Path(file.parent, 'results')
        if not parent.exists():
            os.mkdir(parent)

        new_file_path_denoi = Path(parent, file.stem + '_denoised' + file.suffix)
        imwrite(new_file_path_denoi, prediction.squeeze())


def _predict_np(widget, model, data, axes, prediction, is_tiled=False, n_tiles=4, counter_offset=0):
    for i in range(data.shape[0]):
        if is_tiled:
            tiles = (len(axes) - 2) * (n_tiles,) + (1,)

            _x = model.predict(data[i, ...].astype(np.float32), axes=axes[1:], n_tiles=tiles)
        else:
            _x = model.predict(data[i, ...].astype(np.float32), axes=axes[1:])

        yield {UpdateType.PRED: counter_offset + i + 1}

        # add prediction
        prediction[i, ...] = _x

        if widget.state == State.IDLE:
            break


@thread_worker(start_thread=False)
def prediction_worker(widget):
    # from disk, lazy loading and threshold
    is_from_disk = widget.load_from_disk
    is_lazy_loading = widget.lazy_loading.isChecked()

    is_tiled = widget.is_tiling_checked()
    n_tiles = widget.get_n_tiles()

    # get axes
    axes = widget.axes_widget.get_axes()

    # create model
    weight_path = widget.get_model_path()
    try:
        model = load_model(weight_path)
    except ValueError as e:
        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
        # ntf.show_error('Error loading model weights.')
        ntf.show_info('Error loading model weights.')
        print(e)

        yield {UpdateType.DONE}
        return

    # grab images
    if is_from_disk:
        if is_lazy_loading:
            images, n_img = lazy_load_generator(widget.images_folder.get_folder())

            if n_img == 0:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error('No image found.')
                ntf.show_info('No image found.')
                yield {UpdateType.DONE}
                return

            new_axes = axes
        else:
            # here images is either a tuple of lists ([imgs], [files]) or a numpy array
            images, new_axes = load_from_disk(widget.images_folder.get_folder(), axes)

            if type(images) == tuple and len(images[0]) == 0:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error('No image found.')
                ntf.show_info('No image found.')
                yield {UpdateType.DONE}
                return
    else:
        images = widget.images.value.data
        new_axes = axes

        # remember scales for showing the final images
        widget.scale = widget.images.value.scale

    # common parameters list
    parameters = {'widget': widget,
                  'model': model,
                  'axes': new_axes,
                  'images': images,
                  'is_tiled': is_tiled,
                  'n_tiles': n_tiles}

    if is_from_disk and is_lazy_loading:
        # yield generator size
        yield {UpdateType.N_IMAGES: n_img}
        yield from _run_lazy_prediction(**parameters)
    elif is_from_disk and type(images) == tuple:  # load images from disk with different sizes
        yield from _run_prediction_to_disk(**parameters)
    else:  # numpy array (from napari.layers or from the disk)
        yield from _run_prediction(**parameters)


# TODO: how about doing is_tiled = n_tiles != 1 ?
def _run_prediction(widget, model, axes, images, is_tiled=False, n_tiles=4):
    """
    `images` is either a napari.layer or a numpy array.

    """
    # reshape data
    try:
        _data, new_axes = reshape_data(images, axes)
        yield {UpdateType.N_IMAGES: _data.shape[0]}
    except ValueError as e:
        print(str(e))
        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
        # ntf.show_error(str(e))
        ntf.show_info(str(e))

        # fail elegantly
        return

    # create a numpy array to store the results
    # (note: napari and N2V have different axes orders)
    shape_out = _data.shape
    predict_all = np.zeros(shape_out, dtype=np.float32)

    for i_slice in range(_data.shape[0]):
        # check if stop requested
        if widget.state != State.RUNNING:
            break

        # proceed with prediction
        _x = _data[np.newaxis, i_slice, ...]  # replace S dimension with singleton

        # yield image number + 1
        yield {UpdateType.IMAGE: i_slice + 1}

        # predict
        try:
            if is_tiled:
                # TODO: why is this different than in the other functions, why is sample S's (1,) allowed?
                tiles = (1,) + (len(new_axes) - 2) * (n_tiles,) + (1,)

                predict_all[i_slice, ...] = model.predict(_x, axes=new_axes, n_tiles=tiles)
            else:
                predict_all[i_slice, ...] = model.predict(_x, axes=new_axes)
        except UnknownError as e:
            msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
            # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
            # ntf.show_error(msg)
            ntf.show_info(msg)
            warnings.warn(msg)
            print(e.message)
            break

    # if there are channels, we normalize to the range 0-1
    if predict_all.shape[-1] > 1:
        norm_factor = 65535 if predict_all.max() > 255 else 255
    else:
        norm_factor = 1

    widget.denoi_prediction = (predict_all / norm_factor).squeeze()

    # update done
    yield {UpdateType.DONE}


def _run_prediction_to_disk(widget, model, axes, images, is_tiled=False, n_tiles=4):
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
            try:
                _data, _axes = reshape_data(im, axes_order)

                counter = counter + 1
                yield _data, f, _axes, counter

            except ValueError:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error(f'Wrong image shapes {f.stem} {im.shape}')
                ntf.show_info(f'Wrong image shapes {f.stem} {im.shape}')

    gen = generator(images, axes)

    # TODO this is a weird way to use the generator to pass its total length
    yield {UpdateType.N_IMAGES: next(gen)}

    while True:
        t = next(gen, None)

        if t is not None:
            _x, _f, new_axes, i = t

            # yield image number + 1
            yield {UpdateType.IMAGE: i + 1}

            # shape prediction
            shape_out = _x.shape
            prediction = np.zeros(shape_out, dtype=np.float32)  # (S,(Z), Y, X, C)

            for i_s in range(_x.shape[0]):
                try:
                    if is_tiled:
                        tiles = (len(new_axes) - 2) * (n_tiles,) + (1,)

                        prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:], n_tiles=tiles)
                    else:
                        prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:])

                except UnknownError as e:
                    msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
                    # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                    # ntf.show_error(msg)
                    ntf.show_info(msg)
                    warnings.warn(msg)
                    print(e.message)
                    break

            # save to the disk
            parent = Path(_f.parent, 'results')
            if not parent.exists():
                os.mkdir(parent)

            new_file_path_denoi = Path(parent, _f.stem + '_denoised' + _f.suffix)
            imwrite(new_file_path_denoi, prediction.squeeze())

            # check if stop requested
            if widget.state != State.RUNNING:
                break
        else:
            break

    # update done
    yield {UpdateType.DONE}


def _run_lazy_prediction(widget, model, axes, images, is_tiled=False, n_tiles=4):
    while True:
        next_tuple = next(images, None)

        if next_tuple is not None:
            image, file, i = next_tuple

            yield {UpdateType.IMAGE: i}

            # reshape data
            try:
                _x, new_axes = reshape_data(image, axes)

                # run prediction
                shape_out = _x.shape
                prediction = np.zeros(shape_out, dtype=np.float32)  # (S,(Z), Y, X, C)

                for i_s in range(_x.shape[0]):
                    try:
                        if is_tiled:
                            tiles = (len(new_axes) - 2) * (n_tiles,) + (1,)

                            # model.predict returns ((Z), Y, Z, C)
                            prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:], n_tiles=tiles)
                        else:
                            prediction[i_s, ...] = model.predict(_x[i_s, ...], axes=new_axes[1:])

                    except UnknownError as e:
                        msg = 'UnknownError can be a failure to load cudnn, try restarting the computer.'
                        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                        # ntf.show_error(msg)
                        ntf.show_info(msg)
                        warnings.warn(msg)
                        print(e.message)
                        break

                # save predictions
                parent = Path(file.parent, 'results')
                if not parent.exists():
                    os.mkdir(parent)

                new_file_path_denoi = Path(parent, file.stem + '_denoised' + file.suffix)
                imwrite(new_file_path_denoi, prediction.squeeze())

                # check if stop requested
                if widget.state != State.RUNNING:
                    break

            except ValueError as e:
                print(e.args)
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error(f'Wrong image shapes  {file.stem} {image.shape}')
                ntf.show_info(f'Wrong image shapes  {file.stem} {image.shape}')
        else:
            break

    # update done
    yield {UpdateType.DONE}
