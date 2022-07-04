import os
import warnings
from enum import Enum
from itertools import permutations
from pathlib import Path

import numpy as np
from n2v.models import N2V
from tifffile import imread

REF_AXES = 'TSZYXC'
NAPARI_AXES = 'SCTZYX'

PREDICT = '_denoised'
DENOISING = 'Denoised'
SAMPLE = 'Sample data'


class State(Enum):
    IDLE = 0
    RUNNING = 1


class UpdateType(Enum):
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


def create_config(X_patches,
                  n_epochs=100,
                  n_steps=400,
                  batch_size=16):
    from n2v.models import N2VConfig

    n2v_patch_shape = X_patches.shape[1:-1]
    config = N2VConfig(X_patches, unet_kern_size=3, train_steps_per_epoch=n_steps, train_epochs=n_epochs,
                       train_loss='mse', batch_norm=True, train_batch_size=batch_size, n2v_perc_pix=0.198,
                       n2v_patch_shape=n2v_patch_shape, unet_n_first=96, unet_residual=True,
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=2, single_net_per_channel=False)
    return config


def create_model(X_patches,
                 n_epochs=100,
                 n_steps=400,
                 batch_size=16,
                 model_name='n2v',
                 basedir='models',
                 updater=None,
                 train=True):
    from n2v.models import N2V

    # create config
    config = create_config(X_patches,
                           n_epochs,
                           n_steps,
                           batch_size)

    # create network
    model = N2V(config, model_name, basedir=basedir)

    if train:
        model.prepare_for_training(metrics={})

    # add updater
    if updater:
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


def build_modelzoo(path, weights, inputs, outputs, tf_version, axes='byxc', doc='../resources/documentation.md'):
    import os
    from bioimageio.core.build_spec import build_model

    assert path.endswith('.bioimage.io.zip'), 'Path must end with .bioimage.io.zip'

    tags_dim = '3d' if len(axes) == 5 else '2d'

    build_model(weight_uri=weights,
                test_inputs=[inputs],
                test_outputs=[outputs],
                input_axes=[axes],
                # TODO are the axes in and out always the same? (output has 3 seg classes and 1 denoised channels)
                output_axes=[axes],
                output_path=path,
                name='Noise2Void',
                description='Self-supervised denoising.',
                authors=[{'name': "Tim-Oliver Buchholz"}, {'name': "Alexander Krull"}, {'name': "Florian Jug"}],
                license="BSD-3-Clause",
                documentation=os.path.abspath('../resources/documentation.md'),
                tags=[tags_dim, 'tensorflow', 'unet', 'denoising'],
                cite=[{'text': 'Noise2Void - Learning Denoising from Single Noisy Images',
                       'doi': "10.48550/arXiv.1811.10980"}],
                preprocessing=[[{
                    "name": "zero_mean_unit_variance",
                    "kwargs": {
                        "axes": "yx",
                        "mode": "per_dataset"
                    }
                }]],
                tensorflow_version=tf_version
                )


def load_from_disk(path, axes: str):
    """

    :param axes:
    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    images = []
    dims_agree = True
    for f in image_files:
        images.append(imread(str(f)))
        dims_agree = dims_agree and (images[0].shape == images[-1].shape)

    if dims_agree:
        if 'S' in axes:
            ind_S = axes.find('S')
            final_images = np.concatenate(images, axis=ind_S)
        else:
            final_images = np.stack(images, axis=0)
        return final_images

    return images


def list_diff(l1, l2):
    """
    Return the difference of two lists.
    :param l1:
    :param l2:
    :return: list of elements in l1 that are not in l2.
    """
    return list(set(l1) - set(l2))


def get_shape_order(x, ref_axes, axes):
    """
    Return the new shape and axes order of x, if the axes were to be ordered according to
    the reference axes.

    :param x:
    :param ref_axes: Reference axes order (string)
    :param axes: New axes as a list of strings
    :return:
    """
    # build indices look-up table: indices of each axe in `axes`
    indices = [axes.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes[ind] for ind in indices]
    new_shape = tuple([x.shape[ind] for ind in indices])

    return new_shape, ''.join(new_axes), indices


def reshape_data(x, axes: str):
    """
    """
    _x = x
    _axes = axes

    # sanity checks
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x, REF_AXES, _axes)

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace('T', '')

    # reshape
    _x = _x.reshape(new_x_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, new_axes


def reshape_napari(x, axes: str):
    """

    """
    _x = x
    _axes = axes

    # sanity checks
    if 'X' not in axes or 'Y' not in axes:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x, NAPARI_AXES, _axes)

    # reshape
    _x = _x.reshape(new_x_shape)

    return _x, new_axes


def get_size_from_shape(layer, axes):
    ind_S = axes.find('S')
    ind_T = axes.find('T')

    # layer shape
    shape = layer.data.shape

    if ind_S == -1 < ind_T:  # there is only T
        return shape[ind_T]
    elif ind_T == -1 < ind_S:  # there is only S
        return shape[ind_T]
    elif ind_T > -1 and ind_S > -1:  # there are both
        return shape[ind_T] * shape[ind_S]
    else:
        return 1


def get_images_count(path):
    images_path = Path(path)

    return len([f for f in images_path.glob('*.tif*')])


def lazy_load_generator(path):
    """

    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    def generator(file_list):
        counter = 0
        for f in file_list:
            counter = counter + 1
            yield imread(str(f)), f, counter

    return generator(image_files), len(image_files)


def load_weights(model: N2V, weights_path):
    """

    :param model:
    :param weights_path:
    :return:
    """
    _filename, file_ext = os.path.splitext(weights_path)
    if file_ext == ".zip":
        import bioimageio.core
        # we assume we got a modelzoo file
        rdf = bioimageio.core.load_resource_description(weights_path)
        weights_name = rdf.weights['keras_hdf5'].source
    else:
        # we assure we have a path to a .h5
        weights_name = weights_path

    if not Path(weights_name).exists():
        raise FileNotFoundError('Invalid path to weights.')

    model.keras_model.load_weights(weights_name)
