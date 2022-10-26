import os

import warnings
from contextlib import contextmanager
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

import napari.layers
from napari.utils import notifications as ntf

from n2v.models import N2V, N2VConfig

from napari_n2v.resources import DOC_BIOIMAGE
from .expert_settings import get_default_settings

REF_AXES = 'TSZYXC'
NAPARI_AXES = 'TSZYXC'

PREDICT = '_denoised'
DENOISING = 'Denoised'
SAMPLE = 'Example data'


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
    CRASHED = 'crashed'
    FAILED = 'failed'


class ModelSaveMode(Enum):
    MODELZOO = 'Bioimage.io'
    TF = 'TensorFlow'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def create_config(X_patches,
                  n_epochs=100,
                  n_steps=400,
                  batch_size=16,
                  **kwargs
                  ) -> N2VConfig:
    from n2v.models import N2VConfig

    # n2v patch shape
    n2v_patch_shape = list(X_patches.shape[1:-1])

    parameters = {
        'train_steps_per_epoch': n_steps,
        'train_epochs': n_epochs,
        'train_batch_size': batch_size,
        'n2v_patch_shape': n2v_patch_shape
    }
    return N2VConfig(X_patches, **parameters, **kwargs)


def create_model(X_patches,
                 n_epochs=100,
                 n_steps=400,
                 batch_size=16,
                 model_name='n2v',
                 basedir='models',
                 updater=None,
                 expert_settings=None,
                 train=True) -> N2V:
    from n2v.models import N2V
    with cwd(get_default_path()):
        # create config
        is_3D = len(X_patches.shape) == 5
        if expert_settings is None:
            config = create_config(X_patches,
                                   n_epochs,
                                   n_steps,
                                   batch_size,
                                   **get_default_settings(is_3D))
        else:
            config = create_config(X_patches,
                                   n_epochs,
                                   n_steps,
                                   batch_size,
                                   **expert_settings.get_settings(is_3D))

        if not config.is_valid():
            # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
            # ntf.show_error('Invalid configuration.')
            ntf.show_info('Invalid configuration.')

        # create network
        model = N2V(config, model_name, basedir=basedir)

        if train:
            model.prepare_for_training(metrics={})

        # add updater
        if updater:
            model.callbacks.append(updater)

        return model


def filter_dimensions(shape_length: int, is_3D: bool) -> List[str]:
    """
    """
    axes = list(REF_AXES)
    n = shape_length

    if not is_3D:  # if not 3D, remove it from the
        axes.remove('Z')

    if n > len(axes):
        warnings.warn('Data shape length is too large.')
        return []
    else:
        all_permutations = [''.join(p) for p in permutations(axes, n)]

        # X and Y must be in each permutation and contiguous (#FancyComments)
        all_permutations = [p for p in all_permutations if ('XY' in p) or ('YX' in p)]

        if is_3D:
            all_permutations = [p for p in all_permutations if 'Z' in p]

        if len(all_permutations) == 0 and not is_3D:
            all_permutations = ['YX']

        return all_permutations


def are_axes_valid(axes: str):
    _axes = axes.upper()

    # length 0 and > 6
    if 0 > len(_axes) > 6:
        return False

    # all characters must be in REF_AXES = 'STZYXC'
    if not all([s in REF_AXES for s in _axes]):
        return False

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            return False

    # prior: X and Y contiguous (#FancyComments)
    return ('XY' in _axes) or ('YX' in _axes)


def build_modelzoo(path: Union[str, Path], weights: str, inputs, outputs, tf_version: str, axes='byxc'):
    import os
    from bioimageio.core.build_spec import build_model

    assert path.endswith('.bioimage.io.zip'), 'Path must end with .bioimage.io.zip'

    tags_dim = '3d' if len(axes) == 5 else '2d'
    doc = DOC_BIOIMAGE

    head, _ = os.path.split(weights)
    head = os.path.join(os.path.normcase(head), "config.json")
    build_model(weight_uri=weights,
                test_inputs=[inputs],
                test_outputs=[outputs],
                input_axes=[axes],
                output_axes=[axes],
                output_path=path,
                name='Noise2Void',
                description='Self-supervised denoising.',
                authors=[{'name': "Tim-Oliver Buchholz"}, {'name': "Alexander Krull"}, {'name': "Florian Jug"}],
                license="BSD-3-Clause",
                documentation=os.path.abspath(doc),
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
                tensorflow_version=tf_version,
                attachments={"files": head}
                )


def list_diff(l1, l2):
    """
    Return the difference of two lists.
    :param l1:
    :param l2:
    :return: list of elements in l1 that are not in l2.
    """
    return list(set(l1) - set(l2))


# TODO swap order ref_axes and axes_in
def get_shape_order(shape_in, ref_axes, axes_in):
    """
    Return the new shape and axes order of x, if the axes were to be ordered according to
    the reference axes.

    :param shape_in:
    :param ref_axes: Reference axes order (string)
    :param axes_in: New axes as a list of strings
    :return:
    """
    # build indices look-up table: indices of each axe in `axes`
    indices = [axes_in.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes_in[ind] for ind in indices]
    new_shape = tuple([shape_in[ind] for ind in indices])

    return new_shape, ''.join(new_axes), indices


def reshape_data(x, axes: str):
    """
    Reshape the data to 'SZYXC' or 'SYXC', merging 'S' and 'T' channels if necessary.
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
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, REF_AXES, _axes)

    # if S is not in the list of axes, then add a singleton S
    if 'S' not in new_axes:
        new_axes = 'S' + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

        # need to change the array of indices
        indices = [0] + [1 + i for i in indices]

    # reshape by moving axes
    destination = [i for i in range(len(indices))]
    _x = np.moveaxis(_x, indices, destination)

    # remove T if necessary
    if 'T' in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace('T', '')

        # reshape S and T together
        _x = _x.reshape(new_x_shape)

    # add channel
    if 'C' not in new_axes:
        _x = _x[..., np.newaxis]
        new_axes = new_axes + 'C'

    return _x, new_axes


def reshape_napari(x, axes_in: str):
    """
    Reshape the data according to the napari axes order (or any order if axes_out) it set.
    """
    _x = x
    _axes = axes_in

    # sanity checks
    if 'X' not in axes_in or 'Y' not in axes_in:
        raise ValueError('X or Y dimension missing in axes.')

    if len(_axes) != len(_x.shape):
        raise ValueError('Incompatible data and axes.')

    assert len(list_diff(list(_axes), list(REF_AXES))) == 0  # all axes are part of REF_AXES

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, NAPARI_AXES, _axes)

    # reshape by moving the axes
    destination = [i for i in range(len(indices))]
    _x = np.moveaxis(_x, indices, destination)

    return _x, new_axes


def get_size_from_shape(layer: napari.layers.Layer, axes):
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


def get_images_count(path: Union[str, Path]):
    images_path = Path(path)

    return len([f for f in images_path.glob('*.tif*')])


# TODO write tests
def get_napari_shapes(shape_in, axes_in) -> Tuple[int]:
    """
    Transform shape into what N2V expect and return the denoised and segmented output shapes in napari axes order.

    :param shape_in:
    :param axes_in:
    :return:
    """
    # shape and axes for DenoiSeg
    shape_n2v, axes_n2v, _ = get_shape_order(shape_in, REF_AXES, axes_in)

    # shape and axes for napari
    shape_out, _, _ = get_shape_order(shape_n2v, NAPARI_AXES, axes_n2v)

    return shape_out


def get_default_path():
    return Path(Path.home(), ".napari", "N2V").absolute()


@contextmanager
def cwd(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
