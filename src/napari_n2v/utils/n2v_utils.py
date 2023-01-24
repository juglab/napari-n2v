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

from .expert_settings import get_default_settings, PixelManipulator

REF_AXES = 'TSZYXC'
NAPARI_AXES = 'TSZYXC'

PREDICT = '_denoised'
DENOISING = 'Denoised'
SAMPLE = 'Example data'


class Algorithm(Enum):
    N2V = 0
    StructN2V = 1
    N2V2 = 2


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
    KERAS = 'Keras'
    TF = 'TensorFlow'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def which_algorithm(config: N2VConfig):
    """
    Checks which algorithm the model is configured for (N2V, N2V2, structN2V).
    """
    if config.structN2Vmask is not None:
        return Algorithm.StructN2V
    elif config.n2v_manipulator == PixelManipulator.MEDIAN.value and \
            not config.unet_residual and config.blurpool and config.skip_skipone:
        return Algorithm.N2V2
    else:
        return Algorithm.N2V


def get_algorithm_details(algorithm: Algorithm):
    """
    Returns name, authors and citation related to the algorithm, formatted as expected by bioimage.io
    model builder.
    """
    if algorithm == Algorithm.StructN2V:
        name = 'structN2V'
        authors = [{'name': "Coleman Broaddus"},
                   {'name': "Alexander Krull"},
                   {'name': "Martin Weigert"},
                   {'name': "Uwe Schmidt"},
                   {'name': "Gene Myers"}]
        citation = [{'text': 'C. Broaddus, A. Krull, M. Weigert, U. Schmidt and G. Myers, \"Removing Structured Noise '
                             'with Self-Supervised Blind-Spot Networks,\" 2020 IEEE 17th International Symposium on '
                             'Biomedical Imaging (ISBI), 2020, pp. 159-163',
                     'doi': '10.1109/ISBI45749.2020.9098336'}]
    elif algorithm == Algorithm.N2V2:
        name = 'N2V2'
        authors = [{'name': "Eva Hoeck"},
                   {'name': "Tim-Oliver Buchholz"},
                   {'name': "Anselm Brachmann"},
                   {'name': "Florian Jug"},
                   {'name': "Alexander Freytag"}]
        citation = [{'text': 'E. Hoeck, T.-O. Buchholz, A. Brachmann, F. Jug and A. Freytag, '
                             '\"N2V2--Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a '
                             'Tweaked Network Architecture.\" arXiv preprint arXiv:2211.08512 (2022).',
                     'doi': '10.48550/arXiv.2211.08512'}]
    else:
        name = 'Noise2Void'
        authors = [{'name': "Alexander Krull"}, {'name': "Tim-Oliver Buchholz"}, {'name': "Florian Jug"}]
        citation = [{'text': 'A. Krull, T.-O. Buchholz and F. Jug, \"Noise2Void - Learning Denoising From Single '
                             'Noisy Images,\" 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition  '
                             '(CVPR), 2019, pp. 2124-2132',
                     'doi': '10.48550/arXiv.1811.10980'}]

    return name, authors, citation


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
    """
    Create a model.

    Warning: if Train=true, TF SavedModel bundle export will not be importable because
    of missing custom functions (e.g. loss).
    """
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
        raise ValueError(f'Incompatible data ({_x.shape}) and axes ({_axes}).')

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
        raise ValueError(f'Incompatible data ({_x.shape}) and axes ({_axes}).')

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
