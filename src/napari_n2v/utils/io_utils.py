import os
from pathlib import Path

import numpy as np

from n2v.models import N2V, N2VConfig
from typing import Union
from napari_n2v.resources import DOC_BIOIMAGE
from .n2v_utils import (
    ModelSaveMode,
    get_default_path,
    cwd,
    which_algorithm,
    get_algorithm_details
)


def save_configuration(config: N2VConfig, dir_path: Union[str, Path]):
    from csbdeep.utils import save_json

    # sanity check
    assert Path(dir_path).is_dir()

    # save
    final_path = Path(dir_path) / 'config.json'
    save_json(vars(config), final_path)


def load_configuration(path: Union[str, Path]) -> N2VConfig:
    from csbdeep.utils import load_json
    from n2v.models import N2VConfig

    # load config
    json_config = load_json(path)

    # create N2V configuration
    axes_length = len(json_config['axes'])
    n_channels = json_config['n_channel_in']

    if axes_length == 3:
        X = np.zeros((1, 8, 8, n_channels))
    else:
        X = np.zeros((1, 8, 8, 8, n_channels))

    return N2VConfig(X, **json_config)


def load_model(weight_path: Union[str, Path]) -> N2V:
    if not Path(weight_path).exists():
        raise ValueError('Invalid model path.')

    if not (Path(weight_path).parent / 'config.json').exists():
        raise ValueError('No config.json file found.')

    # load configuration
    config = load_configuration(Path(weight_path).parent / 'config.json')

    # instantiate model
    model = N2V(config, 'DenoiSeg', 'models')

    # load weights
    load_weights(model, weight_path)

    return model


def load_weights(model: N2V, weights_path: Union[str, Path]):
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


def save_modelzoo(where: Union[str, Path],
                  model: N2V,
                  axes: str,
                  input_path: str,
                  output_path: str,
                  tf_version: str):
    from napari_n2v.utils import build_modelzoo

    with cwd(get_default_path()):
        # path to weights
        weights = Path(model.logdir, 'weights_best.h5')
        if not weights.exists():
            raise FileNotFoundError('Invalid path to weights.')

        # format axes for bioimage.io
        new_axes = axes.replace('S', 'b').lower()

        if 'b' not in new_axes:
            new_axes = 'b' + new_axes

        # check algorithm (N2V, structN2V, N2V2) and get the corresponding details
        algorithm = which_algorithm(model.config)
        name, authors, cite = get_algorithm_details(algorithm)

        # check path ending
        where = str(where)
        path = where if where.endswith('.bioimage.io.zip') else where + '.bioimage.io.zip'

        # save model
        build_modelzoo(path,
                       weights,
                       input_path,
                       output_path,
                       name,
                       authors,
                       cite,
                       tf_version,
                       new_axes)

        # save configuration
        save_configuration(model.config, Path(where).parent)


def build_modelzoo(path: Union[str, Path],
                   weights: Union[str, Path],
                   inputs: str,
                   outputs: str,
                   name: str,
                   authors: list,
                   cite: list,
                   tf_version: str,
                   axes: str = 'byxc'):
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
                name=name,
                description='Self-supervised denoising.',
                authors=authors,
                license="BSD-3-Clause",
                documentation=os.path.abspath(doc),
                tags=[tags_dim, 'tensorflow', 'unet', 'denoising'],
                cite=cite,
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


def save_tf(where: Union[str, Path], model):
    where = str(where)
    path = where if where.endswith('.h5') else where + '.h5'

    # save model
    model.keras_model.save_weights(path)

    # save configuration
    save_configuration(model.config, Path(where).parent)


def format_path_for_saving(where: Union[str, Path]):
    """
    We want to create a folder containing the weights and the config file, users must point to a name (file or folder),
    and this function will create a folder with corresponding name in which to save the files.
    """
    where = Path(where)

    if where.suffix == '.h5' or str(where).endswith('.bioimage.io.zip'):
        # file, we want to create a directory with same name but without the suffix(es)
        if where.suffix == '.h5':
            new_parent = Path(where.parent, where.stem)
            new_parent.mkdir(parents=True, exist_ok=True)
        else:
            name = where.name[:-len('.bioimage.io.zip')]  # remove .bioimage.io.zip
            new_parent = Path(where.parent, name)
            new_parent.mkdir(parents=True, exist_ok=True)

        where = Path(new_parent, where.name)
    else:
        # consider it is a folder, create a new parent folder with same name
        where.mkdir(parents=True, exist_ok=True)
        where = Path(where, where.name)

    return where


def save_model(where: Union[str, Path], export_type, model, **kwargs):
    # create target directory
    where = format_path_for_saving(where)

    # save model
    if ModelSaveMode.MODELZOO.value == export_type:
        save_modelzoo(where.absolute(), model, **kwargs)
    else:
        save_tf(where.absolute(), model)
