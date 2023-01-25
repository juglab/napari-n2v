from pathlib import Path
from enum import Enum

import numpy as np

from n2v.models import N2V, N2VConfig
from typing import Union
from .n2v_utils import (
    ModelSaveMode,
    get_default_path,
    cwd,
    which_algorithm,
    Algorithm,
    get_algorithm_details
)

CONFIG = 'config.json'


class Extensions(Enum):
    BIOIMAGE_EXT = '.bioimage.io.zip'
    KERAS_EXT = '.h5'
    TF_EXT = '.zip'


class Format(Enum):
    H5 = 'h5'
    TF = 'tf'


def save_configuration(path: Union[str, Path], model: N2V):
    from csbdeep.utils import save_json

    # sanity check
    if str(path).endswith(CONFIG):
        final_path = path
    else:
        assert Path(path).is_dir()

        # save
        final_path = Path(path) / CONFIG

    save_json(vars(model.config), final_path)


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

    if str(weight_path).endswith(Extensions.BIOIMAGE_EXT.value):
        model = load_model_bioimage(weight_path)
    elif str(weight_path).endswith(Extensions.TF_EXT.value):  # we expect a TF bundle (see save_tf)
        model = load_model_tf(weight_path)
    elif str(weight_path).endswith(Extensions.KERAS_EXT.value):  # we expect .h5
        model = load_model_keras(weight_path)
    else:
        raise ValueError(f'Expected file with extension '
                         f'{Extensions.BIOIMAGE_EXT.value},'
                         f'{Extensions.TF_EXT.value} or '
                         f'{Extensions.KERAS_EXT.value}')

    return model


def save_model(model_path: Union[str, Path], export_type, model, **kwargs):
    # create target directory
    model_path = format_path_for_saving(model_path)

    # save model
    if export_type == ModelSaveMode.MODELZOO.value:
        save_model_bioimage(model_path.absolute(), model, **kwargs)
    elif export_type == ModelSaveMode.KERAS.value:
        save_model_keras(model_path.absolute(), model)
    else:
        save_model_tf(model_path.absolute(), model)


def load_model_keras(weights_path: Union[str, Path]) -> N2V:
    if not Path(weights_path).suffix == Extensions.KERAS_EXT.value:
        raise ValueError(f'Invalid weights type, expected {Extensions.KERAS_EXT.value}.')

    # check if config is present
    if not (Path(weights_path).parent / CONFIG).exists():
        raise ValueError('No config.json file found.')

    # load configuration
    config_path = Path(weights_path).parent / CONFIG
    config = load_configuration(config_path)

    # instantiate model
    model = N2V(config, 'DenoiSeg', 'models')

    # we assume we have a path to a .h5
    model.keras_model.load_weights(weights_path)

    return model


def save_model_keras(model_path: Union[str, Path], model):
    model_path = str(model_path)
    path = model_path if model_path.endswith(Extensions.KERAS_EXT.value) else model_path + Extensions.KERAS_EXT.value

    # save model
    model.keras_model.save_weights(path)

    # save configuration
    save_configuration(Path(model_path).parent, model)

    return path


def load_model_bioimage(weights_path: Union[str, Path]):
    import bioimageio.core

    if not str(weights_path).endswith(Extensions.BIOIMAGE_EXT.value):
        raise ValueError(f'Invalid weights type, expected {Extensions.BIOIMAGE_EXT.value}')

    rdf = bioimageio.core.load_resource_description(weights_path)

    # search for config file
    config_name = None
    for p in rdf.attachments.files:
        if p.name == CONFIG:
            config_name = p
            break

    if config_name:
        config = load_configuration(config_name)

        # instantiate model
        model = N2V(config, 'DenoiSeg', 'models')
    else:
        raise ValueError('Failed to find config.json in the archive.')

    # search for the h5 file
    # TODO as an alternative, load .pb?
    weights_name = None
    for p in rdf.attachments.files:
        if p.suffix == Extensions.KERAS_EXT.value:
            weights_name = p
            break

    if weights_name:
        model.keras_model.load_weights(weights_name)
    else:
        raise ValueError(f'Failed to find {Extensions.KERAS_EXT.value} in the archive.')

    return model


def save_model_bioimage(destination: Path,
                        model: N2V,
                        axes: str,
                        input_path: str,
                        output_path: str,
                        tf_version: str):
    path = get_default_path() / 'bioimage.io'
    with cwd(path):
        # save .h5 weights
        path_weights_h5 = Path('weights.h5')
        save_model_keras(path_weights_h5, model)

        # save configuration
        path_config = Path(CONFIG)
        save_configuration(path_config, model)

        # save TF model bundle
        path_bundle = save_model_tf(Path('tf_model'), model)

        # format axes for bioimage.io
        new_axes = axes.replace('S', 'b').lower()

        if 'b' not in new_axes:
            new_axes = 'b' + new_axes

        # processing
        preprocessing = [{
            'name': 'zero_mean_unit_variance',
            'kwargs': {
                'mode': 'fixed',
                'axes': 'yx' if len(axes) == 4 else 'zyx',
                'mean': [float(m) for m in model.config.means],
                'std': [float(s) for s in model.config.stds]
            }
        }]
        postprocessing = [{
            'name': 'scale_linear',
            'kwargs': {
                'axes': 'yx' if len(axes) == 4 else 'zyx',
                'gain': [float(s) for s in model.config.stds],
                'offset': [float(m) for m in model.config.means]
            }
        }]

        # check algorithm (N2V, structN2V, N2V2) and get the corresponding details
        algorithm = which_algorithm(model.config)
        name, authors, cite = get_algorithm_details(algorithm)

        # create documentation.md
        doc = generate_bioimage_md(name, cite)

        # files
        files = [path_config.absolute(), path_weights_h5.absolute()]

        # check path ending
        destination = str(destination)
        if destination.endswith(Extensions.BIOIMAGE_EXT.value):
            path = destination
        else:
            path = destination + Extensions.BIOIMAGE_EXT.value

        # save model
        build_modelzoo(path,
                       path_bundle,
                       input_path,
                       output_path,
                       preprocessing,
                       postprocessing,
                       doc,
                       name,
                       authors,
                       algorithm,
                       cite,
                       tf_version,
                       new_axes,
                       files)

        return path


def load_model_tf(weights_path: Union[str, Path]):
    import tensorflow as tf
    from zipfile import ZipFile
    from shutil import rmtree

    if not Path(weights_path).suffix == Extensions.TF_EXT.value:
        raise ValueError(f'Invalid weights type, expected {Extensions.TF_EXT.value}.')

    with cwd(get_default_path()):
        path_bundle = Path('tf_model').absolute()
        if path_bundle.exists():
            rmtree(path_bundle)

        # extract zip
        with ZipFile(weights_path, 'r') as zip_file:
            zip_file.extractall(path=path_bundle)

        # configuration file path
        config_expected_path = Path(path_bundle, CONFIG)
        if not config_expected_path.exists():
            raise FileNotFoundError(f'Could not found configuration in {path_bundle}.')

        # create configuration
        config = load_configuration(config_expected_path)

        # instantiate model
        model = N2V(config, 'DenoiSeg', 'models')

        # load weights
        model.keras_model = tf.keras.models.load_model(path_bundle, compile=False)
        # TODO test if it can then be retrained? because it is not compiled
        return model


def save_model_tf(destination: Path, model: N2V):
    """
    Save the model as a TF bundle and returns the path to the archive.
    """
    import tensorflow as tf
    from zipfile import ZipFile
    from shutil import rmtree

    with cwd(get_default_path()):
        path_bundle = Path('tf_model').absolute()
        if path_bundle.exists():
            rmtree(path_bundle)

        # save bundle without including optimizer
        # (otherwise the absence of the custom functions cause errors upon loading)
        tf.keras.models.save_model(
            model.keras_model,
            path_bundle,
            save_format=Format.TF.value,
            include_optimizer=False
        )

        # save configuration
        save_configuration(path_bundle, model)

        # zip it and save to destination
        final_archive = Path(destination.parent, destination.stem + '.zip').absolute()
        with ZipFile(final_archive, mode="w") as archive:
            for file_path in path_bundle.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(path_bundle))

        return final_archive


def build_modelzoo(path: Union[str, Path],
                   weights: Union[str, Path],
                   inputs: str,
                   outputs: str,
                   preprocessing: list,
                   postprocessing: list,
                   doc: Union[str, Path],
                   name: str,
                   authors: list,
                   algorithm: Algorithm,
                   cite: list,
                   tf_version: str,
                   axes: str = 'byxc',
                   files: list = [],
                   **kwargs):
    from bioimageio.core.build_spec import build_model

    assert str(path).endswith(Extensions.BIOIMAGE_EXT.value), f'Path must end with {Extensions.BIOIMAGE_EXT.value}'

    tags_dim = '3d' if len(axes) == 5 else '2d'

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
                documentation=doc,
                tags=[tags_dim, 'unet', 'denoising', algorithm.value, 'tensorflow', 'napari'],
                cite=cite,
                preprocessing=[preprocessing],
                postprocessing=[postprocessing],
                tensorflow_version=tf_version,
                attachments={"files": files},
                **kwargs
                )


def generate_bioimage_md(name: str, cite: list):
    """
    Generate a generic document.md file for the bioimage.io format.
    """
    # create doc
    file = Path('napari-n2v.md')
    with open(file, 'w') as f:
        text = cite[0]['text']

        content = f'## {name}\n' \
                  f'This network was trained using [napari-n2v](https://pypi.org/project/napari-n2v/).\n\n' \
                  f'## Cite {name}\n' \
                  f'{text}'
        f.write(content)

    return file.absolute()


def format_path_for_saving(where: Union[str, Path]):
    """
    We want to create a folder containing the weights and the config file, users must point to a name (file or folder),
    and this function will create a folder with corresponding name in which to save the files.
    """
    where = Path(where)

    if where.suffix == Extensions.KERAS_EXT.value or str(where).endswith(Extensions.BIOIMAGE_EXT.value):
        # file, we want to create a directory with same name but without the suffix(es)
        if where.suffix == Extensions.KERAS_EXT.value:
            new_parent = Path(where.parent, where.stem)
            new_parent.mkdir(parents=True, exist_ok=True)
        else:
            name = where.name[:-len(Extensions.BIOIMAGE_EXT.value)]  # remove .bioimage.io.zip
            new_parent = Path(where.parent, name)
            new_parent.mkdir(parents=True, exist_ok=True)

        where = Path(new_parent, where.name)
    else:
        # consider it is a folder, create a new parent folder with same name
        where.mkdir(parents=True, exist_ok=True)
        where = Path(where, where.name)

    return where
