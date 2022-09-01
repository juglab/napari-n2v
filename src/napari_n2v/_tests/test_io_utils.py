from pathlib import Path
import pytest

import numpy as np

from .test_utils import (
    create_simple_model,
    save_weights_h5,
    create_model_zoo_parameters
)
from napari_n2v.utils import (
    load_weights,
    build_modelzoo,
    create_config,
    save_configuration,
    load_configuration
)


###################################################################
# test load_weights
def test_load_weights_wrong_path(tmp_path):
    create_simple_model(tmp_path, (1, 16, 16, 1))

    # create a new model and load from previous weights
    model2 = create_simple_model(tmp_path, (1, 16, 16, 1))

    with pytest.raises(FileNotFoundError):
        load_weights(model2, 'definitely not a path')


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (64, 16, 32, 1), (1, 8, 16, 16, 1), (8, 32, 16, 64, 1)])
def test_load_weights_h5(tmp_path, shape):
    model = create_simple_model(tmp_path, shape)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model and load previous weights
    model2 = create_simple_model(tmp_path, shape)
    load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    model = create_simple_model(tmp_path, shape1)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model with different shape
    model2 = create_simple_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (8, 16, 32, 1), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_simple_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 32, 1), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_simple_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    model = create_simple_model(tmp_path, shape1)
    path_to_h5 = str(save_weights_h5(model, tmp_path).absolute())

    # second model
    model2 = create_simple_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model2, path_to_h5)


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 16, 16, 1), (16, 16, 16))])
def test_save_configuration(tmp_path, shape, patch_shape):
    X_patches = np.concatenate([np.zeros(shape), np.ones(shape)], axis=0)
    config = create_config(X_patches)

    # sanity check
    assert config.is_valid()

    # save config
    save_configuration(config, tmp_path)

    # check if exists
    assert Path(tmp_path / 'config.json').exists()


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 16, 16, 1), (16, 16, 16))])
def test_load_configuration(tmp_path, shape, patch_shape):
    X_patches = np.concatenate([np.zeros(shape), np.ones(shape)], axis=0)
    config = create_config(X_patches)

    # sanity check
    assert config.is_valid()

    # save config
    save_configuration(config, tmp_path)

    # load config
    config_loaded = load_configuration(tmp_path / 'config.json')
    assert config_loaded.is_valid()

    # todo: this doesn't work because tuple != list, but some config entries accept both...
    # assert vars(config_loaded) == vars(config)
