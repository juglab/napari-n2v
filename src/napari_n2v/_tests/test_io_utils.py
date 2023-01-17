from pathlib import Path
import pytest

import numpy as np

from marshmallow import ValidationError

from .test_utils import (
    create_simple_model,
    save_weights_h5,
    create_model_zoo_parameters
)
from napari_n2v.utils import (
    save_configuration,
    load_configuration,
    create_config,
    cwd
)
from napari_n2v.utils.io_utils import (
    load_model_keras,
    save_model_keras,
    load_model_tf,
    save_model_tf,
    load_model_bioimage,
    save_model_bioimage,
    format_path_for_saving,
    build_modelzoo,
    generate_bioimage_md,
    Extensions,
    CONFIG
)


###################################################################
# test keras
@pytest.mark.parametrize('shape', [(1, 16, 16, 16, 1)])
def test_save_model_keras(tmp_path, shape):
    model = create_simple_model(tmp_path, shape)

    # save model
    model_path = Path(tmp_path, 'my_model')
    save_model_keras(model_path, model)

    # check if saved
    weights_path = Path(str(model_path) + Extensions.KERAS_EXT.value)
    config_path = Path(tmp_path, CONFIG)
    assert weights_path.exists()
    assert config_path.exists()


@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 16, 1), 'SZYXC')])
def test_load_model_keras(tmp_path, shape, axes):
    model = create_simple_model(tmp_path, shape)

    # save model
    model_path = Path(tmp_path, 'my_model')
    save_model_keras(model_path, model)

    # load weights
    weight_path = Path(str(model_path) + Extensions.KERAS_EXT.value)
    new_model = load_model_keras(weight_path)

    # predict using both models
    X = np.random.randn(*shape)
    X_pred = model.predict(X, axes)
    X_new_pred = new_model.predict(X, axes)

    # compare results
    assert (X_new_pred == X_pred).all()


###################################################################
# test TF
@pytest.mark.parametrize('shape', [(1, 16, 16, 16, 1)])
def test_save_model_TF(tmp_path, shape):
    model = create_simple_model(tmp_path, shape)

    # save model
    model_path = Path(tmp_path, 'my_model')
    _ = save_model_tf(model_path, model)

    # check if saved
    weights_path = Path(str(model_path) + Extensions.TF_EXT.value)
    print(weights_path)
    assert weights_path.exists()


@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 16, 1), 'SZYXC')])
def test_load_model_TF(tmp_path, shape, axes):
    model = create_simple_model(tmp_path, shape)

    # save model
    model_path = Path(tmp_path, 'my_model')
    saved_archive = save_model_tf(model_path, model)

    # load new model
    new_model = load_model_tf(saved_archive)

    # predict using both models
    X = np.random.randn(*shape)
    X_pred = model.predict(X, axes)
    X_new_pred = new_model.predict(X, axes)

    # compare results
    assert (X_new_pred == X_pred).all()


###################################################################
# test bioimage.io
@pytest.mark.bioimage_io
@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 16, 1), 'bzyxc')])
def test_save_model_bioimage(tmp_path, shape, axes):
    model = create_simple_model(tmp_path, shape)

    # create inputs and outputs
    X = np.random.randn(*shape)
    Y = np.random.randn(*shape)
    input_path = str(Path(tmp_path, 'inputs.npy'))
    output_path = str(Path(tmp_path, 'outputs.npy'))
    np.save(input_path, X)
    np.save(output_path, Y)

    # tf version
    tf_version = '42'

    # save model
    model_path = Path(tmp_path, 'my_model')
    _ = save_model_bioimage(destination=model_path,
                            model=model,
                            axes=axes,
                            input_path=input_path,
                            output_path=output_path,
                            tf_version=tf_version)

    # check if saved
    weights_path = Path(str(model_path) + Extensions.BIOIMAGE_EXT.value)
    print(weights_path)
    assert weights_path.exists()


@pytest.mark.bioimage_io
@pytest.mark.parametrize('shape, axes', [((1, 16, 16, 16, 1), 'SZYXC')])
def test_load_model_bioimage(tmp_path, shape, axes):
    model = create_simple_model(tmp_path, shape)

    # create inputs and outputs
    X = np.random.randn(*shape)
    Y = model.predict(X, axes)
    input_path = str(Path(tmp_path, 'inputs.npy'))
    output_path = str(Path(tmp_path, 'outputs.npy'))
    np.save(input_path, X)
    np.save(output_path, Y)

    # tf version
    tf_version = '42'

    # save model
    model_path = Path(tmp_path, 'my_model')
    path_to_model = save_model_bioimage(destination=model_path,
                                        model=model,
                                        axes='bzyxc',
                                        input_path=input_path,
                                        output_path=output_path,
                                        tf_version=tf_version)

    # load new model
    new_model = load_model_bioimage(path_to_model)

    # predict using both models
    X_pred = Y
    X_new_pred = new_model.predict(X, axes)

    # compare results
    assert (X_new_pred == X_pred).all()


@pytest.mark.parametrize('shape', [(1, 16, 16, 16, 1)])
def test_save_configuration(tmp_path, shape):
    model = create_simple_model(tmp_path, shape)

    # sanity check
    assert model.config.is_valid()

    # save config
    save_configuration(tmp_path, model)

    # check if exists
    assert Path(tmp_path / CONFIG).exists()


@pytest.mark.parametrize('shape, patch_shape', [((1, 16, 16, 1), (16, 16)),
                                                ((1, 16, 16, 16, 1), (16, 16, 16))])
def test_load_configuration(tmp_path, shape, patch_shape):
    model = create_simple_model(tmp_path, shape)

    # save config
    save_configuration(tmp_path, model)

    # load config
    config_loaded = load_configuration(tmp_path / CONFIG)
    assert config_loaded.is_valid()

    # todo: this doesn't work because tuple != list, but some config entries accept both...
    # assert vars(config_loaded) == vars(config)


@pytest.mark.parametrize('path', ['mydir',
                                  Path('mydir', 'my2dir'),
                                  'mymodel.h5',
                                  'mymodel.bioimage.io.zip',
                                  Path('mydir', 'my2model.h5'),
                                  Path('mydir', 'my2dir.bioimage.io.zip'),
                                  Path('mydir', 'my2dir', 'my2model.h5'),
                                  Path('mydir', 'my2dir', 'my3dir'),
                                  'mydir.tif',
                                  'mydir.'])
def test_format_path_for_saving(tmp_path, path):
    where = format_path_for_saving(Path(tmp_path, path))

    # if pointing to file
    if where.suffix == '.h5' or str(where).endswith('.bioimage.io.zip'):
        # parent is created, but not file
        assert where.parent.exists()
        assert not where.exists()

        if where.suffix == '.h5':
            assert where.parent.name == where.stem
        else:
            name = where.name[:-len('.bioimage.io.zip')]
            assert where.parent.name == name
    else:
        assert not where.exists()
        assert where.parent.exists()
        assert where.parent.name == where.name


###################################################################
# test build_modelzoo
@pytest.mark.bioimage_io
@pytest.mark.parametrize('shape', [(1, 16, 16, 1),
                                   (1, 16, 16, 3),
                                   (1, 16, 8, 1),
                                   (1, 16, 8, 3),
                                   (1, 16, 16, 8, 1),
                                   (1, 16, 16, 8, 1),
                                   (1, 16, 16, 8, 3),
                                   (1, 16, 16, 8, 3),
                                   (1, 8, 16, 32, 1)])
def test_build_modelzoo_allowed_shapes(tmp_path, shape):
    # make sure files are created in tmp_path:
    with cwd(tmp_path):
        # create model and save it to disk
        parameters = create_model_zoo_parameters(tmp_path, shape)
        build_modelzoo(*parameters)

        # check if modelzoo exists
        assert Path(parameters[0]).exists()


@pytest.mark.bioimage_io
@pytest.mark.parametrize('shape', [(8, 16, 16, 1),
                                   (8, 16, 16, 8, 1)])
def test_build_modelzoo_disallowed_batch(tmp_path, shape):
    """
    Test ModelZoo creation based on disallowed shapes.

    :param tmp_path:
    :param shape:
    :return:
    """
    # create model and save it to disk
    with cwd(tmp_path):
        parameters = create_model_zoo_parameters(tmp_path, shape)
        with pytest.raises(ValidationError):
            build_modelzoo(*parameters)


def test_generate_bioimage_md(tmp_path):
    """
    Test that the generated md file exists and that the content is as expected.
    """
    name = 'Arthur Dent'
    text = 'In the beginning the Universe was created. This has made a lot of people very angry and been widely ' \
           'regarded as a bad move.'
    content = f'## {name}\n' \
              f'This network was trained using [napari-n2v](https://pypi.org/project/napari-n2v/).\n\n' \
              f'## Cite {name}\n' \
              f'{text}'

    with cwd(tmp_path):
        file = generate_bioimage_md(name, [{'text': text}])

        assert file.exists()

        # read file
        with open(file, 'r') as f:
            read = f.read()
            assert read == content
