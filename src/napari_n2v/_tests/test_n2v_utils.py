from pathlib import Path

import numpy as np
import pytest
from marshmallow import ValidationError

from napari_n2v.utils import (
    filter_dimensions,
    are_axes_valid,
    build_modelzoo,
    load_from_disk,
    reshape_data,
    lazy_load_generator,
    load_weights,
    reshape_napari,
    load_configuration,
    save_configuration,
    create_config
)
from napari_n2v._tests.test_utils import (
    save_img,
    create_model,
    save_weights_h5,
    create_model_zoo_parameters
)


@pytest.mark.parametrize('shape', [3, 4, 5])
@pytest.mark.parametrize('is_3D', [True, False])
def test_filter_dimensions(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)

    if is_3D:
        assert all(['Z' in p for p in permutations])

    assert all(['YX' == p[-2:] for p in permutations])


def test_filter_dimensions_len6_Z():
    permutations = filter_dimensions(6, True)

    assert all(['Z' in p for p in permutations])
    assert all(['YX' == p[-2:] for p in permutations])


@pytest.mark.parametrize('shape, is_3D', [(2, True), (6, False), (7, True)])
def test_filter_dimensions_error(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)
    print(permutations)
    assert len(permutations) == 0


@pytest.mark.parametrize('axes, valid', [('XSYCZ', False),
                                         ('YZX', True),
                                         ('TCS', False),
                                         ('xsYcZ', False),
                                         ('YzX', True),
                                         ('tCS', False),
                                         ('SCZXYT', False),
                                         ('SZXCZY', False),
                                         ('Xx', False),
                                         ('SZXGY', False),
                                         ('I5SYX', False),
                                         ('STZCYXL', False)])
def test_are_axes_valid(axes, valid):
    assert are_axes_valid(axes) == valid


###################################################################
# test build_modelzoo
@pytest.mark.parametrize('shape', [(1, 16, 16, 1),
                                   (1, 16, 8, 1),
                                   (1, 16, 8, 1),
                                   (1, 16, 16, 8, 1),
                                   (1, 16, 16, 8, 1),
                                   (1, 8, 16, 32, 1)])
def test_build_modelzoo_allowed_shapes(tmp_path, shape):
    # create model and save it to disk
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # check if modelzoo exists
    assert Path(parameters[0]).exists()


@pytest.mark.parametrize('shape', [(8,), (8, 16), (1, 16, 16), (32, 16, 8, 16, 32, 1)])
def test_build_modelzoo_disallowed_shapes(tmp_path, shape):
    """
    Test ModelZoo creation based on disallowed shapes.

    :param tmp_path:
    :param shape:
    :return:
    """
    # create model and save it to disk
    with pytest.raises(AssertionError):
        parameters = create_model_zoo_parameters(tmp_path, shape)
        build_modelzoo(*parameters)


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
    with pytest.raises(ValidationError):
        parameters = create_model_zoo_parameters(tmp_path, shape)
        build_modelzoo(*parameters)


###################################################################
# test load_from_disk
@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'), ((4, 8, 8), 'ZYX'), ((5, 8, 8), 'SYX')])
def test_load_from_disk_same_shapes(tmp_path, shape, axes):
    n = 10
    save_img(tmp_path, n, shape)

    # load images
    images = load_from_disk(tmp_path, axes)
    assert type(images) == np.ndarray

    if 'S' in axes:
        assert len(images.shape) == len(shape)
        assert images.shape[0] == n * shape[0]
        assert images.shape[1:] == shape[1:]
    else:
        assert len(images.shape) == len(shape) + 1
        assert images.shape[0] == n
        assert images.shape[1:] == shape

    assert (images[0, ...] != images[1, ...]).all()


@pytest.mark.parametrize('shape1, shape2, axes', [((8, 8), (4, 4), 'YX'),
                                                  ((4, 8, 8), (2, 16, 16), 'ZYX'),
                                                  ((4, 8, 8), (2, 16, 16), 'SYX'),
                                                  ((8, 16), (4, 8, 8), 'YX')])
def test_load_from_disk_different_shapes(tmp_path, shape1, shape2, axes):
    n = [10, 5]
    save_img(tmp_path, n[0], shape1, prefix='im1-')
    save_img(tmp_path, n[1], shape2, prefix='im2-')

    # load images
    images = load_from_disk(tmp_path, axes)
    assert type(images) == list
    assert len(images) == n[0] + n[1]

    for img in images:
        assert (img.shape == shape1) or (img.shape == shape2)


def test_lazy_generator(tmp_path):
    n = 10
    save_img(tmp_path, n, (8, 8, 8))

    # create lazy generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == n

    # check that it can load n images
    for i in range(n):
        ret = next(gen, None)
        assert len(ret) == 3
        assert all([r is not None for r in ret])

    # test that next(gen, None) works
    assert next(gen, None) is None

    # test that next() throws error
    with pytest.raises(StopIteration):
        next(gen)


###################################################################
# test load_weights
def test_load_weights_wrong_path(tmp_path):
    model = create_model(tmp_path, (1, 16, 16, 1))

    # create a new model and load from previous weights
    model2 = create_model(tmp_path, (1, 16, 16, 1))

    with pytest.raises(FileNotFoundError):
        load_weights(model2, 'definitely not a path')


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (64, 16, 32, 1), (1, 8, 16, 16, 1), (8, 32, 16, 64, 1)])
def test_load_weights_h5(tmp_path, shape):
    model = create_model(tmp_path, shape)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model and load previous weights
    model2 = create_model(tmp_path, shape)
    load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    model = create_model(tmp_path, shape1)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create a new model with different shape
    model2 = create_model(tmp_path, shape2)

    # set previous weights
    with pytest.raises(ValueError):
        load_weights(model2, str(path_to_h5))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (8, 16, 32, 1), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 32, 1), (1, 8, 16, 16, 1), (1, 8, 16, 16, 1)])
def test_load_weights_modelzoo(tmp_path, shape):
    # save model_zoo
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # create a new model and load from previous weights
    model = create_model(tmp_path, shape)
    load_weights(model, str(parameters[0]))


@pytest.mark.parametrize('shape1, shape2', [((1, 16, 16, 1), (1, 8, 16, 16, 1)),
                                            ((1, 8, 16, 16, 1), (1, 16, 16, 1))])
def test_load_weights_h5_incompatible_shapes(tmp_path, shape1, shape2):
    model = create_model(tmp_path, shape1)
    path_to_h5 = str(save_weights_h5(model, tmp_path).absolute())

    # second model
    model2 = create_model(tmp_path, shape2)

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


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (1, 16, 8, 1), 'SYXC'),
                          ((16, 8), 'XY', (1, 8, 16, 1), 'SYXC'),
                          ((16, 3, 8), 'XZY', (1, 3, 8, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'XYZ', (1, 8, 3, 16, 1), 'SZYXC'),
                          ((16, 3, 8), 'ZXY', (1, 16, 8, 3, 1), 'SZYXC'),
                          ((16, 3, 12), 'SXY', (16, 12, 3, 1), 'SYXC'),
                          ((5, 5, 2), 'XYS', (2, 5, 5, 1), 'SYXC'),
                          ((5, 1, 5, 2), 'XZYS', (2, 1, 5, 5, 1), 'SZYXC'),
                          ((5, 12, 5, 2), 'ZXYS', (2, 5, 5, 12, 1), 'SZYXC'),
                          ((16, 8, 5, 12), 'SZYX', (16, 8, 5, 12, 1), 'SZYXC'),
                          ((16, 8, 5), 'YXT', (5, 16, 8, 1), 'SYXC'),  # T, no C
                          ((4, 16, 8), 'TXY', (4, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 8), 'TXSY', (4 * 6, 8, 16, 1), 'SYXC'),
                          ((4, 16, 6, 5, 8), 'ZXTYS', (8 * 6, 4, 5, 16, 1), 'SZYXC'),
                          ((5, 3, 5), 'XCY', (1, 5, 5, 3), 'SYXC'),  # C, no T
                          ((16, 3, 12, 8), 'XCYS', (8, 12, 16, 3), 'SYXC'),
                          ((16, 3, 12, 8), 'ZXCY', (1, 16, 8, 3, 12), 'SZYXC'),
                          ((16, 3, 12, 8), 'XCYZ', (1, 8, 12, 16, 3), 'SZYXC'),
                          ((16, 3, 12, 8), 'ZYXC', (1, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 12, 8), 'ZYSXC', (21, 16, 3, 12, 8), 'SZYXC'),
                          ((16, 3, 21, 8, 12), 'SZYCX', (16, 3, 21, 12, 8), 'SZYXC'),
                          ((5, 3, 8, 6), 'XTCY', (3, 6, 5, 8), 'SYXC'),  # CT
                          ((16, 3, 12, 5, 8), 'XCYTS', (8 * 5, 12, 16, 3), 'SYXC'),
                          ((16, 10, 5, 6, 12, 8), 'ZSXCYT', (10 * 8, 16, 12, 5, 6), 'SZYXC')
                          ])
def test_reshape_data_single(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    _x, new_axes = reshape_data(x, axes)

    assert _x.shape == final_shape
    assert new_axes == final_axes


def test_reshape_single_data_values_XY():
    # test if X and Y are flipped
    axes = 'XY'
    shape = (16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for i in range(shape[1]):
        assert (_x[i, :] == x[:, i]).all()


def test_reshape_single_data_values_XZY():
    axes = 'XZY'
    shape = (16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for z in range(shape[1]):
        for i in range(shape[2]):
            assert (_x[z, i, :] == x[:, z, i]).all()


def test_reshape_single_data_values_XZTY():
    axes = 'XZTY'
    shape = (16, 15, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for t in range(shape[2]):
        for z in range(shape[1]):
            for i in range(shape[3]):
                assert (_x[t, z, i, :] == x[:, z, t, i]).all()


def test_reshape_single_data_values_STYX():
    axes = 'STYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for s in range(shape[0]):
        for t in range(shape[1]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[0] + s, i, :] == x[s, t, i, :]).all()


def test_reshape_single_data_values_TSYX():
    axes = 'TSYX'
    shape = (5, 10, 16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[0]):
            for i in range(shape[2]):
                # here reshaping happens because S and T dims are pulled together
                assert (_x[t * shape[1] + s, i, :] == x[t, s, i, :]).all()


def test_reshape_single_data_values_SZYTX():
    axes = 'ZSYTX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for s in range(shape[1]):
        for t in range(shape[3]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[1] + s, z, i, :] == x[z, s, i, t, :]).all()


def test_reshape_single_data_values_ZTYSX():
    axes = 'ZTYSX'
    shape = (15, 10, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)
    _x = _x.squeeze()

    for s in range(shape[3]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for i in range(shape[2]):
                    # here reshaping happens because S and T dims are pulled together
                    assert (_x[t * shape[3] + s, z, i, :] == x[z, t, i, s, :]).all()


def test_reshape_single_data_values_SZCYTX():
    axes = 'ZTCYSX'
    shape = (15, 10, 3, 16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_data(x, axes)

    for s in range(shape[4]):
        for t in range(shape[1]):
            for z in range(shape[0]):
                for c in range(shape[2]):
                    for i in range(shape[3]):
                        # here reshaping happens because S and T dims are pulled together
                        assert (_x[t * shape[4] + s, z, i, :, c] == x[z, t, c, i, s, :]).all()


##########################################
# reshape napari
def test_reshape_data_napari_values_XY():
    # test if X and Y are flipped
    axes = 'XY'
    shape = (16, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for i in range(shape[1]):
        assert (_x[i, :] == x[:, i]).all()


def test_reshape_data_napari_values_XZY():
    # test if X and Y are flipped
    axes = 'XZY'
    shape = (16, 5, 8)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for z in range(shape[1]):
        for i in range(shape[2]):
            assert (_x[z, i, :] == x[:, z, i]).all()


def test_reshape_data_napari_values_SYXC():
    # test if X and Y are flipped
    axes = 'SYXC'
    shape = (10, 16, 8, 3)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[-1]):
        for s in range(shape[0]):
            for i in range(shape[1]):
                assert (_x[c, s, i, :] == x[s, i, :, c]).all()


def test_reshape_data_napari_values_SZYXC():
    # test if X and Y are flipped
    axes = 'SZYXC'
    shape = (10, 15, 16, 8, 3)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[-1]):
        for s in range(shape[0]):
            for z in range(shape[1]):
                for i in range(shape[2]):
                    assert (_x[c, s, z, i, :] == x[s, z, i, :, c]).all()


def test_reshape_data_napari_values_SZYXC():
    # test if X and Y are flipped
    axes = 'YSXTZC'
    shape = (16, 10, 8, 15, 3, 2)
    x = np.random.randint(0, 255, shape)

    _x, _ = reshape_napari(x, axes)

    for c in range(shape[5]):
        for t in range(shape[3]):
            for s in range(shape[1]):
                for z in range(shape[4]):
                    for i in range(shape[0]):
                        assert (_x[c, t, s, z, i, :] == x[i, s, :, t, z, c]).all()


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (16, 8), 'YX'),
                          ((16, 8), 'XY', (8, 16), 'YX'),
                          ((16, 8, 5), 'XYZ', (5, 8, 16), 'ZYX'),
                          ((5, 16, 8), 'ZXY', (5, 8, 16), 'ZYX'),
                          ((12, 16, 8, 10), 'TXYS', (12, 10, 8, 16), 'TSYX'),
                          ((10, 5, 16, 8, 3), 'SZXYC', (3, 10, 5, 8, 16), 'CSZYX'),
                          ((16, 10, 3, 8), 'YSCX', (3, 10, 16, 8), 'CSYX')
                          ])
def test_reshape_data_napari(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    _x, new_axes = reshape_napari(x, axes)

    assert _x.shape == final_shape
    assert new_axes == final_axes
