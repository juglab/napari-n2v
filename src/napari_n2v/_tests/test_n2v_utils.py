from pathlib import Path

import numpy as np
import pytest
from marshmallow import ValidationError

from napari_n2v.utils import (
    filter_dimensions,
    are_axes_valid,
    build_modelzoo,
    reshape_data,
    reshape_napari
)
from napari_n2v._tests.test_utils import (
    create_model_zoo_parameters
)


@pytest.mark.parametrize('shape', [3, 4, 5])
@pytest.mark.parametrize('is_3D', [True, False])
def test_filter_dimensions(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)

    if is_3D:
        assert all(['Z' in p for p in permutations])

    assert all([('YX' in p) or ('XY' in p) for p in permutations])


def test_filter_dimensions_len6_Z():
    permutations = filter_dimensions(6, True)

    assert all(['Z' in p for p in permutations])
    assert all([('YX' in p) or ('XY' in p) for p in permutations])


@pytest.mark.parametrize('shape, is_3D', [(2, True), (6, False), (7, True)])
def test_filter_dimensions_error(shape, is_3D):
    permutations = filter_dimensions(shape, is_3D)
    print(permutations)
    assert len(permutations) == 0


@pytest.mark.parametrize('axes, valid', [('XSYCZ', False),
                                         ('YXZ', True),
                                         ('YXz', True),
                                         ('ZYx', True),
                                         ('YXC', True),
                                         ('CYX', True),
                                         ('TCS', False),
                                         ('xsYcZ', False),
                                         ('YzX', False),
                                         ('tCS', False),
                                         ('CSZXYT', True),
                                         ('ZSXCZY', False),
                                         ('Xx', False),
                                         ('SZXGY', False),
                                         ('I5SYX', False),
                                         ('STZCYXL', False)])
def test_are_axes_valid(axes, valid):
    assert are_axes_valid(axes) == valid


###################################################################
# test build_modelzoo
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
    # create model and save it to disk
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # check if modelzoo exists
    assert Path(parameters[0]).exists()


@pytest.mark.parametrize('shape', [(8,), (8, 16), (3, 16, 16), (32, 16, 8, 16, 32, 1)])
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


@pytest.mark.parametrize('shape, axes, final_shape, final_axes',
                         [((16, 8), 'YX', (1, 16, 8, 1), 'SYXC'),
                          ((16, 8), 'XY', (1, 8, 16, 1), 'SYXC'),
                          ((16, 3, 8), 'XZY', (1, 3, 8, 16, 1), 'SZYXC'),
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
                          ((10, 5, 16, 8, 3), 'SZXYC', (10, 5, 8, 16, 3), 'SZYXC'),
                          ((16, 10, 3, 8), 'YSCX', (10, 16, 8, 3), 'SYXC')
                          ])
def test_reshape_data_napari(shape, axes, final_shape, final_axes):
    x = np.zeros(shape)

    _x, new_axes = reshape_napari(x, axes)

    assert _x.shape == final_shape
    assert new_axes == final_axes
