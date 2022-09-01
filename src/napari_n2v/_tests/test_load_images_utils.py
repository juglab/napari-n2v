import pytest

import numpy as np

from napari_n2v._tests.test_utils import (
    save_img
)
from napari_n2v.utils import (
    load_and_reshape,
    load_from_disk,
    lazy_load_generator
)


@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('shape, axes', [((16, 16), 'XY'),
                                         ((3, 16, 16), 'SXY'),
                                         ((16, 16, 16), 'ZXY'),
                                         ((16, 5, 16, 16), 'XSYZ')])
def test_load_data_from_disk_np(tmp_path, n, shape, axes):
    # save images to the disk
    save_img(tmp_path, n, shape)

    # load data
    _x, new_axes = load_and_reshape(tmp_path, axes)

    # check results
    m = 1
    if 'S' in axes:
        m = shape[axes.find('S')]

    assert _x.shape[0] == n * m
    if 'Z' in axes:
        assert new_axes == 'SZYXC'
    else:
        assert new_axes == 'SYXC'


@pytest.mark.parametrize('shape1, shape2, axes', [((16, 16), (32, 32), 'XY'),
                                                  ((3, 16, 16), (5, 16, 16), 'SXY'),
                                                  ((16, 16, 16), (8, 16, 16), 'ZXY'),
                                                  ((16, 5, 16, 16), (32, 5, 32, 16), 'XSYZ')])
def test_load_data_from_disk_list(tmp_path, shape1, shape2, axes):
    # save images to the disk
    save_img(tmp_path, 1, shape1, prefix='s1')
    save_img(tmp_path, 1, shape2, prefix='s2')

    # load data
    _x, new_axes = load_and_reshape(tmp_path, axes)

    # check results
    assert type(_x) == tuple
    assert len(_x[0]) == 2

    for im in _x[0]:
        if 'Z' in axes:
            assert len(im.shape) == 5
        else:
            assert len(im.shape) == 4

    if 'Z' in axes:
        assert new_axes == 'SZYXC'
    else:
        assert new_axes == 'SYXC'


###################################################################
# test load_from_disk
@pytest.mark.parametrize('shape, axes', [((8, 8), 'YX'),
                                         ((4, 8, 8), 'ZYX'),
                                         ((5, 8, 8), 'SYX'),
                                         ((5, 8, 8, 3), 'SYXT')])
def test_load_from_disk_same_shapes(tmp_path, shape, axes):
    n = 10
    save_img(tmp_path, n, shape)

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == np.ndarray

    if 'S' in axes:
        assert new_axes == axes
        assert len(images.shape) == len(shape)
        assert images.shape[0] == n * shape[0]
        assert images.shape[1:] == shape[1:]
    else:
        assert new_axes == 'S' + axes
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
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == tuple
    assert len(images[0]) == n[0] + n[1]
    assert len(images[1]) == n[0] + n[1]
    assert new_axes == axes

    for img in images[0]:
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
