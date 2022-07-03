import pytest
from napari_n2v.utils import filter_dimensions, are_axes_valid, build_modelzoo
from napari_n2v._tests.test_utils import (
    save_img,
    create_data,
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


@pytest.mark.parametrize('axes, valid', [('XSYCZ', True),
                                         ('YZX', True),
                                         ('TCS', True),
                                         ('xsYcZ', True),
                                         ('YzX', True),
                                         ('tCS', True),
                                         ('SCZXYT', True),
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
                                   (1, 16, 8, 3),
                                   (1, 16, 16, 8, 1),
                                   (1, 16, 16, 8, 3),
                                   (1, 8, 16, 32, 1)])
def test_build_modelzoo_allowed_shapes(tmp_path, shape):
    # create model and save it to disk
    parameters = create_model_zoo_parameters(tmp_path, shape)
    build_modelzoo(*parameters)

    # check if modelzoo exists
    assert Path(parameters[0]).exists()


@pytest.mark.parametrize('shape', [(8,), (8, 16), (1, 16, 16), (32, 16, 8, 16, 32, 3)])
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


@pytest.mark.parametrize('shape', [(8, 16, 16, 3),
                                   (8, 16, 16, 8, 3)])
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

