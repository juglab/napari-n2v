import numpy as np
import pytest

from qtpy.QtWidgets import QWidget

from napari_n2v.widgets import TrainingSettingsWidget
from napari_n2v.utils import (
    filter_dimensions,
    are_axes_valid,
    reshape_data,
    reshape_napari,
    create_model,
    get_default_settings,
    get_pms,
    get_losses,
    create_config,
    which_algorithm,
    Algorithm,
    PixelManipulator
)


def test_which_algorithm():
    # fake data
    shape = (1, 8, 8, 1)
    X = np.concatenate([np.ones(shape), np.zeros(shape)], axis=0)
    name = 'myModel'

    # get default settings for N2V
    expert_settings = get_default_settings(False)
    config = create_config(X, **expert_settings)
    assert which_algorithm(config) == Algorithm.N2V

    # structN2V
    expert_settings['structN2Vmask'] = [0, 1, 1, 1, 0]
    config = create_config(X, **expert_settings)
    assert which_algorithm(config) == Algorithm.StructN2V

    # N2V2
    expert_settings['structN2Vmask'] = None
    expert_settings['blurpool'] = True
    expert_settings['skip_skipone'] = True
    expert_settings['n2v_manipulator'] = PixelManipulator.MEDIAN.value
    config = create_config(X, **expert_settings)
    assert which_algorithm(config) == Algorithm.N2V2


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
                assert (_x[s, i, :, c] == x[s, i, :, c]).all()


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
                    assert (_x[s, z, i, :, c] == x[s, z, i, :, c]).all()


def test_reshape_data_napari_values_STZYXC():
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
                        assert (_x[t, s, z, i, :, c] == x[i, s, :, t, z, c]).all()


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


#############################################
# create model
@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 16, 16, 1)])
def test_create_model_default_settings(shape):
    """
    Test that the default settings are the same that in the default configuration
    created in the napari plugin.
    """
    x = np.concatenate([np.ones(shape), np.zeros(shape)])
    model = create_model(x)

    # test config
    assert model.config.is_valid()

    # assert that all default settings are correctly set
    is_3D = len(x.shape) == 5
    default_settings = get_default_settings(is_3D)

    assert model.config.unet_n_depth == default_settings['unet_n_depth']
    assert model.config.unet_kern_size == default_settings['unet_kern_size']
    assert model.config.unet_n_first == default_settings['unet_n_first']
    assert model.config.train_learning_rate == default_settings['train_learning_rate']
    assert model.config.n2v_perc_pix == default_settings['n2v_perc_pix']
    assert model.config.n2v_neighborhood_radius == default_settings['n2v_neighborhood_radius']
    assert model.config.n2v_manipulator == default_settings['n2v_manipulator']
    assert model.config.train_loss == default_settings['train_loss']
    assert model.config.unet_residual == default_settings['unet_residual']
    assert model.config.single_net_per_channel == default_settings['single_net_per_channel']
    assert model.config.structN2Vmask == default_settings['structN2Vmask']


@pytest.mark.qt
@pytest.mark.parametrize('shape', [(1, 16, 16, 1), (1, 16, 16, 16, 1)])
def test_create_model_expert_settings(qtbot, shape):
    """
    Tests that expert settings are correctly assigned to the configuration.
    """
    x = np.concatenate([np.ones(shape), np.zeros(shape)])
    is_3D = len(x.shape) == 5

    # create expert settings
    widget = QWidget()
    widget_settings = TrainingSettingsWidget(widget, is_3D)

    # modify the settings
    unet_n_first = 64
    unet_depth = 3
    unet_kernelsize = 3
    train_learning_rate = 0.0002
    n2v_perc_pix = 0.1
    n2v_neighborhood_radius = 7
    n2v_pm = get_pms()[3]
    loss = get_losses()[1]
    unet_residuals = True
    single_net = False

    widget_settings.unet_n_first.setValue(unet_n_first)
    widget_settings.unet_depth.setValue(unet_depth)
    widget_settings.unet_kernelsize.setValue(unet_kernelsize)
    widget_settings.train_learning_rate.setValue(train_learning_rate)
    widget_settings.n2v_perc_pix.setValue(n2v_perc_pix)
    widget_settings.n2v_neighborhood_radius.setValue(n2v_neighborhood_radius)
    widget_settings.n2v_pm = n2v_pm
    widget_settings.loss = loss
    widget_settings.unet_residuals.setChecked(unet_residuals)
    widget_settings.single_net.setChecked(single_net)
    widget_settings.structN2V_mask.setText('0, 1, 1, 1, 0')

    # check settings
    settings = widget_settings.get_settings(is_3D=is_3D)
    assert settings['unet_n_depth'] == unet_depth
    assert settings['unet_kern_size'] == unet_kernelsize
    assert settings['unet_n_first'] == unet_n_first
    assert settings['train_learning_rate'] == train_learning_rate
    assert settings['n2v_perc_pix'] == n2v_perc_pix
    assert settings['n2v_neighborhood_radius'] == n2v_neighborhood_radius
    assert settings['n2v_manipulator'] == n2v_pm
    assert settings['train_loss'] == loss
    assert settings['unet_residual'] == unet_residuals
    assert settings['single_net_per_channel'] == single_net
    assert settings['structN2Vmask'] == [[[0, 1, 1, 1, 0]]] if is_3D else [[0, 1, 1, 1, 0]]

    # create model
    model = create_model(x, expert_settings=widget_settings)

    # test config
    assert model.config.is_valid()

    # assert that all settings are correctly set
    assert model.config.unet_n_depth == settings['unet_n_depth']
    assert model.config.unet_kern_size == settings['unet_kern_size']
    assert model.config.unet_n_first == settings['unet_n_first']
    assert model.config.train_learning_rate == settings['train_learning_rate']
    assert model.config.n2v_perc_pix == settings['n2v_perc_pix']
    assert model.config.n2v_neighborhood_radius == settings['n2v_neighborhood_radius']
    assert model.config.n2v_manipulator == settings['n2v_manipulator']
    assert model.config.train_loss == settings['train_loss']
    assert model.config.unet_residual == settings['unet_residual']
    assert model.config.single_net_per_channel == settings['single_net_per_channel']
    assert model.config.structN2Vmask == settings['structN2Vmask']


@pytest.mark.qt
@pytest.mark.parametrize('shape', [(1, 16, 16, 1)])
def test_create_model_expert_settings_n2v2(qtbot, shape):
    """
    Tests that expert settings for N2V2 are correctly assigned.
    """
    x = np.concatenate([np.ones(shape), np.zeros(shape)])
    is_3D = False

    # create expert settings
    widget = QWidget()
    widget_settings = TrainingSettingsWidget(widget, is_3D)

    # modify the settings
    widget_settings.n2v2.setChecked(True)
    widget_settings._update_N2V2()

    # check settings
    settings = widget_settings.get_settings(is_3D=is_3D)
    assert settings['n2v_manipulator'] == 'median'
    assert settings['blurpool']
    assert settings['skip_skipone']

    # create model
    model = create_model(x, expert_settings=widget_settings)

    # test config
    assert model.config.is_valid()

    # assert that all settings are correctly set
    assert model.config.n2v_manipulator == settings['n2v_manipulator']
    assert model.config.blurpool == settings['blurpool']
    assert model.config.skip_skipone == settings['skip_skipone']
