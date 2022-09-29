import pytest

import numpy as np

from qtpy.QtWidgets import QWidget

from napari_n2v.widgets import TrainingSettingsWidget
from napari_n2v.utils import create_config, get_default_settings


@pytest.mark.parametrize('is_3D, shape', [(False, (1, 16, 16, 1)), (True, (1, 16, 16, 16, 1))])
def test_default_n2v_values(qtbot, is_3D, shape):
    # parent widget
    widget = QWidget()

    # expert settings
    widget_settings = TrainingSettingsWidget(widget, is_3D)
    settings = widget_settings.get_settings(is_3D)

    # create default N2V configuration
    config = create_config(np.ones(shape))

    # compare the default values
    assert config.unet_kern_size == settings['unet_kern_size']
    assert config.unet_n_first == settings['unet_n_first']
    assert config.unet_n_depth == settings['unet_n_depth']
    assert config.unet_residual == settings['unet_residual']
    assert config.train_learning_rate == settings['train_learning_rate']
    assert config.n2v_manipulator == settings['n2v_manipulator']
    assert config.n2v_neighborhood_radius == settings['n2v_neighborhood_radius']
    assert config.single_net_per_channel == settings['single_net_per_channel']
    assert config.structN2Vmask == settings['structN2Vmask']

    # mae is the default in N2VConfig but all example use mse loss
    assert config.train_loss != settings['train_loss']

    # same for pixel perc which 1.5 instead of 0.198
    assert config.n2v_perc_pix != settings['n2v_perc_pix']


@pytest.mark.parametrize('is_3D, shape', [(False, (1, 16, 16, 1)), (True, (1, 16, 16, 16, 1))])
def test_default_expert_values(qtbot, is_3D, shape):
    # parent widget
    widget = QWidget()

    # expert settings
    widget_settings = TrainingSettingsWidget(widget, is_3D)
    settings = widget_settings.get_settings(is_3D)

    # compare the default values
    assert settings == get_default_settings(is_3D)


@pytest.mark.parametrize('shape', [(2, 16, 16, 1), (2, 16, 16, 16, 1)])
def test_configuration_compatibility(qtbot, shape):
    is_3D = len(shape) == 5

    # parent widget
    widget = QWidget()

    # expert settings
    widget_settings = TrainingSettingsWidget(widget)
    settings = widget_settings.get_settings(is_3D)

    # create configuration using the expert settings
    x = np.ones(shape)
    x[0, ...] = np.zeros(shape[1:])  # hack to get non 0 stds
    config = create_config(x, **settings)
    assert config.is_valid()


@pytest.mark.parametrize('is_3D', [True, False])
@pytest.mark.parametrize('horizontal', [True, False])
@pytest.mark.parametrize('text, array', [('0', [0]),
                                         ('0,', [0]),
                                         ('0,0', [0, 1, 0]),
                                         ('0,,0', [0, 1, 0]),
                                         (',0,1,0,,', [0, 1, 0]),
                                         ('0,1  ,1 , 1, 0', [0, 1, 1, 1, 0])])
def test_structN2V_array(qtbot, is_3D, horizontal, text, array):
    # parent widget
    widget = QWidget()

    # expert settings
    widget_settings = TrainingSettingsWidget(widget)

    # set text and horizontal or not
    widget_settings.structN2V_text.setText(text)
    widget_settings.is_horizontal = horizontal

    # get sttings
    settings = widget_settings.get_settings(is_3D=is_3D)

    # test array
    if array:
        if horizontal:
            final_array = [array]
        else:
            final_array = [[i] for i in array]

        final_array = [final_array] if is_3D else final_array
        assert settings['structN2Vmask'] == final_array
    else:
        assert settings['structN2Vmask'] is None
