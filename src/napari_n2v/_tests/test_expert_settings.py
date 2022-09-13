import pytest

import numpy as np

from qtpy.QtWidgets import QWidget

from napari_n2v.widgets import TrainingSettingsWidget
from napari_n2v.utils import create_config


@pytest.mark.parametrize('is_3D, shape', [(False, (1, 16, 16, 1)), (True, (1, 16, 16, 16, 1))])
def test_default_values(qtbot, is_3D, shape):
    # parent widget
    widget = QWidget()

    # expert settings
    widget_settings = TrainingSettingsWidget(widget, is_3D)
    settings = widget_settings.get_settings()

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