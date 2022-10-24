from enum import Enum


class PixelManipulator(Enum):
    UNIFORM_WITH_CP = 'uniform_withCP'
    UNIFORM_WITHOUT_CP = 'uniform_withoutCP'
    NORMAL_WITHOUT_CP = 'normal_withoutCP'
    NORMAL_ADDITIVE = 'normal_additive'
    NORMAL_FITTED = 'normal_fitted'
    IDENTITY = 'identity'
    MEAN = 'mean'
    MEDIAN = 'median'


def get_pms():
    return [PixelManipulator.UNIFORM_WITH_CP.value,
            PixelManipulator.UNIFORM_WITHOUT_CP.value,
            PixelManipulator.NORMAL_WITHOUT_CP.value,
            PixelManipulator.NORMAL_ADDITIVE.value,
            PixelManipulator.NORMAL_ADDITIVE.value,
            PixelManipulator.NORMAL_FITTED.value,
            PixelManipulator.IDENTITY.value,
            PixelManipulator.MEAN.value,
            PixelManipulator.MEDIAN.value]


class Loss(Enum):
    MSE = 'mse'
    MAE = 'mae'


def get_losses():
    return [Loss.MSE.value, Loss.MAE.value]


def get_default_settings(is_3D):
    return {
        'unet_kern_size': 5 if not is_3D else 3,
        'unet_n_first': 32,
        'unet_n_depth': 2,
        'unet_residual': False,
        'train_learning_rate': 0.0004,
        'train_loss': Loss.MSE.value,
        'n2v_perc_pix': 0.198,
        'n2v_manipulator': PixelManipulator.UNIFORM_WITH_CP.value,
        'n2v_neighborhood_radius': 5,
        'single_net_per_channel': True,
        'structN2Vmask': None,
        'blurpool': False,
        'skip_skipone': False
    }
