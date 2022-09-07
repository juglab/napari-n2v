from enum import Enum
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QGroupBox,
    QComboBox
)
from .qt_widgets import create_int_spinbox, create_double_spinbox
from .magicgui_widgets import load_button
from .axes_widget import LettersValidator


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


# TODO Although this a widget, the presence in the widgets module of some important N2V-related settings is confusing
class TrainingSettingsWidget(QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Expert settings')
        self.setLayout(QVBoxLayout())

        # defaults values
        unet_n_depth = 2
        unet_kern_size = 3
        unet_n_first = 96
        train_learning_rate = 0.0004
        n2v_perc_pix = 0.198
        n2v_neighborhood_radius = 2
        n2v_pm = get_pms()[0]
        loss = get_losses()[0]

        # groups
        self.retraining = QGroupBox()
        self.retraining.setTitle("Retrain model")

        self.expert_settings = QGroupBox()
        self.expert_settings.setTitle("Expert settings")

        self.structN2V = QGroupBox()
        self.structN2V.setTitle("structN2V")

        ####################################################
        # create widgets for expert settings
        label_unet_depth = QLabel('U-Net depth')
        desc_unet_depth = 'Number of resolution levels of the U-Net architecture'
        self.unet_depth = create_int_spinbox(value=unet_n_depth, min_value=2, max_value=5)
        label_unet_depth.setToolTip(desc_unet_depth)
        self.unet_depth.setToolTip(desc_unet_depth)

        label_unet_kernelsize = QLabel('U-Net kernel size')
        desc_unet_kernelsize = 'Size of the convolution filters in all image dimensions'
        self.unet_kernelsize = create_int_spinbox(value=unet_kern_size, min_value=3, max_value=7, step=2)
        label_unet_kernelsize.setToolTip(desc_unet_kernelsize)
        self.unet_kernelsize.setToolTip(desc_unet_kernelsize)

        label_unet_n_first = QLabel('U-Net n filters')
        desc_unet_n_first = 'Number of convolution filters for first U-Net resolution level (value is doubled after ' \
                            'each down-sampling operation) '
        self.unet_n_first = create_int_spinbox(value=unet_n_first, min_value=8, step=8)
        label_unet_n_first.setToolTip(desc_unet_n_first)
        self.unet_n_first.setToolTip(desc_unet_n_first)

        label_train_learning_rate = QLabel('Learning rate')
        desc_train_learning_rate = 'Fixed learning rate'
        self.train_learning_rate = create_double_spinbox(step=0.0001)
        self.train_learning_rate.setDecimals(4)
        self.train_learning_rate.setValue(train_learning_rate)  # TODO: bug? cannot be set in create_double_spinbox.
        label_train_learning_rate.setToolTip(desc_train_learning_rate)
        self.train_learning_rate.setToolTip(desc_train_learning_rate)

        label_loss = QLabel('Train loss')
        desc_loss = 'Loss used to train the network.'
        self.loss_combobox = QComboBox()
        for s in get_losses():
            self.loss_combobox.addItem(s)
        self.loss = loss
        self.loss_combobox.activated[str].connect(self._onLossChange)

        self.loss_combobox.setToolTip(desc_loss)
        label_loss.setToolTip(desc_loss)

        label_n2v_perc_pix = QLabel('N2V pixel %')
        desc_n2v_perc_pix = 'Percentage of pixel to mask per patch'
        self.n2v_perc_pix = create_double_spinbox(value=n2v_perc_pix, step=0.1, max_value=100)
        self.n2v_perc_pix.setDecimals(1)
        self.n2v_perc_pix.setToolTip(desc_n2v_perc_pix)
        label_n2v_perc_pix.setToolTip(desc_n2v_perc_pix)

        label_n2v_manipulator = QLabel('N2V manipulator')
        desc_n2v_manipulator = 'Pixel manipulator.'
        self.n2v_pmanipulator = QComboBox()
        for s in get_pms():
            self.n2v_pmanipulator.addItem(s)
        self.n2v_pm = n2v_pm
        self.n2v_pmanipulator.activated[str].connect(self._onPMChange)

        self.n2v_pmanipulator.setToolTip(desc_n2v_manipulator)
        label_n2v_manipulator.setToolTip(desc_n2v_manipulator)

        label_n2v_neighborhood_radius = QLabel('N2V radius')
        desc_n2v_neighborhood_radius = 'Neighborhood radius for n2v manipulator'
        self.n2v_neighborhood_radius = create_int_spinbox(value=n2v_neighborhood_radius, min_value=3, max_value=16)
        self.n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)
        label_n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)

        # arrange form layout
        form = QFormLayout()
        form.addRow(label_unet_depth, self.unet_depth)
        form.addRow(label_unet_kernelsize, self.unet_kernelsize)
        form.addRow(label_unet_n_first, self.unet_n_first)
        form.addRow(label_train_learning_rate, self.train_learning_rate)
        form.addRow(label_loss, self.loss_combobox)
        form.addRow(label_n2v_perc_pix, self.n2v_perc_pix)
        form.addRow(label_n2v_manipulator, self.n2v_pmanipulator)
        form.addRow(label_n2v_neighborhood_radius, self.n2v_neighborhood_radius)

        self.expert_settings.setLayout(form)

        ####################################################
        # create widgets for load model
        self.load_model_button = load_button()
        self.load_model_button.native.setToolTip('Load a pre-trained model (weights and configuration)')

        self.retraining.setLayout(QVBoxLayout())
        self.retraining.layout().addWidget(self.load_model_button.native)

        ####################################################
        # create widgets for structN2V
        label_n2v_perc_pix = QLabel('structN2V mask')
        desc_structn2v = 'Mask for structN2V, a N2V flavor which removes structural noise along\n' \
                         'an axis. For instance enter:\n' \
                         '                       0,1,1,1,1,1,0\n' \
                         'Note the following points:\n' \
                         '  - the size of the mask is also chosen according to the noise\n' \
                         '  - the mask should be an odd sequence (otherwise 1 is automatically added)\n' \
                         '  - currently augmentation is disabled with structN2V'

        self.structN2V_text = QLineEdit()
        self.structN2V_text.setValidator(LettersValidator('0,1'))

        self.hv_choice = QComboBox()
        self.hv_choice.addItem('horizontal')
        self.hv_choice.addItem('vertical')
        self.hv_choice.setToolTip('Choose the orientation of the structN2V mask.')
        self.is_horizontal = True
        self.hv_choice.activated[str].connect(self._onOrientationChanged)

        label_n2v_perc_pix.setToolTip(desc_structn2v)
        self.structN2V_text.setToolTip(desc_structn2v)

        # arrange form layout
        form = QFormLayout()

        form.addRow('', self.hv_choice)
        form.addRow(label_n2v_perc_pix, self.structN2V_text)

        self.structN2V.setLayout(form)

        ####################################################
        # assemble expert settings
        self.layout().addWidget(self.retraining)
        self.layout().addWidget(self.expert_settings)
        self.layout().addWidget(self.structN2V)

    def _onLossChange(self, val):
        self.loss = val

    def _onPMChange(self, val):
        self.n2v_pm = val

    def _onOrientationChanged(self, val):
        self.is_horizontal = val == 'horizontal'

    def get_model_path(self):
        return self.load_model_button.Model.value

    def has_model(self):
        return self.get_model_path().exists() and self.get_model_path().is_file()

    def has_mask(self):
        return self.structN2V_text.text() == ''

    def _get_structN2V(self, is_3D=False):
        if self.structN2V_text.text() != '':
            mask = self.structN2V_text.text()

            # make sure there's no multiple ','
            mask = mask.split(',')
            mask = [s for s in mask if len(s) == 1]  # removes '' / '10' etc.

            # check that it is odd, otherwise extend by one
            if len(mask) % 2 == 0:
                mask = mask[: len(mask)//2] + ['1'] + mask[len(mask)//2:]

            # check orientation
            if self.is_horizontal:
                mask = [[int(s) for s in mask]]
            else:
                mask = [[int(s)] for s in mask]

            return [mask] if is_3D else mask
        else:
            return None

    def get_settings(self, is_3D=False):
        return {'unet_kern_size': self.unet_kernelsize.value(),
                'unet_n_first': self.unet_n_first.value(),
                'unet_n_depth': self.unet_depth.value(),
                'train_learning_rate': self.train_learning_rate.value(),
                'train_loss': self.loss,
                'n2v_perc_pix': self.n2v_perc_pix.value(),
                'n2v_manipulator': self.n2v_pm,
                'n2v_neighborhood_radius': self.n2v_neighborhood_radius.value(),
                'structN2Vmask': self._get_structN2V(is_3D)}
