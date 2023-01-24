from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QCheckBox
)
from napari_n2v.utils import get_pms, get_losses, get_default_settings
from .qt_widgets import create_int_spinbox, create_double_spinbox
from .magicgui_widgets import load_button
from .axes_widget import LettersValidator


class TrainingSettingsWidget(QDialog):

    def __init__(self, parent, is_3D=False):
        super().__init__(parent)
        self.setWindowTitle('Expert settings')
        self.setLayout(QVBoxLayout())

        # defaults values
        default_settings = get_default_settings(is_3D)
        unet_n_depth = default_settings['unet_n_depth']
        unet_kern_size = default_settings['unet_kern_size']
        unet_n_first = default_settings['unet_n_first']
        train_learning_rate = default_settings['train_learning_rate']
        n2v_perc_pix = default_settings['n2v_perc_pix']
        n2v_neighborhood_radius = default_settings['n2v_neighborhood_radius']
        n2v_manipulator = default_settings['n2v_manipulator']
        train_loss = default_settings['train_loss']
        unet_residual = default_settings['unet_residual']
        single_net_per_channel = default_settings['single_net_per_channel']
        use_n2v2 = default_settings['blurpool'] == True and \
                   default_settings['skip_skipone'] == True and \
                   default_settings['n2v_manipulator'] == 'median'
        use_augment = True
        n_val = 5

        # groups
        self.retraining = QGroupBox()
        self.retraining.setTitle("Retrain model")

        self.expert_settings = QGroupBox()
        self.expert_settings.setTitle("Expert settings")

        self.structN2V = QGroupBox()
        self.structN2V.setTitle("structN2V")

        ####################################################
        # create widgets for expert settings
        label_n_validation = QLabel('N validation')
        desc_n_validation = 'Number of patches used for validation. This is only used when no\n' \
                            'validation data is defined (i.e. validation is taken from the\n' \
                            'training patches.).'
        self.n_val = create_int_spinbox(value=n_val, min_value=1, max_value=1000)
        label_n_validation.setToolTip(desc_n_validation)
        self.n_val.setToolTip(desc_n_validation)

        # augmentation
        label_augment = QLabel('Use augmentation')
        desc_augment = 'If checked, augmentation will be applied to the training patches. Note\n' \
                       'that it is incompatible with structN2V (augmentation is off).'
        self.augment = QCheckBox()
        self.augment.setChecked(use_augment)
        label_augment.setToolTip(desc_augment)
        self.augment.setToolTip(desc_augment)

        # n2v2
        label_n2v2 = QLabel('Use N2V2')
        desc_n2v2 = 'If checked, the model will use N2V2, a version of N2V that mitigates\n' \
                    'check-board artefacts. This only works with 2D data and uses a median\n' \
                    'pixel manipulator. N2V2 is currently not compatible with structN2V.'
        self.n2v2 = QCheckBox()
        self.n2v2.setChecked(use_n2v2)
        label_n2v2.setToolTip(desc_n2v2)
        self.n2v2.setToolTip(desc_n2v2)
        self.n2v2.stateChanged.connect(self._update_N2V2)

        label_unet_depth = QLabel('U-Net depth')
        desc_unet_depth = 'Number of resolution levels of the U-Net architecture'
        self.unet_depth = create_int_spinbox(value=unet_n_depth, min_value=2, max_value=5)
        label_unet_depth.setToolTip(desc_unet_depth)
        self.unet_depth.setToolTip(desc_unet_depth)

        label_unet_kernelsize = QLabel('U-Net kernel size')
        desc_unet_kernelsize = 'Size of the convolution filters in all image dimensions'
        self.unet_kernelsize = create_int_spinbox(value=unet_kern_size, min_value=3, max_value=9, step=2)
        label_unet_kernelsize.setToolTip(desc_unet_kernelsize)
        self.unet_kernelsize.setToolTip(desc_unet_kernelsize)

        label_unet_n_first = QLabel('U-Net n filters')
        desc_unet_n_first = 'Number of convolution filters for first U-Net resolution level\n' \
                            '(value is doubled after each down-sampling operation)'
        self.unet_n_first = create_int_spinbox(value=unet_n_first, min_value=8, step=8)
        label_unet_n_first.setToolTip(desc_unet_n_first)
        self.unet_n_first.setToolTip(desc_unet_n_first)

        label_unet_residuals = QLabel('U-Net residuals')
        desc_unet_residuals = 'If checked, model will internally predict the residual w.r.t.\n' \
                              'the input (typically better), this requires the number of input\n' \
                              'and output image channels to be equal'
        self.unet_residuals = QCheckBox()
        self.unet_residuals.setChecked(unet_residual)
        label_unet_residuals.setToolTip(desc_unet_residuals)
        self.unet_residuals.setToolTip(desc_unet_residuals)

        label_train_learning_rate = QLabel('Learning rate')
        desc_train_learning_rate = 'Starting learning rate'
        self.train_learning_rate = create_double_spinbox(step=0.0001, n_decimal=4)
        self.train_learning_rate.setValue(train_learning_rate)  # TODO: bug? cannot be set in create_double_spinbox.
        label_train_learning_rate.setToolTip(desc_train_learning_rate)
        self.train_learning_rate.setToolTip(desc_train_learning_rate)

        label_loss = QLabel('Train train_loss')
        desc_loss = 'Loss used to train the network.'
        self.loss_combobox = QComboBox()
        for s in get_losses():
            self.loss_combobox.addItem(s)
        self.loss = train_loss
        self.loss_combobox.activated[str].connect(self._on_loss_change)

        self.loss_combobox.setToolTip(desc_loss)
        label_loss.setToolTip(desc_loss)

        label_n2v_perc_pix = QLabel('N2V pixel %')
        desc_n2v_perc_pix = 'Percentage of pixel to mask per patch'
        self.n2v_perc_pix = create_double_spinbox(value=n2v_perc_pix, step=0.001, max_value=100, n_decimal=3)
        self.n2v_perc_pix.setToolTip(desc_n2v_perc_pix)
        label_n2v_perc_pix.setToolTip(desc_n2v_perc_pix)

        label_n2v_manipulator = QLabel('N2V manipulator')
        desc_n2v_manipulator = 'Pixel manipulator.'
        self.n2v_pmanipulator = QComboBox()
        for s in get_pms():
            self.n2v_pmanipulator.addItem(s)
        self.n2v_pm = n2v_manipulator
        self.n2v_pmanipulator.activated[str].connect(self._on_pm_change)

        self.n2v_pmanipulator.setToolTip(desc_n2v_manipulator)
        label_n2v_manipulator.setToolTip(desc_n2v_manipulator)

        label_n2v_neighborhood_radius = QLabel('N2V radius')
        desc_n2v_neighborhood_radius = 'Neighborhood radius for n2v manipulator'
        self.n2v_neighborhood_radius = create_int_spinbox(value=n2v_neighborhood_radius,
                                                          min_value=2,
                                                          step=1,
                                                          max_value=17)
        self.n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)
        label_n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)

        label_single_net = QLabel('Split channels')
        desc_single_net = 'Enabling this creates a unet for each channel and each channel will be ' \
                          'treated independently.'
        self.single_net = QCheckBox()
        self.single_net.setChecked(single_net_per_channel)
        label_single_net.setToolTip(desc_single_net)
        self.single_net.setToolTip(desc_single_net)

        # arrange form layout
        form = QFormLayout()
        form.addRow(label_n_validation, self.n_val)
        form.addRow(label_augment, self.augment)
        form.addRow(label_n2v2, self.n2v2)
        form.addRow(label_unet_depth, self.unet_depth)
        form.addRow(label_unet_kernelsize, self.unet_kernelsize)
        form.addRow(label_unet_n_first, self.unet_n_first)
        form.addRow(label_unet_residuals, self.unet_residuals)
        form.addRow(label_train_learning_rate, self.train_learning_rate)
        form.addRow(label_loss, self.loss_combobox)
        form.addRow(label_n2v_perc_pix, self.n2v_perc_pix)
        form.addRow(label_n2v_manipulator, self.n2v_pmanipulator)
        form.addRow(label_n2v_neighborhood_radius, self.n2v_neighborhood_radius)
        form.addRow(label_single_net, self.single_net)

        self.expert_settings.setLayout(form)

        ####################################################
        # create widgets for load model
        self.load_model_button = load_button()
        self.load_model_button.native.setToolTip('Load a pre-trained model (weights and configuration).')

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

        self.structN2V_mask = QLineEdit()
        self.structN2V_mask.setValidator(LettersValidator('0,1'))
        self.structN2V_mask.textChanged.connect(self._on_structn2v_mask_change)  # remove augmentation if mask

        self.hv_choice = QComboBox()
        self.hv_choice.addItem('horizontal')
        self.hv_choice.addItem('vertical')
        self.hv_choice.setToolTip('Choose the orientation of the structN2V mask.')
        self.is_horizontal = True
        self.hv_choice.activated[str].connect(self._on_structn2v_orientation_changed)

        label_n2v_perc_pix.setToolTip(desc_structn2v)
        self.structN2V_mask.setToolTip(desc_structn2v)

        # arrange form layout
        form = QFormLayout()

        form.addRow('', self.hv_choice)
        form.addRow(label_n2v_perc_pix, self.structN2V_mask)

        self.structN2V.setLayout(form)

        ####################################################
        # assemble expert settings
        self.layout().addWidget(self.retraining)
        self.layout().addWidget(self.expert_settings)
        self.layout().addWidget(self.structN2V)

    def _on_loss_change(self, val):
        self.loss = val

    def _on_pm_change(self, val):
        self.n2v_pm = val

    def _on_structn2v_mask_change(self):
        if self.has_mask():
            self.augment.setChecked(False)
            self.augment.setEnabled(False)
        else:
            self.augment.setChecked(True)
            self.augment.setEnabled(True)

    def _on_structn2v_orientation_changed(self, val):
        self.is_horizontal = val == 'horizontal'

    def _update_N2V2(self):
        if self.n2v2.isChecked():
            # we need median pixel manipulator
            self.n2v_pmanipulator.setCurrentText('median')
            self.n2v_pm = 'median'
            self.n2v_pmanipulator.setEnabled(False)

            # no residuals
            self.unet_residuals.setChecked(False)
            self.unet_residuals.setEnabled(False)

            # no structN2V
            self.structN2V_mask.setEnabled(False)
            self.structN2V_mask.setText('')
        else:
            self.n2v_pmanipulator.setEnabled(True)
            self.unet_residuals.setEnabled(True)
            self.structN2V_mask.setEnabled(True)

            # change the pixel manipulator to default
            if self.n2v_pm == 'median':
                # since checking N2V2 selects the median pixel manipulator
                # we consider that if it was checked, we need to change the pm
                self.n2v_pmanipulator.setCurrentText(get_pms()[0])
                self.n2v_pm = get_pms()[0]

    def get_model_path(self):
        return self.load_model_button.Model.value

    def has_model(self):
        return self.get_model_path().exists() and self.get_model_path().is_file()

    def has_mask(self):
        return self.structN2V_mask.text() != ''

    def get_val_size(self):
        return self.n_val.value()

    def use_augmentation(self):
        return self.augment.isChecked()

    # todo could refactor this into a single function easy to test
    # todo currently only 1D
    def _get_structN2V(self, is_3D=False):
        if self.structN2V_mask.text() != '':
            mask = self.structN2V_mask.text()

            # make sure there's no multiple ',' and no space
            mask = mask.replace(' ', '')
            mask = mask.split(',')
            mask = [s for s in mask if len(s) == 1]  # removes '' / '10' etc.

            # check that it is odd, otherwise extend by one
            if len(mask) % 2 == 0:
                mask = mask[: len(mask) // 2] + ['1'] + mask[len(mask) // 2:]

            # check orientation
            if self.is_horizontal:
                mask = [[int(s) for s in mask]]
            else:
                mask = [[int(s)] for s in mask]

            return [mask] if is_3D else mask
        else:
            return None

    def _get_pixel_manipulator(self, is_3D):
        if self.n2v2.isChecked() and not is_3D:
            return 'median'
        else:
            return self.n2v_pm

    def _is_N2V2(self, is_3D):
        return self.n2v2.isChecked() and not is_3D

    def update_3D(self, is_3D):
        if is_3D:
            # change the pixel manipulator to default
            if self.n2v2.isChecked() and self.n2v_pm == 'median':
                # since checking N2V2 selects the median pixel manipulator
                # we consider that if it was checked, we need to change the pm
                self.n2v_pmanipulator.setCurrentText(get_pms()[0])
                self.n2v_pm = get_pms()[0]

            # uncheck and disable N2V2
            self.n2v2.setChecked(False)
            self.n2v2.setEnabled(False)

            # enable the pixel manipulator and structN2V
            self.n2v_pmanipulator.setEnabled(True)
            self.structN2V_mask.setEnabled(True)
        else:
            # enable N2V2
            self.n2v2.setEnabled(True)

    def get_settings(self, is_3D):
        return {
            'unet_kern_size': self.unet_kernelsize.value(),
            'unet_n_first': self.unet_n_first.value(),
            'unet_n_depth': self.unet_depth.value(),
            'unet_residual': self.unet_residuals.isChecked(),
            'train_learning_rate': self.train_learning_rate.value(),
            'train_loss': self.loss,
            'n2v_perc_pix': self.n2v_perc_pix.value(),
            'n2v_manipulator': self._get_pixel_manipulator(is_3D),
            'n2v_neighborhood_radius': self.n2v_neighborhood_radius.value(),
            'single_net_per_channel': self.single_net.isChecked(),
            'structN2Vmask': self._get_structN2V(is_3D),
            'blurpool': self._is_N2V2(is_3D),
            'skip_skipone': self._is_N2V2(is_3D)
        }
