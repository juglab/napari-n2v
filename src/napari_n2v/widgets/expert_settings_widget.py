
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget
)
from .qt_widgets import create_int_spinbox, create_double_spinbox
from .magicgui_widgets import load_button


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

        # create widgets
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

        label_n2v_perc_pix = QLabel('N2V pixel %')
        desc_n2v_perc_pix = 'Percentage of pixel to mask per patch'
        self.n2v_perc_pix = create_double_spinbox(value=n2v_perc_pix, step=0.1, max_value=100)
        self.n2v_perc_pix.setDecimals(1)
        self.n2v_perc_pix.setToolTip(desc_n2v_perc_pix)
        label_n2v_perc_pix.setToolTip(desc_n2v_perc_pix)

        label_n2v_neighborhood_radius = QLabel('N2V radius')
        desc_n2v_neighborhood_radius = 'Neighborhood radius for n2v manipulator'
        self.n2v_neighborhood_radius = create_int_spinbox(value=n2v_neighborhood_radius, min_value=3, max_value=16)
        self.n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)
        label_n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)

        self.load_model_button = load_button()
        self.load_model_button.native.setToolTip('Load a pre-trained model (weights and configuration)')

        # arrange form layout
        form = QFormLayout()
        form.addRow(label_unet_depth, self.unet_depth)
        form.addRow(label_unet_kernelsize, self.unet_kernelsize)
        form.addRow(label_unet_n_first, self.unet_n_first)
        form.addRow(label_train_learning_rate, self.train_learning_rate)
        form.addRow(label_n2v_perc_pix, self.n2v_perc_pix)
        form.addRow(label_n2v_neighborhood_radius, self.n2v_neighborhood_radius)

        form_widget = QWidget()
        form_widget.setLayout(form)

        self.layout().addWidget(form_widget)
        self.layout().addWidget(self.load_model_button.native)

    def get_model_path(self):
        return self.load_model_button.Model.value

    def has_model(self):
        return self.get_model_path().exists() and self.get_model_path().is_file()

    def get_settings(self):
        return {'unet_kern_size': self.unet_kernelsize.value(),
                'unet_n_first': self.unet_n_first.value(),
                'unet_n_depth': self.unet_depth.value(),
                'train_learning_rate': self.train_learning_rate.value(),
                'n2v_perc_pix': self.n2v_perc_pix.value(),
                'n2v_neighborhood_radius': self.n2v_neighborhood_radius.value()}
