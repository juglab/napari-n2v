"""
"""
from pathlib import Path

import napari
import napari.utils.notifications as ntf

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFormLayout,
    QComboBox,
    QFileDialog,
    QLabel,
    QTabWidget,
    QCheckBox
)
from napari_n2v.widgets import (
    TBPlotWidget,
    FolderWidget,
    AxesWidget,
    BannerWidget,
    TrainingSettingsWidget,
    ScrollWidgetWrapper,
    enable_3d,
    create_int_spinbox,
    create_progressbar,
    create_gpu_label,
    two_layers_choice
)
from napari_n2v.utils import (
    State,
    ModelSaveMode,
    UpdateType,
    train_worker,
    prediction_after_training_worker,
    loading_worker,
    save_model,
    PREDICT,
    SAMPLE
)
from napari_n2v.resources import ICON_GEAR, ICON_JUGLAB


class TrainingWidgetWrapper(ScrollWidgetWrapper):
    def __init__(self, napari_viewer):
        super().__init__(TrainWidget(napari_viewer))


class TrainWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(200)

        # add banner
        self.layout().addWidget(BannerWidget('N2V - Training',
                                             ICON_JUGLAB,
                                             'A self-supervised denoising algorithm.',
                                             'https://juglab.github.io/napari-n2v/',
                                             'https://github.com/juglab/napari-n2v/issues'))

        # add GPU button
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(gpu_button)

        # other widgets
        self._build_data_selection_widgets(napari_viewer)
        self._build_training_param_widgets()
        self._build_train_widgets()
        self._build_progress_widgets()
        self._build_predict_widgets()
        self._build_save_widgets()
        self.expert_settings = None

        # place-holders for the trained model, prediction and parameters (bioimage.io)
        self.model = None
        self.x_train, self.x_val, self.pred_train, self.pred_val = None, None, None, None
        self.new_axes = None
        self.inputs, self.outputs = None, None
        self.tf_version = None
        self.train_worker = None
        self.predict_worker = None
        self.pred_count = 0
        self.weights_path = ''
        self.is_3D = False
        self.pred_train_name, self.pred_val_name = None, None
        self.scale = None

        # actions
        self._set_actions()

        # update axes widget in case of data
        self._update_layer_axes()

    def _set_actions(self):
        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.enable_3d.use3d.changed.connect(self._update_3D)
        self.img_train.changed.connect(self._update_layer_axes)
        self.train_images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.train_button.clicked.connect(lambda: self._start_training(self.model))
        self.reset_model_button.clicked.connect(self._reset_model)
        self.n_epochs_spin.valueChanged.connect(self._update_epochs)
        self.n_steps_spin.valueChanged.connect(self._update_steps)
        self.predict_button.clicked.connect(self._start_prediction)
        self.save_button.clicked.connect(self._save_model)
        self.tiling_cbox.stateChanged.connect(self._update_tiling)

    def _build_data_selection_widgets(self, napari_viewer):
        # QTabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())
        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setTabToolTip(0, 'Use images from napari layers')
        self.tabs.setTabToolTip(1, 'Use patches saved on the disk')
        self.tabs.setMaximumHeight(200)

        # layer tabs
        self.layer_choice = two_layers_choice()
        self.img_train = self.layer_choice.Train
        self.img_val = self.layer_choice.Val
        tab_layers.layout().addWidget(self.layer_choice.native)

        self.img_train.native.setToolTip('Select an image for training')
        self.img_val.native.setToolTip('Select a n image for validation (can be the same as for training)')

        # disk tab
        self.train_images_folder = FolderWidget('Choose')
        self.val_images_folder = FolderWidget('Choose')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Train', self.train_images_folder)
        form.addRow('Val', self.val_images_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        self.train_images_folder.setToolTip('Select a folder containing the training image')
        self.val_images_folder.setToolTip('Select a folder containing the validation images')

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.img_train.choices = [x for x in self.viewer.layers if type(x) is napari.layers.Image]
        self.img_val.choices = [x for x in self.viewer.layers if type(x) is napari.layers.Image]

    def _build_training_param_widgets(self):

        self.training_param_group = QGroupBox()
        self.training_param_group.setTitle("Training parameters")
        self.training_param_group.setMinimumWidth(100)

        # expert settings
        icon = QtGui.QIcon(ICON_GEAR)
        self.training_expert_btn = QPushButton(icon, '')
        self.training_expert_btn.clicked.connect(self._training_expert_setter)
        self.training_expert_btn.setFixedSize(30, 30)
        self.training_expert_btn.setToolTip('Open the expert settings menu')

        # axes
        self.axes_widget = AxesWidget()

        # others
        self.n_epochs_spin = create_int_spinbox(1, 1000, 30, tooltip='Number of epochs')
        self.n_epochs = self.n_epochs_spin.value()

        self.n_steps_spin = create_int_spinbox(1, 1000, 200, tooltip='Number of steps per epochs')
        self.n_steps = self.n_steps_spin.value()

        # batch size
        self.batch_size_spin = create_int_spinbox(1, 512, 16, 1)
        self.batch_size_spin.setToolTip('Number of patches per batch (decrease if GPU memory is insufficient)')

        # patch size
        self.patch_XY_spin = create_int_spinbox(16, 512, 64, 8, tooltip='Dimension of the patches in XY')

        # 3D checkbox
        self.enable_3d = enable_3d()
        self.enable_3d.native.setToolTip('Use a 3D network')
        self.patch_Z_spin = create_int_spinbox(16, 512, 16, 8, False, tooltip='Dimension of the patches in Z')

        formLayout = QFormLayout()
        formLayout.addRow(self.axes_widget.label.text(), self.axes_widget.text_field)
        formLayout.addRow('Enable 3D', self.enable_3d.native)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_spin)
        formLayout.addRow('Patch XY', self.patch_XY_spin)
        formLayout.addRow('Patch Z', self.patch_Z_spin)
        formLayout.minimumSize()

        hlayout = QVBoxLayout()
        hlayout.addWidget(self.training_expert_btn, alignment=Qt.AlignRight | Qt.AlignVCenter)
        hlayout.addLayout(formLayout)

        self.training_param_group.setLayout(hlayout)
        self.training_param_group.layout().setContentsMargins(5, 20, 5, 10)
        self.layout().addWidget(self.training_param_group)

    def _training_expert_setter(self):
        if self.expert_settings is None:
            self.expert_settings = TrainingSettingsWidget(self, self.is_3D)
        self.expert_settings.show()

    def _build_train_widgets(self):
        self.train_group = QGroupBox()
        self.train_group.setTitle("Train")
        self.train_group.setLayout(QVBoxLayout())

        # train button
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton('Train', self)

        self.reset_model_button = QPushButton('', self)
        self.reset_model_button.setEnabled(False)
        self.reset_model_button.setToolTip('Reset the weights of the model (forget the training)')

        train_buttons.layout().addWidget(self.reset_model_button)
        train_buttons.layout().addWidget(self.train_button)
        self.train_group.layout().addWidget(train_buttons)

        self.layout().addWidget(self.train_group)

    def _build_save_widgets(self):
        self.save_group = QGroupBox()
        self.save_group.setTitle("Save")
        self.save_group.setLayout(QVBoxLayout())

        # Save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())
        self.save_choice = QComboBox()
        self.save_choice.addItems(ModelSaveMode.list())
        self.save_choice.setToolTip('Output format')

        self.save_button = QPushButton("Save model", self)
        self.save_button.setEnabled(False)
        self.save_choice.setToolTip('Save the model weights and configuration')

        save_widget.layout().addWidget(self.save_button)
        save_widget.layout().addWidget(self.save_choice)
        self.save_group.layout().addWidget(save_widget)

        self.layout().addWidget(self.save_group)

    def _build_progress_widgets(self):
        self.progress_group = QGroupBox()
        self.progress_group.setTitle("Training progress")
        self.progress_group.setLayout(QVBoxLayout())

        # progress bars
        self.progress_group.layout().setContentsMargins(20, 20, 20, 0)

        self.pb_epochs = create_progressbar(max_value=self.n_epochs_spin.value(),
                                            text_format=f'Epoch ?/{self.n_epochs_spin.value()}')

        self.pb_steps = create_progressbar(max_value=self.n_steps_spin.value(),
                                           text_format=f'Step ?/{self.n_steps_spin.value()}')

        self.progress_group.layout().addWidget(self.pb_epochs)
        self.progress_group.layout().addWidget(self.pb_steps)

        # plot widget
        self.plot = TBPlotWidget(max_width=300, max_height=300, min_height=250)
        self.progress_group.layout().addWidget(self.plot.native)
        self.layout().addWidget(self.progress_group)

    def _build_predict_widgets(self):
        self.predict_group = QGroupBox()
        self.predict_group.setTitle("Prediction")
        self.predict_group.setLayout(QVBoxLayout())
        self.predict_group.layout().setContentsMargins(20, 20, 20, 0)

        # checkbox
        self.tiling_cbox = QCheckBox('Tile prediction')
        self.tiling_cbox.setToolTip('Select to predict the image by tiles')
        self.predict_group.layout().addWidget(self.tiling_cbox)

        # tiling spinbox
        self.tiling_spin = create_int_spinbox(1, 1000, 4, tooltip='Minimum number of tiles to use')
        self.tiling_spin.setEnabled(False)

        tiling_form = QFormLayout()
        tiling_form.addRow('Number of tiles', self.tiling_spin)
        tiling_widget = QWidget()
        tiling_widget.setLayout(tiling_form)
        self.predict_group.layout().addWidget(tiling_widget)

        # prediction progress bar
        self.pb_prediction = create_progressbar(max_value=19,
                                                text_format=f'Prediction ?/?')
        self.pb_prediction.setToolTip('Show the progress of the prediction')

        # predict button
        predictions = QWidget()
        predictions.setLayout(QHBoxLayout())
        self.predict_button = QPushButton('', self)
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip('Run the trained model on the images')

        predictions.layout().addWidget(QLabel(''))
        predictions.layout().addWidget(self.predict_button)

        # add to the group
        self.predict_group.layout().addWidget(self.pb_prediction)
        self.predict_group.layout().addWidget(predictions)
        self.layout().addWidget(self.predict_group)

    def _start_training(self,  pretrained_model=None):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():
                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                if not self.load_from_disk:
                    if self.img_train.value is None:
                        # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                        # ntf.show_error('No layer selected for training.')
                        ntf.show_info('No layer selected for training.')
                        return

                self.state = State.RUNNING

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # modify UI
                self.plot.clear_plot()
                self.train_button.setText('Stop')
                self.reset_model_button.setText('')
                self.reset_model_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.predict_button.setEnabled(False)
                self.predict_button.setText('')

                self.train_worker = train_worker(self,
                                                 pretrained_model=pretrained_model,
                                                 expert_settings=self.expert_settings)
                self.train_worker.yielded.connect(lambda x: self._update_all(x))
                self.train_worker.returned.connect(self._training_done)
                self.train_worker.start()
            else:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error('Invalid axes')
                ntf.show_info('Invalid axes')
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _start_prediction(self):
        if self.state == State.IDLE:
            if self.model is not None:
                self.state = State.RUNNING

                self.pb_prediction.setValue(0)

                self.predict_button.setText('Stop')

                # prepare layers name and remove them if they exist
                self.pred_train_name = self.img_train.name + PREDICT
                self.pred_val_name = self.img_val.name + PREDICT
                if self.pred_train_name in self.viewer.layers:
                    self.viewer.layers.remove(self.pred_train_name)
                if self.pred_val_name in self.viewer.layers:
                    self.viewer.layers.remove(self.pred_val_name)

                # images are already in CSBDeep axes order
                if type(self.x_train) == tuple:  # np.array
                    self.pred_count = len(self.x_train[0])
                else:  # list of np.
                    self.pred_count = self.x_train.shape[0]
                self.pred_train = None

                # also predict val if val is different from x_train
                if self.x_val is not None:
                    if type(self.x_val) == tuple:
                        self.pred_count += len(self.x_val[0])
                    else:
                        self.pred_count += self.x_val.shape[0]
                self.pred_val = None

                self.pb_prediction.setMaximum(self.pred_count)

                self.predict_worker = prediction_after_training_worker(self)
                self.predict_worker.yielded.connect(lambda x: self._update_prediction(x))
                self.predict_worker.start()
            else:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error('No model available.')
                ntf.show_info('No model available.')
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _training_done(self):
        self.state = State.IDLE
        self.train_button.setText('Continue training')
        self.reset_model_button.setText('Reset model')
        self.reset_model_button.setEnabled(True)

        self.save_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.predict_button.setText('Predict')

    def _prediction_done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

        if self.pred_train is not None:
            if self.scale is not None:
                self.viewer.add_image(self.pred_train, name=self.pred_train_name, scale=self.scale, visible=True)
            else:
                self.viewer.add_image(self.pred_train, name=self.pred_train_name, visible=True)

        if self.pred_val is not None:
            if self.scale is not None:
                self.viewer.add_image(self.pred_val, name=self.pred_val_name, scale=self.scale, visible=True)
            else:
                self.viewer.add_image(self.pred_val, name=self.pred_val_name, visible=True)

    def _update_prediction(self, update):
        if self.state == State.RUNNING:
            if update == UpdateType.DONE:
                self._prediction_done()
            else:
                val = update[UpdateType.PRED]
                self.pb_prediction.setValue(val)
                self.pb_prediction.setFormat(f'Prediction {val}/{self.pred_count}')

    def _reset_model(self):
        """
        Reset the model, causing the next training session to train from scratch.
        :return:
        """
        if self.state == State.IDLE:
            self.model = None
            self.reset_model_button.setText('')
            self.reset_model_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.predict_button.setText('')
            self.predict_button.setEnabled(False)
            self.train_button.setText('Train')

    def _update_3D(self, state):
        """
        Update the UI based on the status of the 3D checkbox.
        :param state:
        :return:
        """
        self.is_3D = state
        self.patch_Z_spin.setVisible(self.is_3D)

        # update axes widget
        self.axes_widget.update_is_3D(self.is_3D)
        self.axes_widget.set_text_field(self.axes_widget.get_default_text())

        # update expert settings
        if self.expert_settings:
            self.expert_settings.update_3D(state)

    def _update_tiling(self, state):
        self.tiling_spin.setEnabled(state)

    def _update_layer_axes(self):
        """
        Update the axes widget based on the shape of the data selected in the layer selection drop-down widget.
        :return:
        """
        if self.img_train.value is not None:
            shape = self.img_train.value.data.shape

            # update shape length in the axes widget
            self.axes_widget.update_axes_number(len(shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_disk_axes(self):
        # TODO: this is quite complex for loading a single image... Should load it directly
        """
        Load an example image from the disk and update the axes widget based on its shape.

        :return:
        """
        def add_image(widget, image):
            if image is not None:
                if SAMPLE in widget.viewer.layers:
                    widget.viewer.layers.remove(SAMPLE)

                widget.viewer.add_image(image, name=SAMPLE, visible=True)

                # update the axes widget
                widget.axes_widget.update_axes_number(len(image.shape))
                widget.axes_widget.set_text_field(widget.axes_widget.get_default_text())

        path = self.train_images_folder.get_folder()

        if path is not None and path != '':
            # load one image
            load_worker = loading_worker(path)
            load_worker.yielded.connect(lambda x: add_image(self, x))
            load_worker.start()

    def _update_tab_axes(self):
        """
        Updates the axes widget following the newly selected tab.

        :return:
        """
        self.load_from_disk = self.tabs.currentIndex() == 1

        if self.load_from_disk:
            self._update_disk_axes()
        else:
            self._update_layer_axes()

    def _update_epochs(self):
        if self.state == State.IDLE:
            self.n_epochs = self.n_epochs_spin.value()
            self.pb_epochs.setValue(0)
            self.pb_epochs.setMaximum(self.n_epochs)
            self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs}')

    def _update_steps(self):
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setMaximum(self.n_steps)
            self.pb_steps.setFormat(f'Step ?/{self.n_steps}')

    def _update_all(self, updates):
        if self.state == State.RUNNING:
            if UpdateType.EPOCH in updates:
                val = updates[UpdateType.EPOCH]
                self.pb_epochs.setValue(val)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if UpdateType.BATCH in updates:
                val = updates[UpdateType.BATCH]
                self.pb_steps.setValue(val)
                self.pb_steps.setFormat(f'Step {val}/{self.n_steps}')

            if UpdateType.LOSS in updates:
                self.plot.update_plot(*updates[UpdateType.LOSS])

    def _save_model(self):
        if self.state == State.IDLE:
            if self.model:
                destination = Path(QFileDialog.getSaveFileName(caption='Save model')[0])
                export_type = self.save_choice.currentText()

                # save
                parameters = {
                    'export_type': export_type,
                    'model': self.model,
                    'axes': self.new_axes,
                    'input_path': self.inputs,
                    'output_path': self.outputs,
                    'tf_version': self.tf_version
                }
                save_model(destination, **parameters)

    def is_tiling_checked(self):
        return self.tiling_cbox.isChecked()

    def get_n_tiles(self):
        return self.tiling_spin.value()

    def get_batch_size(self):
        return self.batch_size_spin.value()

    def get_patch_XY(self):
        return self.patch_XY_spin.value()

    def get_patch_Z(self):
        return self.patch_Z_spin.value()


if __name__ == "__main__":
    from napari_n2v._sample_data import n2v_2D_data, n2v_3D_data

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(TrainingWidgetWrapper(viewer))

    dims = '2D'  # 2D, 3D
    if dims == '2D':
        data = n2v_2D_data()

        # add images
        viewer.add_image(data[0][0][0:50], name=data[0][1]['name'])
        viewer.add_image(data[1][0], name=data[1][1]['name'])
    else:
        data = n2v_3D_data()

        viewer.add_image(data[0][0], name=data[0][1]['name'])

    napari.run()
