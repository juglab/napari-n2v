"""
"""
import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QFormLayout,
    QComboBox,
    QFileDialog,
    QLabel,
    QTabWidget
)

from napari_n2v.widgets import (
    TBPlotWidget,
    AxesWidget,
    FolderWidget,
    two_layers_choice,
    enable_3d
)
from napari_n2v.utils import (
    State,
    SaveMode,
    UpdateType,
    train_worker,
    prediction_after_training_worker,
    loading_worker,
    build_modelzoo,
    reshape_napari,
    get_shape_order,
    PREDICT,
    SAMPLE,
    NAPARI_AXES
)


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        ###############################
        # QTabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())

        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setMaximumHeight(200)

        # layer tabs
        self.layer_choice = two_layers_choice()
        self.img_train = self.layer_choice.Train
        self.img_val = self.layer_choice.Val
        tab_layers.layout().addWidget(self.layer_choice.native)

        # disk tab
        self.train_images_folder = FolderWidget('Choose')
        self.val_images_folder = FolderWidget('Choose')

        buttons = QWidget()
        form = QFormLayout()

        form.addRow('Train images', self.train_images_folder)
        form.addRow('Val images', self.val_images_folder)

        buttons.setLayout(form)
        tab_disk.layout().addWidget(buttons)

        # add to main layout
        self.layout().addWidget(self.tabs)
        self.img_train.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]
        self.img_val.choices = [x for x in napari_viewer.layers if type(x) is napari.layers.Image]

        ###############################
        # axes
        self.axes_widget = AxesWidget()

        # number of epochs
        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setMinimum(1)
        self.n_epochs_spin.setMaximum(1000)
        self.n_epochs_spin.setValue(2)
        self.n_epochs = self.n_epochs_spin.value()

        # number of steps
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setMaximum(1000)
        self.n_steps_spin.setMinimum(1)
        self.n_steps_spin.setValue(2)
        self.n_steps = self.n_steps_spin.value()

        # batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMaximum(512)
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setSingleStep(1)
        self.batch_size_spin.setValue(8)

        # patch XY size
        self.patch_XY_spin = QSpinBox()
        self.patch_XY_spin.setMaximum(512)
        self.patch_XY_spin.setMinimum(16)
        self.patch_XY_spin.setSingleStep(8)
        self.patch_XY_spin.setValue(64)

        # 3D checkbox
        self.enable_3d = enable_3d()
        self.patch_Z_spin = QSpinBox()
        self.patch_Z_spin.setMaximum(512)
        self.patch_Z_spin.setMinimum(16)
        self.patch_Z_spin.setSingleStep(8)
        self.patch_Z_spin.setValue(16)
        self.patch_Z_spin.setVisible(False)

        # add widgets
        # TODO add tooltips
        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('', self.axes_widget)
        formLayout.addRow('Enable 3D', self.enable_3d.native)
        formLayout.addRow('N epochs', self.n_epochs_spin)
        formLayout.addRow('N steps', self.n_steps_spin)
        formLayout.addRow('Batch size', self.batch_size_spin)
        formLayout.addRow('Patch XY', self.patch_XY_spin)
        formLayout.addRow('Patch Z', self.patch_Z_spin)
        others.setLayout(formLayout)
        self.layout().addWidget(others)

        # progress bars
        progress_widget = QWidget()
        progress_widget.setLayout(QVBoxLayout())

        self.pb_epochs = QProgressBar()
        self.pb_epochs.setValue(0)
        self.pb_epochs.setMinimum(0)
        self.pb_epochs.setMaximum(100)
        self.pb_epochs.setTextVisible(True)
        self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

        self.pb_steps = QProgressBar()
        self.pb_steps.setValue(0)
        self.pb_steps.setMinimum(0)
        self.pb_steps.setMaximum(100)
        self.pb_steps.setTextVisible(True)
        self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

        progress_widget.layout().addWidget(self.pb_epochs)
        progress_widget.layout().addWidget(self.pb_steps)
        self.layout().addWidget(progress_widget)

        # train button
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton("Train", self)
        self.zero_model_button = QPushButton('', self)
        self.zero_model_button.setEnabled(False)

        train_buttons.layout().addWidget(self.zero_model_button)
        train_buttons.layout().addWidget(self.train_button)

        self.layout().addWidget(train_buttons)

        # prediction
        self.pb_pred = QProgressBar()
        self.pb_pred.setValue(0)
        self.pb_pred.setMinimum(0)
        self.pb_pred.setMaximum(100)
        self.pb_pred.setTextVisible(True)
        self.pb_pred.setFormat(f'Prediction ?/?')
        self.layout().addWidget(self.pb_pred)

        predictions = QWidget()
        predictions.setLayout(QHBoxLayout())
        self.predict_button = QPushButton('', self)
        self.predict_button.setEnabled(False)

        predictions.layout().addWidget(QLabel(''))
        predictions.layout().addWidget(self.predict_button)

        self.layout().addWidget(predictions)

        # save button
        save_widget = QWidget()
        save_widget.setLayout(QHBoxLayout())

        self.save_choice = QComboBox()
        self.save_choice.addItems(SaveMode.list())

        self.save_button = QPushButton("Save model", self)
        self.save_button.setEnabled(False)

        save_widget.layout().addWidget(self.save_button)
        save_widget.layout().addWidget(self.save_choice)
        self.layout().addWidget(save_widget)

        # plot widget
        self.plot = TBPlotWidget(max_width=300, min_height=200)
        self.layout().addWidget(self.plot.native)

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

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.enable_3d.use3d.changed.connect(self._update_3D)
        self.img_train.changed.connect(self._update_layer_axes)
        self.train_images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.train_button.clicked.connect(lambda: self._start_training(self.model))
        self.zero_model_button.clicked.connect(self._zero_model)
        self.n_epochs_spin.valueChanged.connect(self._update_epochs)
        self.n_steps_spin.valueChanged.connect(self._update_steps)
        self.predict_button.clicked.connect(self._start_prediction)
        self.save_button.clicked.connect(self._save_model)

        # update axes widget in case of data
        self._update_layer_axes()

    def _start_training(self,  pretrained_model=None):
        if self.state == State.IDLE:

            if self.axes_widget.is_valid():
                self.state = State.RUNNING

                # register which data tab: layers or disk
                self.load_from_disk = self.tabs.currentIndex() == 1

                # modify UI
                self.plot.clear_plot()
                self.train_button.setText('Stop')
                self.zero_model_button.setText('')
                self.zero_model_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.predict_button.setEnabled(False)
                self.predict_button.setText('')

                self.train_worker = train_worker(self, pretrained_model=pretrained_model)
                self.train_worker.yielded.connect(lambda x: self._update_all(x))
                self.train_worker.returned.connect(self._training_done)
                self.train_worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _start_prediction(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.pb_pred.setValue(0)

            # prepare layers name and remove them if they exist
            pred_train_name = self.img_train.name + PREDICT
            pred_val_name = self.img_val.name + PREDICT
            if pred_train_name in self.viewer.layers:
                self.viewer.layers.remove(pred_train_name)
            if pred_val_name in self.viewer.layers:
                self.viewer.layers.remove(pred_val_name)

            # place-holders
            # TODO doesn't work if list!
            final_shape, _, _ = get_shape_order(self.x_train.shape, NAPARI_AXES, self.new_axes)  # napari axes end with YX
            self.pred_train = np.zeros(final_shape, dtype=np.float32).squeeze()
            self.viewer.add_image(self.pred_train, name=pred_train_name, visible=True)
            self.pred_count = final_shape[0]

            if self.x_val is not None:
                final_shape_val, _, _ = get_shape_order(self.x_val.shape, NAPARI_AXES, self.new_axes)
                self.pred_val = np.zeros(final_shape_val, dtype=np.float32).squeeze()
                self.viewer.add_image(self.pred_val, name=pred_val_name, visible=True)
                self.pred_count += final_shape_val[0]

            self.predict_worker = prediction_after_training_worker(self)
            self.predict_worker.yielded.connect(lambda x: self._update_prediction(x))
            self.predict_worker.start()

    def _training_done(self):
        self.state = State.IDLE
        self.train_button.setText('Train again')
        self.zero_model_button.setText('Zero model')
        self.zero_model_button.setEnabled(True)

        self.save_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.predict_button.setText('Predict')

    def _prediction_done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

    def _update_prediction(self, update):
        if self.state == State.RUNNING:
            if update == UpdateType.DONE:
                self._prediction_done()
            else:
                val = update[UpdateType.PRED]
                p_perc = int(100 * val / self.pred_count + 0.5)
                self.pb_pred.setValue(p_perc)
                self.pb_pred.setFormat(f'Prediction {val}/{self.pred_count}')

            self.viewer.layers[self.img_train.name + PREDICT].refresh()
            if self.img_train.value != self.img_val.value:
                self.viewer.layers[self.img_val.name + PREDICT].refresh()

    def _zero_model(self):
        """
        Zero the model, causing the next training session to train from scratch.
        :return:
        """
        if self.state == State.IDLE:
            self.model = None
            self.zero_model_button.setText('')
            self.zero_model_button.setEnabled(False)

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

        if path is not None or path != '':
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
            self.pb_epochs.setFormat(f'Epoch ?/{self.n_epochs_spin.value()}')

    def _update_steps(self):
        if self.state == State.IDLE:
            self.n_steps = self.n_steps_spin.value()
            self.pb_steps.setValue(0)
            self.pb_steps.setFormat(f'Step ?/{self.n_steps_spin.value()}')

    def _update_all(self, updates):
        if self.state == State.RUNNING:
            if UpdateType.EPOCH in updates:
                val = updates[UpdateType.EPOCH]
                e_perc = int(100 * updates[UpdateType.EPOCH] / self.n_epochs + 0.5)
                self.pb_epochs.setValue(e_perc)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if UpdateType.BATCH in updates:
                val = updates[UpdateType.BATCH]
                s_perc = int(100 * val / self.n_steps + 0.5)
                self.pb_steps.setValue(s_perc)
                self.pb_steps.setFormat(f'Step {val}/{self.n_steps}')

            if UpdateType.LOSS in updates:
                self.plot.update_plot(*updates[UpdateType.LOSS])

    def _save_model(self):
        if self.state == State.IDLE:
            if self.model:
                where = QFileDialog.getSaveFileName(caption='Save model')[0]

                export_type = self.save_choice.currentText()
                if SaveMode.MODELZOO.value == export_type:
                    from bioimageio.core.build_spec import build_model

                    axes = self.new_axes
                    axes = axes.replace('S', 'b').lower()

                    build_modelzoo(where + '.bioimage.io.zip',
                                   self.model.logdir / "weights_best.h5",
                                   self.inputs,
                                   self.outputs,
                                   self.tf_version,
                                   axes)

                else:
                    self.model.keras_model.save_weights(where + '.h5')


if __name__ == "__main__":
    from napari_n2v._sample_data import n2v_2D_data, n2v_3D_data

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(TrainWidget(viewer))

    dims = '2D'  # 2D, 3D
    if dims == '2D':
        data = n2v_2D_data()

        # add images
        viewer.add_image(data[0][0][0:50], name=data[0][1]['name'])
        viewer.add_image(data[1][0][0:50], name=data[1][1]['name'])
    else:
        data = n2v_3D_data()

        viewer.add_image(data[0][0], name=data[0][1]['name'])

    napari.run()
