"""
"""
import os.path

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
    QCheckBox
)
from napari_n2v.widgets import TBPlotWidget, create_choice_widget
from napari_n2v.utils import State, SaveMode, Updates


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setMaximumWidth(350)

        # layer choice widgets
        self.layer_choice = create_choice_widget(napari_viewer)
        self.img_train = self.layer_choice.Train
        self.img_val = self.layer_choice.Val
        self.layout().addWidget(self.layer_choice.native)

        # 3D checkbox
        self.checkbox_3d = QCheckBox('3D')

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
        self.batch_size_spin.setSingleStep(8)
        self.batch_size_spin.setValue(16)

        # patch XY size
        self.patch_XY_spin = QSpinBox()
        self.patch_XY_spin.setMaximum(512)
        self.patch_XY_spin.setMinimum(8)
        self.patch_XY_spin.setSingleStep(8)
        self.patch_XY_spin.setValue(64)

        # patch Z size
        self.patch_Z_spin = QSpinBox()
        self.patch_Z_spin.setMaximum(512)
        self.patch_Z_spin.setMinimum(4)
        self.patch_Z_spin.setSingleStep(8)
        self.patch_Z_spin.setValue(32)
        self.patch_Z_spin.setEnabled(False)
        self.patch_Z_spin.setVisible(False)

        # add widgets
        # TODO add tooltips
        others = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow('', self.checkbox_3d)
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

        self.pb_pred = QProgressBar()
        self.pb_pred.setValue(0)
        self.pb_pred.setMinimum(0)
        self.pb_pred.setMaximum(100)
        self.pb_pred.setTextVisible(True)
        self.pb_pred.setFormat(f'Prediction ?/?')

        progress_widget.layout().addWidget(self.pb_epochs)
        progress_widget.layout().addWidget(self.pb_steps)
        self.layout().addWidget(progress_widget)

        # train button
        train_buttons = QWidget()
        train_buttons.setLayout(QHBoxLayout())

        self.train_button = QPushButton("Train", self)
        self.retrain_button = QPushButton("", self)
        self.retrain_button.setEnabled(False)

        train_buttons.layout().addWidget(self.retrain_button)
        train_buttons.layout().addWidget(self.train_button)

        self.layout().addWidget(train_buttons)

        # prediction button
        self.layout().addWidget(self.pb_pred)
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setEnabled(False)
        self.layout().addWidget(self.predict_button)

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
        self.plot = TBPlotWidget(max_width=300, min_height=150)
        self.layout().addWidget(self.plot.native)

        # actions
        self.n_epochs_spin.valueChanged.connect(self.update_epochs)
        self.n_steps_spin.valueChanged.connect(self.update_steps)
        self.checkbox_3d.stateChanged.connect(self.update_patch)

        # this allows stopping the thread when the napari window is closed,
        # including reducing the risk that an update comes after closing the
        # window and appearing as a new Qt view. But the call to qt_viewer
        # will be deprecated. Hopefully until then an on_window_closing event
        # will be available.
        # napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

        # place-holders for the trained model, prediction and parameters (bioimage.io)
        self.model, self.pred_train, self.pred_val = None, None, None
        self.inputs, self.outputs = None, None
        self.tf_version = None
        self.train_worker = None
        self.predict_worker = None
        self.pred_count = 0
        self.weights_path = ''

        # button and worker actions
        self.train_button.clicked.connect(self.start_training)
        self.retrain_button.clicked.connect(self.continue_training)
        self.predict_button.clicked.connect(self.start_prediction)
        self.save_button.clicked.connect(self.save_model)

    def interrupt(self):
        if self.train_worker:
            self.train_worker.quit()

    def start_training(self,  pretrained_model=None):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.plot.clear_plot()
            self.train_button.setText('Stop')

            self.save_button.setEnabled(False)
            self.predict_button.setEnabled(False)

            self.train_worker = train_worker(self, pretrained_model=pretrained_model)
            self.train_worker.yielded.connect(lambda x: self.update_all(x))
            self.train_worker.returned.connect(self.training_done)
            self.train_worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def continue_training(self):
        if self.state == State.IDLE:
            self.start_training(pretrained_model=self.model)

    def start_prediction(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING
            self.pb_pred.setValue(0)

            # set up the layers for the denoised predictions
            pred_train_name = self.img_train.name + PREDICT
            if pred_train_name in self.viewer.layers:
                self.viewer.layers.remove(pred_train_name)
            self.pred_train = np.zeros(self.img_train.value.data.shape, dtype=np.int16)
            self.viewer.add_labels(self.pred_train, name=pred_train_name, visible=True)

            pred_val_name = self.img_val.name + PREDICT
            if self.img_train.value != self.img_val.value:
                if pred_val_name in self.viewer.layers:
                    self.viewer.layers.remove(pred_val_name)
                self.pred_val = np.zeros(self.img_val.value.data.shape, dtype=np.int16)
                self.viewer.add_labels(self.pred_val, name=pred_val_name, visible=True)

            if self.checkbox_3d.isChecked():
                self.pred_count = 1
            else:
                self.pred_count = self.img_train.value.data.shape[0]

            # check if there is validation data and add it to the prediction count
            if self.img_train.value != self.img_val.value:
                if self.checkbox_3d.isChecked():
                    self.pred_count += 1
                else:
                    self.pred_count += self.img_val.value.data.shape[0]

            self.predict_worker = predict_worker(self)
            self.predict_worker.yielded.connect(lambda x: self.update_predict(x))
            self.predict_worker.start()

    def _training_done(self):
        self.state = State.IDLE
        self.train_button.setText('Train new')
        self.retrain_button.setText('Retrain')
        self.retrain_button.setEnabled(True)

        self.save_button.setEnabled(True)
        self.predict_button.setEnabled(True)

    def _prediction_done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

    def _update_predict(self, update):
        if self.state == State.RUNNING:
            if update == Updates.DONE:
                self.prediction_done()
            else:
                val = update[Updates.PRED]
                p_perc = int(100 * val / self.pred_count + 0.5)
                self.pb_pred.setValue(p_perc)
                self.pb_pred.setFormat(f'Prediction {val}/{self.pred_count}')

            self.viewer.layers[self.img_train.name + PREDICT].refresh()
            if self.img_train.value != self.img_val.value:
                self.viewer.layers[self.img_val.name + PREDICT].refresh()

    def _update_patch(self):
        if self.checkbox_3d.isChecked():
            self.patch_Z_spin.setEnabled(True)
            self.patch_Z_spin.setVisible(True)
        else:
            self.patch_Z_spin.setEnabled(False)
            self.patch_Z_spin.setVisible(False)

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
            if Updates.EPOCH in updates:
                val = updates[Updates.EPOCH]
                e_perc = int(100 * updates[Updates.EPOCH] / self.n_epochs + 0.5)
                self.pb_epochs.setValue(e_perc)
                self.pb_epochs.setFormat(f'Epoch {val}/{self.n_epochs}')

            if Updates.BATCH in updates:
                val = updates[Updates.BATCH]
                s_perc = int(100 * val / self.n_steps + 0.5)
                self.pb_steps.setValue(s_perc)
                self.pb_steps.setFormat(f'Step {val}/{self.n_steps}')

            if Updates.LOSS in updates:
                self.plot.update_plot(*updates[Updates.LOSS])

    def _save_model(self):
        if self.state == State.IDLE:
            if self.model:
                where = QFileDialog.getSaveFileName(caption='Save model')[0]

                if self.checkbox_3d.isChecked():
                    axes = 'bzyxc'
                    dimensions = '3d'
                else:
                    axes = 'byxc'
                    dimensions = '2d'

                export_type = self.save_choice.currentText()
                if SaveMode.MODELZOO.value == export_type:
                    from bioimageio.core.build_spec import build_model

                    build_model(
                        weight_uri=self.weights_path,
                        test_inputs=[self.inputs],
                        test_outputs=[self.outputs],
                        input_axes=[axes],
                        output_axes=[axes],
                        output_path=where + '.bioimage.io.zip',
                        name='Noise2Void',
                        description='Self-supervised denoising.',
                        authors=[{'name': "Tim-Oliver Buchholz"}, {'name': "Alexander Krull"}, {'name': "Florian Jug"}],
                        license="BSD-3-Clause",
                        documentation=os.path.abspath('../resources/documentation.md'),
                        tags=[dimensions, 'tensorflow', 'unet', 'denoising'],
                        cite=[{'text': 'Noise2Void - Learning Denoising from Single Noisy Images',
                               'doi': "10.48550/arXiv.1811.10980"}],
                        preprocessing=[],
                        postprocessing=[],
                        tensorflow_version=self.tf_version
                    )
                else:
                    self.model.keras_model.save_weights(where + '.h5')




if __name__ == "__main__":
    from napari_n2v._sample_data import n2v_2D_data, n2v_3D_data

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(TrainWidget(viewer))

    dims = '3D'  # 2D, 3D
    if dims == '2D':
        data = n2v_2D_data()

        # add images
        viewer.add_image(data[0][0][0:50], name=data[0][1]['name'])
        viewer.add_image(data[1][0][0:50], name=data[1][1]['name'])
    else:
        data = n2v_3D_data()

        viewer.add_image(data[0][0], name=data[0][1]['name'])

    napari.run()
