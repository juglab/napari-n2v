"""
"""
from pathlib import Path

import bioimageio.core
import napari
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from magicgui.widgets import create_widget
import numpy as np
from napari_n2v._train_widget import State, prepare_data, create_model
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QProgressBar,
    QCheckBox
)
from enum import Enum

DENOISING = 'denoised'


@magic_factory(auto_call=True,
               Threshold={"widget_type": "FloatSpinBox", "min": 0, "max": 1., "step": 0.1, 'value': 0.6})
def get_threshold_spin(Threshold: int):
    pass


@magic_factory(auto_call=True, Model={'mode': 'r', 'filter': '*.h5 *.zip'})
def get_load_button(Model: Path):
    pass


def layer_choice_widget(np_viewer, annotation, **kwargs):
    widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()
    np_viewer.layers.events.inserted.connect(widget.reset_choices)
    np_viewer.layers.events.removed.connect(widget.reset_choices)
    return widget


class Updates(Enum):
    N_IMAGES = 'number of images'
    IMAGE = 'image'
    DONE = 'done'


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # load model button
        self.load_button = get_load_button()
        self.layout().addWidget(self.load_button.native)

        # image layer
        self.images = layer_choice_widget(napari_viewer, annotation=napari.layers.Image, name="Images")
        self.layout().addWidget(self.images.native)

        # 3D checkbox
        self.checkbox_3d = QCheckBox('3D')
        self.layout().addWidget(self.checkbox_3d)

        # threshold slider
        self.threshold_spin = get_threshold_spin()
        self.layout().addWidget(self.threshold_spin.native)

        # progress bar
        self.pb_prediction = QProgressBar()
        self.pb_prediction.setValue(0)
        self.pb_prediction.setMinimum(0)
        self.pb_prediction.setMaximum(100)
        self.pb_prediction.setTextVisible(True)
        self.pb_prediction.setFormat(f'Images ?/?')
        self.layout().addWidget(self.pb_prediction)

        # predict button
        self.worker = None
        self.denoi_prediction = None
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.start_prediction)
        self.layout().addWidget(self.predict_button)

        self.n_im = 0

        # napari_viewer.window.qt_viewer.destroyed.connect(self.interrupt)

    def update(self, updates):
        if Updates.N_IMAGES in updates:
            self.n_im = updates[Updates.N_IMAGES]
            self.pb_prediction.setValue(0)
            self.pb_prediction.setFormat(f'Prediction 0/{self.n_im}')

        if Updates.IMAGE in updates:
            val = updates[Updates.IMAGE]
            perc = int(100 * val / self.n_im + 0.5)
            self.pb_prediction.setValue(perc)
            self.pb_prediction.setFormat(f'Prediction {val}/{self.n_im}')
            self.viewer.layers[DENOISING].refresh()

        if Updates.DONE in updates:
            self.pb_prediction.setValue(100)
            self.pb_prediction.setFormat(f'Prediction done')

    def interrupt(self):
        self.worker.quit()

    def start_prediction(self):
        if self.state == State.IDLE:
            self.state = State.RUNNING

            self.predict_button.setText('Stop')

            if DENOISING in self.viewer.layers:
                self.viewer.layers.remove(DENOISING)

            self.denoi_prediction = np.zeros(self.images.value.data.shape, dtype=np.int16)
            viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)

            self.worker = prediction_worker(self)
            self.worker.yielded.connect(lambda x: self.update(x))
            self.worker.returned.connect(self.done)
            self.worker.start()
        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')


@thread_worker(start_thread=False)
def prediction_worker(widget: PredictWidget):
    import os
    import threading

    # TODO remove (just used because I currently cannot use the GPU)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    # get images
    images = widget.images.value.data

    # get other parameters
    n_epochs = widget.n_epochs
    n_steps = widget.n_steps
    batch_size = widget.batch_size_spin.value()
    patch_XY = widget.patch_XY_spin.value()
    patch_Z = widget.patch_Z_spin.value()

    # patch shape
    is_3d = widget.checkbox_3d.isChecked()
    if is_3d:
        patch_shape = (patch_Z, patch_XY, patch_XY)
    else:
        patch_shape = (patch_XY, patch_XY)

    # prepare data
    X_train, X_val = prepare_data(images, None, patch_shape)

    # create model
    if is_3d:
        model_name = 'n2v_3D'
    else:
        model_name = 'n2v_2D'
    base_dir = 'models'
    model = create_model(X_train, n_epochs, n_steps, batch_size, model_name, base_dir, None)
    widget.weights_path = os.path.join(base_dir, model_name, 'weights_best.h5')

    # TODO: predict images and yield progress to the progress bar


if __name__ == "__main__":
    from napari_n2v._sample_data import n2v_2D_data, n2v_3D_data

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(PredictWidget(viewer))

    # add images
    dim = '2D'
    if dim == '2D':
        data = n2v_2D_data()
        viewer.add_image(data[0][0][0:30], name=data[0][1]['name'])
    else:
        data = n2v_3D_data()
        viewer.add_image(data[0][0], name=data[0][1]['name'])

    napari.run()
