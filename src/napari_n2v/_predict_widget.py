"""
"""
import napari
import numpy as np
from napari_n2v.utils import State, Updates, DENOISING, prediction_worker
from napari_n2v.widgets import get_load_button, layer_choice_widget, get_threshold_spin
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QProgressBar,
    QCheckBox
)


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

    def _update(self, updates):
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

    def _start_prediction(self):
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

    def _done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')


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
