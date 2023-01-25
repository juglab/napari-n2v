"""
"""
from pathlib import Path
import napari
from napari.utils import notifications as ntf

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QTabWidget,
    QGroupBox,
    QLabel,
    QFormLayout
)

from napari_n2v.resources import ICON_JUGLAB
from napari_n2v.utils import (
    State,
    UpdateType,
    DENOISING,
    prediction_worker,
    loading_worker
)
from napari_n2v.widgets import (
    AxesWidget,
    FolderWidget,
    load_button,
    layer_choice,
    ScrollWidgetWrapper,
    BannerWidget,
    create_gpu_label,
    create_progressbar,
    create_int_spinbox
)


SAMPLE = 'Sample data'


class PredictWidgetWrapper(ScrollWidgetWrapper):
    def __init__(self, napari_viewer):
        self.widget = PredictWidget(napari_viewer)

        super().__init__(self.widget)


class PredictWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.state = State.IDLE
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(200)
        self.setMaximumHeight(720)

        ###############################
        # add banner
        self.layout().addWidget(BannerWidget('N2V - Prediction',
                                             ICON_JUGLAB,
                                             'A self-supervised denoising algorithm.',
                                             'https://juglab.github.io/napari-n2v/',
                                             'https://github.com/juglab/napari-n2v/issues'))

        # add GPU button
        gpu_button = create_gpu_label()
        gpu_button.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(gpu_button)

        # QTabs
        self.tabs = QTabWidget()
        tab_layers = QWidget()
        tab_layers.setLayout(QVBoxLayout())

        tab_disk = QWidget()
        tab_disk.setLayout(QVBoxLayout())

        # add tabs
        self.tabs.addTab(tab_layers, 'From layers')
        self.tabs.addTab(tab_disk, 'From disk')
        self.tabs.setMaximumHeight(120)

        # image layer tab
        self.images = layer_choice(annotation=napari.layers.Image, name="Images")
        tab_layers.layout().addWidget(self.images.native)

        # disk tab
        self.lazy_loading = QCheckBox('Lazy loading')
        tab_disk.layout().addWidget(self.lazy_loading)
        self.images_folder = FolderWidget('Choose')
        tab_disk.layout().addWidget(self.images_folder)

        # add to main layout
        self.layout().addWidget(self.tabs)

        ###############################
        self._build_params_widgets()
        self._build_tiling_widgets()
        self._build_predict_widgets()

        # place holders
        self.worker = None
        self.denoi_prediction = None
        self.sample_image = None
        self.n_im = 0
        self.load_from_disk = False
        self.scale = None

        # actions
        self.tabs.currentChanged.connect(self._update_tab_axes)
        self.predict_button.clicked.connect(self._start_prediction)
        self.images.changed.connect(self._update_layer_axes)
        self.images_folder.text_field.textChanged.connect(self._update_disk_axes)
        self.enable_3d.stateChanged.connect(self._update_3D)
        self.tiling_cbox.stateChanged.connect(self._update_tiling)

        # update image layer
        self.images.choices = [x for x in self.viewer.layers if type(x) is napari.layers.Image]

        # update axes if necessary
        self._update_layer_axes()

    def _build_params_widgets(self):
        self.params_group = QGroupBox()
        self.params_group.setTitle("Parameters")
        self.params_group.setLayout(QVBoxLayout())
        self.params_group.layout().setContentsMargins(20, 20, 20, 0)
        # load model button
        self.load_model_button = load_button()
        self.params_group.layout().addWidget(self.load_model_button.native)
        # load 3D enabling checkbox
        self.enable_3d = QCheckBox('Enable 3D')
        self.params_group.layout().addWidget(self.enable_3d)
        # axes widget
        self.axes_widget = AxesWidget()
        self.params_group.layout().addWidget(self.axes_widget)
        self.layout().addWidget(self.params_group)

    def _build_tiling_widgets(self):
        # tiling
        self.tilling_group = QGroupBox()
        self.tilling_group.setTitle("Tiling (optional)")
        self.tilling_group.setLayout(QVBoxLayout())
        self.tilling_group.layout().setContentsMargins(20, 20, 20, 0)

        # checkbox
        self.tiling_cbox = QCheckBox('Tile prediction')
        self.tiling_cbox.setToolTip('Select to predict the image by tiles')
        self.tilling_group.layout().addWidget(self.tiling_cbox)

        # tiling spinbox
        self.tiling_spin = create_int_spinbox(1, 1000, 4, tooltip='Minimum number of tiles to use')
        self.tiling_spin.setEnabled(False)

        tiling_form = QFormLayout()
        tiling_form.addRow('Number of tiles', self.tiling_spin)
        tiling_widget = QWidget()
        tiling_widget.setLayout(tiling_form)
        self.tilling_group.layout().addWidget(tiling_widget)

        self.layout().addWidget(self.tilling_group)

    def _build_predict_widgets(self):
        self.predict_group = QGroupBox()
        self.predict_group.setTitle("Prediction")
        self.predict_group.setLayout(QVBoxLayout())
        self.predict_group.layout().setContentsMargins(20, 20, 20, 0)

        # prediction progress bar
        self.pb_prediction = create_progressbar(text_format=f'Prediction ?/?')
        self.pb_prediction.setToolTip('Show the progress of the prediction')

        # predict button
        predictions = QWidget()
        predictions.setLayout(QHBoxLayout())
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.setEnabled(True)
        self.predict_button.setToolTip('Run the trained model on the images')

        predictions.layout().addWidget(QLabel(''))
        predictions.layout().addWidget(self.predict_button)

        # add to the group
        self.predict_group.layout().addWidget(self.pb_prediction)
        self.predict_group.layout().addWidget(predictions)
        self.layout().addWidget(self.predict_group)

    def _update_tiling(self, state):
        self.tiling_spin.setEnabled(state)

    def _update_3D(self):
        self.axes_widget.update_is_3D(self.enable_3d.isChecked())
        self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_layer_axes(self):
        if self.images.value is not None:
            shape = self.images.value.data.shape

            # update shape length in the axes widget
            self.axes_widget.update_axes_number(len(shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _add_image(self, image):
        if SAMPLE in self.viewer.layers:
            self.viewer.layers.remove(SAMPLE)

        if image is not None:
            self.viewer.add_image(image, name=SAMPLE, visible=True)
            self.sample_image = image

            # update the axes widget
            self.axes_widget.update_axes_number(len(image.shape))
            self.axes_widget.set_text_field(self.axes_widget.get_default_text())

    def _update_disk_axes(self):
        path = self.images_folder.get_folder()

        # load one image
        load_worker = loading_worker(path)
        load_worker.yielded.connect(lambda x: self._add_image(x))
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

    def _update(self, updates):
        if UpdateType.N_IMAGES in updates:
            self.n_im = updates[UpdateType.N_IMAGES]
            self.pb_prediction.setValue(0)
            self.pb_prediction.setFormat(f'Prediction 0/{self.n_im}')

        if UpdateType.IMAGE in updates:
            val = updates[UpdateType.IMAGE]
            perc = int(100 * val / self.n_im + 0.5)
            self.pb_prediction.setValue(perc)
            self.pb_prediction.setFormat(f'Prediction {val}/{self.n_im}')
            # self.viewer.layers[DENOISING].refresh()

        if UpdateType.DONE in updates:
            self.pb_prediction.setValue(100)
            self.pb_prediction.setFormat(f'Prediction done')

    def _start_prediction(self):
        if self.state == State.IDLE:
            if self.axes_widget.is_valid():
                if self.get_model_path().exists() and self.get_model_path().is_file():
                    self.state = State.RUNNING

                    self.predict_button.setText('Stop')

                    if DENOISING in self.viewer.layers:
                        self.viewer.layers.remove(DENOISING)

                    self.denoi_prediction = None
                    self.worker = prediction_worker(self)
                    self.worker.yielded.connect(lambda x: self._update(x))
                    self.worker.returned.connect(self._done)
                    self.worker.start()
                else:
                    # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                    # ntf.show_error('Select a valid model path')
                    ntf.show_info('Select a valid model path')
            else:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error('Invalid axes')
                ntf.show_info('Invalid axes')

        elif self.state == State.RUNNING:
            self.state = State.IDLE

    def _done(self):
        self.state = State.IDLE
        self.predict_button.setText('Predict again')

        if self.denoi_prediction is not None:
            if self.scale is not None:
                self.viewer.add_image(
                    self.denoi_prediction,
                    name=DENOISING,
                    scale = self.scale,
                    visible=True
                )
            else:
                self.viewer.add_image(self.denoi_prediction, name=DENOISING, visible=True)

    def get_model_path(self):
        return self.load_model_button.Model.value

    def set_model_path(self, path: Path):
        self.load_model_button.Model.value = path

    def set_layer(self, layer):
        self.images.choices = [x for x in self.viewer.layers if type(x) is napari.layers.Image]
        if layer in self.images.choices:
            self.images.native.value = layer

    # TODO call these methods throughout the workers
    def get_axes(self):
        return self.axes_widget.get_axes()

    def is_tiling_checked(self):
        return self.tiling_cbox.isChecked()

    def get_n_tiles(self):
        return self.tiling_spin.value()


class DemoPrediction(PredictWidgetWrapper):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        # dowload demo files
        from napari_n2v._sample_data import demo_files
        ntf.show_info('Downloading data can take a few minutes.')

        # get files
        img, model = demo_files()

        # add image to viewer
        name = 'Demo image'
        napari_viewer.add_image(img[0:471, 200:671], name=name)

        # modify path
        self.widget.set_model_path(model)
        self.widget.set_layer(name)


if __name__ == "__main__":
    from napari_n2v._sample_data import n2v_2D_data, n2v_3D_data

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(PredictWidgetWrapper(viewer))

    # add images
    dim = '2D'
    if dim == '2D':
        data = n2v_2D_data()
        viewer.add_image(data[0][0][-10:], name=data[0][1]['name'])
    else:
        data = n2v_3D_data()
        viewer.add_image(data[0][0][4:20, 150:378, 150:378], name=data[0][1]['name'])
        # viewer.add_image(data[0][0], name=data[0][1]['name'])

    napari.run()
