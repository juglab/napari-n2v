"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import pathlib

import napari
import numpy as np
from magicgui.types import FileDialogMode
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QFormLayout, QGridLayout, QLineEdit, QSpinBox, QComboBox
from qtpy.QtCore import Qt
from magicgui import magic_factory
from napari.layers import Image


@magic_factory(auto_call=False, patch_shape={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, "value": 64},
               neighborhood_radius={"widget_type": "Slider", "min": 1, "max": 16, "value": 5},
               filename={"label": "Save model", "mode": FileDialogMode.OPTIONAL_FILE})
def example_magic_widget(training_image: "napari.layers.Image",
                         validation_image: "napari.layers.Image", number_of_epochs: int = 5,
                         number_of_steps: int = 5, batch_size: int = 16, patch_shape=64,
                         neighborhood_radius=1, filename=pathlib.Path.home()) -> Image:
    # understood magicgui, thanks documentation.
    # N2V code execution be here
    # add graphs, progressbar(s)
    # create image layer with result on training end
    shape = (patch_shape, patch_shape)
    # datagen = N2V_DataGenerator()
    X = training_image.data[np.newaxis, ..., np.newaxis]
    # X = datagen.generate_patches(X.data, shape=shape)
    X_val = validation_image.data[np.newaxis, ..., np.newaxis]
    # X_val = datagen.generate_patches(X_val, shape=shape)
    # TODO: hackaround for the slicing error in N2V, find solution
    X_patches = []
    if X.shape[1] > shape[0] and X.shape[2] > shape[1]:
        for y in range(0, X.shape[1] - shape[0] + 1, shape[0]):
            for x in range(0, X.shape[2] - shape[1] + 1, shape[1]):
                X_patches.append(X[:, y:y + shape[0], x:x + shape[1]])
    X_patches = np.concatenate(X_patches)
    X_val_patches = []
    if X_val.shape[1] > shape[0] and X_val.shape[2] > shape[1]:
        for y in range(0, X_val.shape[1] - shape[0] + 1, shape[0]):
            for x in range(0, X_val.shape[2] - shape[1] + 1, shape[1]):
                X_val_patches.append(X[:, y:y + shape[0], x:x + shape[1]])
    X_val_patches = np.concatenate(X_val_patches)
    config = N2VConfig(X_patches, unet_kern_size=3,
                       train_steps_per_epoch=number_of_steps, train_epochs=number_of_epochs, train_loss='mse',
                       batch_norm=True, train_batch_size=batch_size, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)

    # a name used to identify the model
    model_name = 'n2v_2D'
    # the base directory in which our model will live
    basedir = 'models'
    # We are now creating our network model.
    model = N2V(config, model_name, basedir=basedir)
    history = model.train(X_patches, X_val_patches)
    model.load_weights('weights_best.h5')
    p_ = model.predict(training_image.data.astype(np.float32), 'YX', tta=False)

    print(f"you have selected {training_image}", number_of_epochs, number_of_steps, batch_size, patch_shape,
          neighborhood_radius)
    print(p_)
    return Image(p_, name='prediction result')


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter

    # training_image: "napari.layers.Image",
    # validation_image: "napari.layers.Image", number_of_epochs: int = 5,
    # number_of_steps: int = 5, batch_size: int = 16, patch_shape=64,
    # neighborhood_radius=1, filename=pathlib.Path.home()

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        #self.setLayout(QVBoxLayout())
        layout = QGridLayout()
        layout.columnMinimumWidth(15)
        self.setLayout(layout)
        training_image_selector = QComboBox()
        validation_image_selector = QComboBox()
        for image in image_layers:
            training_image_selector.addItem(image.name, userData=image)
            validation_image_selector.addItem(image.name, userData=image)
        self.layout().addWidget(QLabel("Training Image"), 0, 0, 1, 1)
        self.layout().addWidget(training_image_selector, 0, 2, 2, 1)
        self.layout().addWidget(QLabel("Validation Image"), 1, 0, 1, 1)
        self.layout().addWidget(validation_image_selector, 1, 2, 2, 1)
        GridSpinBoxQWidget(self.viewer, layout, "number of epochs", 1, 500, 1, 200, 2)
        GridSpinBoxQWidget(self.viewer, layout, "number of steps", 1, 500, 1, 100, 3)
        GridSliderQWidget(self.viewer, layout, "patch shape", 16, 512, 16, 16, 4)
        GridSliderQWidget(self.viewer, layout, "neighborhood radius", 1, 64, 1, 5, 5)
        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.layout().addWidget(btn, 6, 0, 3, 1)
        self.layout().setRowStretch(6, 1)
        for i in range(5):
            self.layout().setColumnStretch(i, 1)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


class GridSpinBoxQWidget(QWidget):
    def __init__(self, napari_viewer, layout, label, min=1, max=64, step=1, default_value=1, row=0):
        super().__init__()
        self.viewer = napari_viewer
        spinbox = QSpinBox(self)
        spinbox.setMinimum(min)
        spinbox.setMaximum(max)
        spinbox.setValue(default_value)
        spinbox.setSingleStep(step)

        self.spinbox = spinbox
        label_widget = QLabel(label)
        layout.addWidget(label_widget, row, 0, 1, 1)
        layout.addWidget(spinbox, row, 2, 1, 2)

    def get_value(self):
        return self.spinbox.value()


class GridSliderQWidget(QWidget):
    def __init__(self, napari_viewer, layout, label, min=1, max=64, step=1, default_value=1, row=0):
        super().__init__()
        layout.setRowStretch(row,1)
        self.viewer = napari_viewer
        patch_shape_slider = self.create_slider(min, max, step, default_value)
        label_widget = QLabel(label)
        value_widget = QLineEdit(str(default_value))
        value_widget.setInputMask("ddd")
        value_widget.setMaxLength(3)
        value_widget.setFixedWidth(30)
        self.value_widget = value_widget
        layout.addWidget(label_widget, row, 0, 1, 1)
        layout.addWidget(patch_shape_slider, row, 2, 1, 2)
        layout.addWidget(value_widget, row, 4, 1, 1)

    def create_slider(self, min, max, step, default_value):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(default_value)
        slider.setTickInterval(step)
        slider.setSingleStep(step)
        slider.setPageStep(step)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.valueChanged.connect(self.update_label)
        return slider

    def update_label(self, value):
        self.value_widget.setText(str(value))

    def get_value(self):
        return self.value_widget.text()



@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [ExampleQWidget, example_magic_widget]
