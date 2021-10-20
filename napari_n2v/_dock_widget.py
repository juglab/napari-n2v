"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import pathlib

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory(patch_shape={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, "value": 64},
               neighborhood_radius={"widget_type": "Slider", "min": 1, "max": 16, "value": 5})
def example_magic_widget(training_image: "napari.layers.Image",
                         validation_image: "napari.layers.Image", number_of_epochs: int = 200,
                         number_of_steps: int = 100, batch_size: int = 16, patch_shape=64,
                         neighborhood_radius=1):
    # understood magicgui, thanks documentation.
    # N2V code execution be here
    # add graphs, progressbar(s)
    # create image layer with result on training end
    print(f"you have selected {training_image}", number_of_epochs, number_of_steps, batch_size, patch_shape, neighborhood_radius)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [ExampleQWidget, example_magic_widget]
