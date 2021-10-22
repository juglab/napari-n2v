"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import pathlib

import numpy as np
from magicgui.types import FileDialogMode
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V
from napari_plugin_engine import napari_hook_implementation
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



    datagen = N2V_DataGenerator()
    shape = (patch_shape, patch_shape)
    X = training_image.data[np.newaxis, ..., np.newaxis]
    # X = datagen.generate_patches(X.data, shape=shape)
    X_val = validation_image.data[np.newaxis, ..., np.newaxis]
    # X_val = datagen.generate_patches(X_val, shape=shape)
    #TODO: hackaround for the slicing error in N2V, find solution
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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #example_magic_widget.filename.called.connect()

    return [example_magic_widget]
