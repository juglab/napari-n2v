from __future__ import annotations

import os
import urllib
import zipfile
import numpy as np

from napari.types import LayerDataTuple


def _load_3D():
    from skimage import io

    zipPath = 'data/flywing-data.zip'
    if not os.path.exists(zipPath):
        # download and unzip data
        data = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/flywing-data.zip', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall('data')

    Train_img = io.imread('data/flywing.tif')

    return [(Train_img, {'name': 'Train'})]


def _load_2D():
    # check if data has been downloaded already
    zipPath = "data/BSD68_reproducibility.zip"
    if not os.path.exists(zipPath):
        # download and unzip data
        data = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/BSD68_reproducibility.zip', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("data")

    Train_img = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
    Val_img = np.load('data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy')

    return [(Train_img, {'name': 'Train'}), (Val_img, {'name': 'Val'})]


def _n2v_data(dim):
    assert dim in [2, 3]

    # create a folder for our data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    if dim == 2:
        return _load_2D()
    else:
        return _load_3D()


def n2v_3D_data() -> LayerDataTuple:
    return _n2v_data(3)


def n2v_2D_data() -> LayerDataTuple:
    return _n2v_data(2)
