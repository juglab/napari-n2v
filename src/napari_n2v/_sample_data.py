import urllib
import zipfile
from pathlib import Path

import numpy as np
from skimage import io

from napari.types import LayerDataTuple
from napari.utils import notifications as ntf

from napari_n2v.utils import cwd, get_default_path

# todo the logic is the same for all functions, possibility to refactor


def _load_3D():
    from skimage import io

    with cwd(get_default_path()):
        data_path = Path('data', 'flywing')
        if not data_path.exists():
            data_path.mkdir(parents=True)

        # check if data has been downloaded already
        zip_path = Path(data_path, 'flywing-data.zip')
        if not zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/n2v/flywing-data.zip', zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        im_path = Path(data_path, 'flywing.tif')
        Train_img = io.imread(im_path)

        return [(Train_img, {'name': 'Train'})]


def _load_2D():
    with cwd(get_default_path()):
        data_path = Path('data')
        if not data_path.exists():
            data_path.mkdir()

        # check if data has been downloaded already
        zip_path = Path(data_path, 'BSD68_reproducibility.zip')
        if not zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/n2v/BSD68_reproducibility.zip', zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        train_path = Path(data_path, 'BSD68_reproducibility_data', 'train', 'DCNN400_train_gaussian25.npy')
        val_path = Path(data_path, 'BSD68_reproducibility_data', 'val', 'DCNN400_validation_gaussian25.npy')

        Train_img = np.load(str(train_path))
        Val_img = np.load(str(val_path))

        return [(Train_img, {'name': 'Train'}), (Val_img, {'name': 'Val'})]


def _load_rgb():
    with cwd(get_default_path()):
        data_path = Path('data', 'RGB')
        if not data_path.exists():
            data_path.mkdir(parents=True)

        # check if data has been downloaded already
        zipPath = Path(data_path, 'RGB.zip')
        if not zipPath.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/n2v/RGB.zip', zipPath)
            with zipfile.ZipFile(zipPath, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        img_path = Path(data_path, 'longBeach.png')
        img = io.imread(img_path)[..., :3]  # remove alpha channel

        return [(img, {'name': 'RGB'})]


def _load_sem():
    with cwd(get_default_path()):
        # create data folder if it doesn't already exist
        data_path = Path('data', 'sem')
        if not data_path.exists():
            data_path.mkdir(parents=True)

        # download sem data
        img_zip_path = Path(data_path, 'SEM.zip')
        if not img_zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/n2v/SEM.zip', img_zip_path)
            with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)

        # load data

        # load data
        train_path = Path(data_path, 'train.tif')
        val_path = Path(data_path, 'validation.tif')
        img_train = io.imread(train_path)
        img_val = io.imread(val_path)

        return [(img_train, {'name': 'train'}), (img_val, {'name': 'val'})]


def demo_files():
    with cwd(get_default_path()):
        # load sem validation
        img = _load_sem()[1][0]

        # create models folder if it doesn't already exist
        model_path = Path('models', 'trained_sem_N2V2').absolute()
        if not model_path.exists():
            model_path.mkdir(parents=True)

        # download sem model
        model_zip_path = Path(model_path, 'trained_sem_N2V2.zip')
        if not model_zip_path.exists():
            # download and unzip data
            urllib.request.urlretrieve('https://download.fht.org/jug/napari/trained_sem_N2V2.zip', model_zip_path)
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)

        return img, Path(model_path, 'sem_N2V2.h5')


def _n2v_data(dim):
    assert dim in [2, 3]

    if dim == 2:
        return _load_2D()
    else:
        return _load_3D()


def n2v_3D_data() -> LayerDataTuple:
    ntf.show_info('Downloading data might take a few minutes.')
    return _n2v_data(3)


def n2v_2D_data() -> LayerDataTuple:
    ntf.show_info('Downloading data might take a few minutes.')
    return _n2v_data(2)


def n2v_rgb_data() -> LayerDataTuple:
    ntf.show_info('Downloading data might take a few minutes.')
    return _load_rgb()


def n2v_sem_data() -> LayerDataTuple:
    ntf.show_info('Downloading data might take a few minutes.')
    return _load_sem()
