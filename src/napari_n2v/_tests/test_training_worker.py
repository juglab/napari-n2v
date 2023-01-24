from pathlib import Path

import numpy as np
import pytest

from napari_n2v.utils import State
from napari_n2v.utils.training_worker import load_images, prepare_data, train_worker
from napari_n2v._tests.test_utils import save_img


class MonkeyPatchAxesWidget:
    def __init__(self, axes='SYX'):
        self.axes = axes
        # TODO should we make sure that the same AxesWidget constraints are enforced?

    def get_axes(self):
        return self.axes


class MonkeyPatchFolder:
    def __init__(self, path='.'):
        self.path = path

    def get_folder(self):
        return self.path


class MonkeyPatchLayerEntry:
    def __init__(self, data=None, name='Nobody'):
        self.data = data
        self.name = name
        self.scale = [1, 2, 1]


class MonkeyPatchLayer:
    def __init__(self, **kwargs):
        self.value = MonkeyPatchLayerEntry(**kwargs)


class MonkeyPatchWidget:
    def __init__(self,
                 axes,
                 load_from_disk,
                 train,
                 val,
                 is_3D=False,
                 n_epochs=1,
                 n_steps=1,
                 batch_size=1,
                 patch_XY=32,
                 patch_Z=16):
        self.state = State.RUNNING
        self.axes_widget = MonkeyPatchAxesWidget(axes)
        self.load_from_disk = load_from_disk

        if self.load_from_disk:
            self.train_images_folder = MonkeyPatchFolder(train)
            self.val_images_folder = MonkeyPatchFolder(val)
        else:
            # todo: this prevents testing when name_test == name_val
            self.img_train = MonkeyPatchLayer(data=train, name='train')
            self.img_val = MonkeyPatchLayer(data=val, name='val')

        self.is_3D = is_3D
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.patch_XY = patch_XY
        self.patch_Z = patch_Z
        self.scale = None

        self.model = None

    def get_batch_size(self):
        return self.batch_size

    def get_patch_XY(self):
        return self.patch_XY

    def get_patch_Z(self):
        return self.patch_Z


@pytest.mark.qt
@pytest.mark.parametrize('shape, axes, is_3D', [((1, 64, 64), 'SYX', False),
                                                ((1, 64, 64, 3), 'SYXC', False),
                                                ((1, 3, 64, 64), 'STYX', False),
                                                ((1, 32, 64, 64), 'SZYX', True),
                                                ((1, 32, 64, 64, 3), 'SZYXC', True)])
def test_train_layers(qtbot, shape, axes, is_3D):
    """
    Test the training pipelines from "layers".
    """
    # create data
    x = np.concatenate([np.ones(shape), np.zeros(shape)])
    y = np.concatenate([np.ones(shape), np.zeros(shape)])

    # create widget
    widget = MonkeyPatchWidget(axes, False, x, y, is_3D)

    # create worker
    t = train_worker(widget)

    with qtbot.waitSignal(t.finished, timeout=100_000):
        t.start()

    assert widget.model is not None


@pytest.mark.qt
@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('shape, axes, is_3D', [((64, 64), 'YX', False),
                                                ((64, 64, 3), 'YXC', False),
                                                ((3, 64, 64), 'TYX', False),
                                                ((32, 64, 64), 'ZYX', True),
                                                ((32, 64, 64, 3), 'ZYXC', True)])
def test_train_from_disk(qtbot, tmp_path, n, shape, axes, is_3D):
    """
    Test the training pipelines from disk.
    """
    # create data
    train_path = Path(tmp_path, 'train')
    train_path.mkdir()
    val_path = Path(tmp_path, 'vak')
    val_path.mkdir()
    save_img(train_path, n, shape)
    save_img(val_path, n, shape)

    # create widget
    widget = MonkeyPatchWidget(axes, True, train_path, val_path, is_3D)

    # create worker
    t = train_worker(widget)

    with qtbot.waitSignal(t.finished, timeout=100_000):
        t.start()

    assert widget.model is not None


@pytest.mark.qt
@pytest.mark.parametrize('shape1, shape2, axes, is_3D', [((64, 64), (48, 48), 'YX', False),
                                                         ((64, 48), (64, 48), 'YX', False),
                                                         ((64, 64, 3), (48, 48, 3), 'YXC', False),
                                                         ((3, 64, 64), (3, 48, 48), 'TYX', False),
                                                         ((32, 64, 64), (30, 48, 48), 'ZYX', True),
                                                         ((32, 64, 64, 3), (30, 48, 48, 3), 'ZYXC', True)])
def test_train_from_disk_list(qtbot, tmp_path, shape1, shape2, axes, is_3D):
    """
    Test the training pipelines from disk when data is loaded as a list.
    """
    # create data
    train_path = Path(tmp_path, 'train')
    train_path.mkdir()
    val_path = Path(tmp_path, 'vak')
    val_path.mkdir()
    save_img(train_path, 1, shape1)
    save_img(val_path, 1, shape2)
    save_img(train_path, 1, shape1, prefix='2_')
    save_img(val_path, 1, shape2, prefix='2_')

    # create widget
    widget = MonkeyPatchWidget(axes, True, train_path, val_path, is_3D)

    # create worker
    t = train_worker(widget)

    with qtbot.waitSignal(t.finished, timeout=100_000):
        t.start()

    assert widget.model is not None


#############################################################################################
@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('shape, axes, new_axes', [((8, 8), 'YX', 'SYXC'),
                                                   ((8, 8, 8), 'ZYX', 'SZYXC'),
                                                   ((2, 8, 8), 'SYX', 'SYXC'),
                                                   ((8, 8, 3), 'YXC', 'SYXC'),
                                                   ((8, 2, 8), 'YTX', 'SYXC'),
                                                   ((2, 2, 8, 8, 3), 'TSYXC', 'SYXC'),
                                                   ((8, 2, 8, 8, 3), 'ZTYXC', 'SZYXC')])
def test_load_images_from_disk_same_val(tmp_path, n, shape, axes, new_axes):
    """
    Test loading from disk using the same train and val path.
    """
    # create data
    save_img(tmp_path, n, shape)

    # create widget
    widget = MonkeyPatchWidget(axes, True, train=tmp_path, val=tmp_path)

    # find final 'S' dim
    prod = n
    if 'S' in axes:
        prod *= shape[axes.find('S')]
    if 'T' in axes:
        prod *= shape[axes.find('T')]

    _x_train, _x_val, _new_axes = load_images(widget)
    assert len(_x_train.shape) == len(new_axes)
    assert _x_train.shape[0] == prod
    assert _new_axes == new_axes
    assert _x_val is None


@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('shape1, shape2, axes, new_axes', [((8, 8), (16, 16), 'YX', 'SYXC'),
                                                            ((8, 8, 8), (3, 8, 8), 'ZYX', 'SZYXC'),
                                                            ((2, 8, 8), (1, 8, 8), 'SYX', 'SYXC'),
                                                            ((8, 8, 3), (16, 16, 3), 'YXC', 'SYXC'),
                                                            ((8, 2, 8), (8, 3, 8), 'YTX', 'SYXC'),
                                                            ((2, 2, 8, 8, 3), (2, 1, 8, 8, 3), 'TSYXC', 'SYXC'),
                                                            ((8, 2, 8, 8, 3), (8, 1, 8, 8, 3), 'ZTYXC', 'SZYXC')])
def test_load_images_from_disk_val(tmp_path, n, shape1, shape2, axes, new_axes):
    """
    Test loading from disk using different train and val paths.
    """

    # create data
    train_path = Path(tmp_path, 'train')
    train_path.mkdir()
    val_path = Path(tmp_path, 'val')
    val_path.mkdir()
    save_img(train_path, n, shape1)
    save_img(val_path, n, shape2)

    # create widget
    widget = MonkeyPatchWidget(axes, True, train=train_path, val=val_path)

    # find final 'S' dim
    prod_x = n
    if 'S' in axes:
        prod_x *= shape1[axes.find('S')]
    if 'T' in axes:
        prod_x *= shape1[axes.find('T')]

    prod_y = n
    if 'S' in axes:
        prod_y *= shape2[axes.find('S')]
    if 'T' in axes:
        prod_y *= shape2[axes.find('T')]

    _x_train, _x_val, _new_axes = load_images(widget)
    assert len(_x_train.shape) == len(new_axes)
    assert _x_train.shape[0] == prod_x
    assert _new_axes == new_axes

    assert len(_x_val.shape) == len(new_axes)
    assert _x_val.shape[0] == prod_y


@pytest.mark.parametrize('shape, axes, new_axes', [((8, 8), 'YX', 'SYXC'),
                                                   ((8, 8, 8), 'ZYX', 'SZYXC'),
                                                   ((2, 8, 8), 'SYX', 'SYXC'),
                                                   ((8, 8, 3), 'YXC', 'SYXC'),
                                                   ((8, 2, 8), 'YTX', 'SYXC'),
                                                   ((2, 2, 8, 8, 3), 'TSYXC', 'SYXC'),
                                                   ((8, 2, 8, 8, 3), 'ZTYXC', 'SZYXC')])
def test_load_images_from_layers_same_val(tmp_path, shape, axes, new_axes):
    """
    Test loading from disk using the same train and val path.
    """
    # create data
    x = np.ones(shape)

    # create widget
    widget = MonkeyPatchWidget(axes, False, train=x, val=None)

    # find final 'S' dim
    prod = 1
    if 'S' in axes:
        prod *= shape[axes.find('S')]
    if 'T' in axes:
        prod *= shape[axes.find('T')]

    _x_train, _x_val, _new_axes = load_images(widget)
    assert len(_x_train.shape) == len(new_axes)
    assert _x_train.shape[0] == prod
    assert _new_axes == new_axes
    assert _x_val is None


@pytest.mark.parametrize('shape1, shape2, axes, new_axes', [((8, 8), (16, 16), 'YX', 'SYXC'),
                                                            ((8, 8, 8), (3, 8, 8), 'ZYX', 'SZYXC'),
                                                            ((2, 8, 8), (1, 8, 8), 'SYX', 'SYXC'),
                                                            ((8, 8, 3), (16, 16, 3), 'YXC', 'SYXC'),
                                                            ((8, 2, 8), (8, 3, 8), 'YTX', 'SYXC'),
                                                            ((2, 2, 8, 8, 3), (2, 1, 8, 8, 3), 'TSYXC', 'SYXC'),
                                                            ((8, 2, 8, 8, 3), (8, 1, 8, 8, 3), 'ZTYXC', 'SZYXC')])
def test_load_images_from_layers_val(tmp_path, shape1, shape2, axes, new_axes):
    """
    Test loading from disk using different train and val paths.
    """
    # create data
    x = np.ones(shape1)
    y = np.ones(shape2)

    # create widget
    widget = MonkeyPatchWidget(axes, False, train=x, val=y)

    # find final 'S' dim
    prod_x = 1
    if 'S' in axes:
        prod_x *= shape1[axes.find('S')]
    if 'T' in axes:
        prod_x *= shape1[axes.find('T')]

    prod_y = 1
    if 'S' in axes:
        prod_y *= shape2[axes.find('S')]
    if 'T' in axes:
        prod_y *= shape2[axes.find('T')]

    _x_train, _x_val, _new_axes = load_images(widget)
    assert len(_x_train.shape) == len(new_axes)
    assert _x_train.shape[0] == prod_x

    assert len(_x_val.shape) == len(new_axes)
    assert _x_val.shape[0] == prod_y

    assert _new_axes == new_axes


@pytest.mark.parametrize('shape1, shape2', [((2, 8, 8, 1), (1, 8, 8, 1)),
                                            ((2, 8, 8, 8, 1), (1, 8, 8, 8, 1))])
def test_prepare_data(shape1, shape2):
    """
    Test preparing data for training
    """
    # create data
    x = np.ones(shape1)
    y = np.ones(shape2)

    # patch and augmentation and patching factor
    if len(shape1) == 4:
        patch_shape = (4, 4)
        prod = 8 * 4
    else:
        patch_shape = (4, 4, 4)
        prod = 8 * 4 * 2

    # prepare data
    _x, _y = prepare_data(x, y, patch_shape)
    assert _x.shape[1:-1] == patch_shape
    assert _x.shape[0] == shape1[0] * prod
    assert _y.shape[1:-1] == patch_shape
    assert _y.shape[0] == shape2[0] * prod


@pytest.mark.parametrize('n_val', [5, 15])
@pytest.mark.parametrize('shape', [(2, 8, 8, 1),
                                   (2, 8, 8, 8, 1)])
def test_prepare_data_None_val(n_val, shape):
    """
    Test preparing data for training without val data (using parameter to set the number of validation patches).
    """
    # create data
    x = np.ones(shape)
    y = None

    # patch and augmentation and patching factor
    if len(shape) == 4:
        patch_shape = (4, 4)
        prod = 8 * 4
    else:
        patch_shape = (4, 4, 4)
        prod = 8 * 4 * 2

    # prepare data
    _x, _y = prepare_data(x, y, patch_shape, n_val=n_val)
    assert _x.shape[1:-1] == patch_shape
    assert _x.shape[0] == shape[0] * prod - n_val
    assert _y.shape[1:-1] == patch_shape
    assert _y.shape[0] == n_val
