import numpy as np
import pytest

from napari_n2v._tests.test_utils import create_simple_model, save_img, save_weights_h5
from napari_n2v.utils import reshape_data
from napari_n2v.utils.prediction_worker import (
    _run_lazy_prediction,
    _run_prediction,
    _run_prediction_to_disk,
    _predict
)
from napari_n2v.utils import State, UpdateType, lazy_load_generator, load_from_disk


class MonkeyPatchWidget:
    def __init__(self):
        self.state = State.RUNNING
        self.denoi_prediction = None
        self.scale = None

    def get_model_path(self):
        return self.path


@pytest.mark.parametrize('n_tiles', [1, 2])
@pytest.mark.parametrize('shape, axes',
                         [((16, 16), 'YX'),
                          ((16, 16, 3), 'YXC'),
                          ((2, 16, 16), 'TYX'),
                          ((16, 16, 16), 'ZYX'),
                          ((2, 16, 16, 3), 'SYXC'),
                          ((2, 16, 16, 16), 'SZYX'),
                          ((2, 16, 3, 16, 16), 'SZCYX')])
def test_predict_after_training_same_size(tmp_path, n_tiles, shape, axes):
    # create data
    x = np.ones(shape)

    # shape for n2v
    _x, new_axes = reshape_data(x, axes)

    # create model
    model = create_simple_model(tmp_path, _x.shape)

    # create widget
    widget = MonkeyPatchWidget()

    # prediction variable
    pred = np.ones(_x.shape)

    # predict
    results = []
    predictor = _predict(widget, model, _x, new_axes, pred, is_tiled=n_tiles != 1, n_tiles=n_tiles)
    while True:
        t = next(predictor, None)

        if t is not None:
            results.append(t)
        else:
            break

    # check results
    assert len(results) == _x.shape[0]
    assert [pred[i, ...].max() != 1 for i in range(pred.shape[0])]


@pytest.mark.parametrize('n_tiles', [1, 2])
@pytest.mark.parametrize('shape1, shape2, axes',
                         [((16, 16), (32, 32), 'YX'),
                          ((2, 16, 16), (2, 32, 32), 'TYX'),
                          ((3, 16, 16), (3, 32, 32), 'CYX'),
                          ((16, 16, 16), (16, 32, 32), 'ZYX'),
                          ((2, 16, 16, 3), (2, 32, 32, 3), 'SYXC'),
                          ((3, 16, 32, 32, 3), (2, 16, 16, 16, 3), 'SZYXC')])
def test_predict_after_training_list(tmp_path, n_tiles, shape1, shape2, axes):
    # create data
    x1 = np.ones(shape1)
    x2 = np.ones(shape2)

    # shape for n2v
    _x1, new_axes = reshape_data(x1, axes)
    _x2, _ = reshape_data(x2, axes)
    _x = ([_x1, _x2], [tmp_path / 'x1.tif', tmp_path / 'x2.tif'])

    # create model
    model = create_simple_model(tmp_path, _x1.shape)

    # create widget
    widget = MonkeyPatchWidget()

    # predict
    results = []
    predictor = _predict(widget, model, _x, new_axes, None, is_tiled=n_tiles != 1, n_tiles=n_tiles)
    while True:
        t = next(predictor, None)

        if t is not None:
            results.append(t)
        else:
            break

    # check results
    assert len(results) == 2

    final_path = tmp_path / 'results'
    assert len([f for f in final_path.glob('*.tif')]) == 2


@pytest.mark.parametrize('shape1, shape2, axes',
                         [((5, 16, 16), (3, 16, 16), 'CYX'),
                          ((2, 16, 16, 5), (2, 16, 16, 3), 'SYXC')])
def test_predict_after_training_list_incompatible_C(tmp_path, shape1, shape2, axes):
    # create data
    x1 = np.ones(shape1)
    x2 = np.ones(shape2)

    # shape for n2v
    _x1, new_axes = reshape_data(x1, axes)
    _x2, _ = reshape_data(x2, axes)
    _x = ([_x1, _x2], [tmp_path / 'x1.tif', tmp_path / 'x2.tif'])

    # create model
    model = create_simple_model(tmp_path, _x1.shape)

    # create widget
    widget = MonkeyPatchWidget()

    # predict
    with pytest.raises(ValueError):
        predictor = _predict(widget, model, _x, new_axes, None)
        while True:
            t = next(predictor, None)


@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('n_tiles', [1, 2])
@pytest.mark.parametrize('shape, shape_n2v, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (2, 16, 16, 1), 'TYX'),
                          ((3, 16, 16), (1, 16, 16, 3), 'CYX'),
                          ((16, 16, 16), (1, 16, 16, 16, 1), 'ZYX'),
                          ((2, 16, 16, 3), (2, 16, 16, 3), 'SYXC'),
                          ((2, 16, 32, 32), (2, 16, 32, 32, 1), 'SZYX'),
                          ((2, 16, 3, 32, 32), (2, 16, 32, 32, 3), 'SZCYX')])
def test_run_lazy_prediction_same_size(tmp_path, n, n_tiles, shape, shape_n2v, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)

    # create files
    save_img(tmp_path, n, shape)

    # instantiate generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == n

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget()
    parameters = (mk, model, axes, gen)
    hist = list(_run_lazy_prediction(*parameters,
                                     is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                     n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n + 1

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == n


@pytest.mark.parametrize('n_tiles', [1])
@pytest.mark.parametrize('shape1, shape2, shape_n2v, axes',
                         [((16, 16), (32, 32), (1, 16, 16, 1), 'YX'),
                          ((2, 16, 16), (1, 32, 32), (2, 16, 16, 1), 'TYX'),
                          ((16, 16, 3), (32, 32, 3), (1, 16, 16, 3), 'YXC'),
                          ((16, 32, 32), (16, 16, 16), (1, 16, 32, 32, 1), 'ZYX'),
                          ((16, 32, 3, 32), (16, 16, 3, 16), (1, 16, 32, 32, 3), 'ZYCX'),
                          ((2, 16, 32, 32), (1, 16, 16, 16), (2, 16, 32, 32, 1), 'SZYX'),
                          ((2, 3, 16, 32, 32), (1, 3, 16, 16, 16), (2, 16, 32, 32, 3), 'SCZYX')])
def test_run_lazy_prediction_different_sizes(tmp_path, n_tiles, shape1, shape2, shape_n2v, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)
    save_weights_h5(model, tmp_path)

    # create files
    n = 1
    save_img(tmp_path, n, shape1, prefix='i1_')
    save_img(tmp_path, n, shape2, prefix='i2_')

    # instantiate generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == 2 * n

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget()
    parameters = (mk, model, axes, gen)
    hist = list(_run_lazy_prediction(*parameters,
                                     is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                     n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == 2 * n + 1

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == 2 * n


@pytest.mark.parametrize('shape1, shape2, shape_n2v, axes',
                         [((16, 16, 3), (32, 32, 4), (1, 16, 16, 3), 'YXC'),
                          ((16, 32, 3, 32), (16, 16, 1, 16), (1, 16, 32, 32, 3), 'ZYCX'),
                          ((2, 3, 16, 32, 32), (2, 5, 16, 16, 16), (2, 16, 32, 32, 3), 'SCZYX')])
def test_run_lazy_prediction_different_C(tmp_path, shape1, shape2, shape_n2v, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    n = 1
    save_img(tmp_path, n, shape1, prefix='i1_')
    save_img(tmp_path, n, shape2, prefix='i2_')

    # instantiate generator
    gen, m = lazy_load_generator(tmp_path)
    assert m == 2 * n

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget()
    parameters = (mk, model, axes, gen)
    hist = list(_run_lazy_prediction(*parameters))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == 2 * n + 1

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == n


@pytest.mark.parametrize('n_tiles', [1])
@pytest.mark.parametrize('shape1, shape2, shape_n2v, axes',
                         [((16, 16), (32, 32), (1, 16, 16, 1), 'YX'),
                          ((16, 16, 3), (32, 32, 3), (1, 16, 16, 3), 'YXC'),
                          ((3, 16, 16), (3, 32, 32), (1, 16, 16, 3), 'CYX'),
                          ((5, 16, 16), (3, 32, 32), (5, 16, 16, 1), 'TYX'),
                          ((16, 32, 32), (16, 16, 16), (1, 16, 32, 32, 1), 'ZYX'),
                          ((16, 3, 32, 32), (16, 3, 16, 16), (1, 16, 32, 32, 3), 'ZCYX'),
                          ((5, 16, 32, 32), (3, 16, 16, 16), (5, 16, 32, 32, 1), 'SZYX'),
                          ((5, 16, 32, 32, 3), (3, 16, 16, 16, 3), (5, 16, 32, 32, 3), 'SZYXC')])
def test_run_from_disk_prediction_different_sizes(tmp_path, n_tiles, shape1, shape2, shape_n2v, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    n = 1
    save_img(tmp_path, n, shape1, prefix='i1_')
    save_img(tmp_path, n, shape2, prefix='i2_')

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)
    assert type(images) == tuple
    assert new_axes == axes if 'S' in axes else 'S' + axes

    # run prediction
    mk = MonkeyPatchWidget()
    parameters = (mk, model, new_axes, images)
    hist = list(_run_prediction_to_disk(*parameters,
                                        is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                        n_tiles=n_tiles))

    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n * 2 + 2

    # check that images have been saved
    folder = tmp_path / 'results'
    image_files = [f for f in folder.glob('*.tif*')]
    assert len(image_files) == 2 * n


@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('n_tiles', [1])
@pytest.mark.parametrize('shape, shape_n2v, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((16, 16, 3), (1, 16, 16, 3), 'YXC'),
                          ((5, 16, 16), (5, 16, 16, 1), 'TYX'),
                          ((16, 32, 32), (1, 16, 32, 32, 1), 'ZYX'),
                          ((16, 3, 32, 32), (1, 16, 32, 32, 3), 'ZCYX'),
                          ((5, 16, 32, 32), (5, 16, 32, 32, 1), 'SZYX'),
                          ((5, 16, 32, 32, 3), (5, 16, 32, 32, 3), 'SZYXC')])
def test_run_prediction_from_disk_numpy(tmp_path, n, n_tiles, shape, shape_n2v, axes):
    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)
    path_to_h5 = save_weights_h5(model, tmp_path)

    # create files
    save_img(tmp_path, n, shape)

    # load images
    images, new_axes = load_from_disk(tmp_path, axes)

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget()
    parameters = (mk, model, new_axes, images)
    hist = list(_run_prediction(*parameters,
                                is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == n * shape_n2v[0] + 2


@pytest.mark.qt
@pytest.mark.parametrize('n_tiles', [1])
@pytest.mark.parametrize('shape, shape_n2v, axes',
                         [((16, 16), (1, 16, 16, 1), 'YX'),
                          ((16, 16, 3), (1, 16, 16, 3), 'YXC'),
                          ((5, 16, 16), (5, 16, 16, 1), 'TYX'),
                          ((16, 32, 32), (1, 16, 32, 32, 1), 'ZYX'),
                          ((16, 3, 32, 32), (1, 16, 32, 32, 3), 'ZCYX'),
                          ((5, 3, 16, 16), (15, 16, 16, 1), 'TSYX'),
                          ((5, 16, 32, 32, 3), (5, 16, 32, 32, 3), 'SZYXC')])
def test_run_prediction_from_layers(tmp_path, make_napari_viewer, n_tiles, shape, shape_n2v, axes):
    viewer = make_napari_viewer()

    # create images
    name = 'images'
    img = np.zeros(shape)
    viewer.add_image(img, name=name)

    # create model and save it to disk
    model = create_simple_model(tmp_path, shape_n2v)

    # run prediction (it is a generator)
    mk = MonkeyPatchWidget()
    parameters = (mk, model, axes, viewer.layers['images'].data)
    hist = list(_run_prediction(*parameters,
                                is_tiled=n_tiles != 1,  # use tiles if n_tiles != 1
                                n_tiles=n_tiles))
    assert hist[-1] == {UpdateType.DONE}
    assert len(hist) == shape_n2v[0] + 2
