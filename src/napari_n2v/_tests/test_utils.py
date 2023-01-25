import os
import pytest
from pathlib import Path

from tifffile import imwrite
import numpy as np
from n2v.models import N2V
from napari_n2v.utils import create_config
from napari_n2v.utils.io_utils import generate_bioimage_md


###################################################################
# convenience functions: save images
def save_img(folder_path, n, shape, prefix='', axes=None):
    """
    Create `n` images in `folder_path` with shape `shape. The name of the files can be preceded by `prefix` in order
    to create files with different shapes in the same folder (by calling the method twice with different prefixes).
    """
    for i in range(n):
        im = np.random.randint(0, 65535, shape, dtype=np.uint16)

        if axes is None:
            imwrite(os.path.join(folder_path, prefix + str(i) + '.tif'), im)
        else:
            assert len(axes) == len(shape)
            imwrite(os.path.join(folder_path, prefix + str(i) + '.tif'), im, metadata={'axes': axes})


def create_data(main_dir, folders, sizes, shapes):
    for n, f, sh in zip(sizes, folders, shapes):
        source = main_dir / f
        os.mkdir(source)
        save_img(source, n, sh)


@pytest.mark.parametrize('shape', [(16, 16), (8, 16, 16)])
def test_create_data(tmp_path, shape):
    folders = ['train_X', 'train_Y', 'val_X', 'val_Y']
    sizes = [20, 8, 5, 5]

    create_data(tmp_path, folders, sizes, [shape for _ in sizes])

    for n, f in zip(sizes, folders):
        source = tmp_path / f
        files = [f for f in Path(source).glob('*.tif*')]
        assert len(files) == n


# convenience functions: create and save models
def create_simple_model(basedir, shape):
    # create model
    X = np.concatenate([np.ones(shape), np.zeros(shape)], axis=0)
    name = 'myModel'
    config = create_config(X)

    assert config.is_valid()

    return N2V(config, name, basedir)


@pytest.mark.parametrize('shape', [(1, 8, 8, 1),
                                   (20, 8, 16, 1),
                                   (1, 8, 16, 16, 1),
                                   (42, 8, 16, 32, 1)])
def test_create_model(tmp_path, shape):
    create_simple_model(tmp_path, shape)


# todo still necessary?
def save_weights_h5(model, basedir):
    name_weights = 'myModel.h5'
    path_to_weights = Path(basedir, name_weights)

    # save model
    model.keras_model.save_weights(path_to_weights)

    return path_to_weights


@pytest.mark.parametrize('shape', [(1, 8, 8, 1),
                                   (1, 8, 16, 1),
                                   (1, 8, 16, 16, 1),
                                   (1, 8, 16, 32, 1)])
def test_saved_weights_h5(tmp_path, shape):
    model = create_simple_model(tmp_path, shape)
    path_to_weights = save_weights_h5(model, tmp_path)

    assert path_to_weights.name.endswith('.h5')
    assert path_to_weights.exists()


def create_model_zoo_parameters(folder, shape):
    # create model and save it to disk
    model = create_simple_model(folder, shape)
    path_to_h5 = str(save_weights_h5(model, folder).absolute())

    # path to modelzoo
    path_to_modelzoo = path_to_h5[:-len('.h5')] + '.bioimage.io.zip'

    # inputs/outputs
    path_to_input = path_to_h5[:-len('.h5')] + '-input.npy'
    np.save(path_to_input, np.zeros(shape))
    assert Path(path_to_input).exists()

    path_to_output = path_to_h5[:-len('.h5')] + '-output.npy'
    np.save(path_to_output, np.zeros(shape))
    assert Path(path_to_output).exists()

    name = 'Rincewind'
    text = 'Great A’Tuin the turtle comes, swimming slowly through the interstellar gulf, hydrogen frost on his ' \
           'ponderous limbs, his huge and ancient shell pocked with meteor craters. '
    path_to_doc = generate_bioimage_md(name, [{'text': text}])
    assert path_to_doc.exists()

    import shutil
    folder_path, _ = os.path.split(path_to_h5)
    shutil.copyfile(os.path.join(path_to_h5[:-len('.h5')], "config.json"), os.path.join(folder_path, "config.json"))

    # algorithm details
    name = 'One algo to rule them all'
    authors = [{'name': 'Sauron'}, {'name': 'Aule'}]
    cite = [{'text': 'It is told among the wise that the First War began before Arda was full-shaped',
            'doi': '10.1126/science.356.6335.236'}]

    # tf version
    tf_version = 42

    # axes
    if len(shape) == 4:
        axes = 'byxc'
    elif len(shape) == 5:
        axes = 'bzyxc'
    else:
        axes = ''

    return path_to_modelzoo, path_to_h5, path_to_input, path_to_output, path_to_doc, name, authors, cite, tf_version, axes
