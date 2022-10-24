from pathlib import Path

import numpy as np
from typing import Union
from tifffile import imread

from napari.utils import notifications as ntf

from napari_n2v.utils import(
    reshape_data
)


def load_and_reshape(source, axes):
    """
    Load and reshape data for use in N2V. Data can be a numpy array or a list of arrays.
    """
    # load data
    _x, new_axes = load_from_disk(source, axes)

    # reshape data
    if type(_x) == tuple:
        # loop over the images
        # _x is now tuple( list[np.array], list[Path] )
        final_axes = None
        skip = []
        for i, img in enumerate(_x[0]):

            try:
                _x[0][i], final_axes = reshape_data(img, new_axes)
            except ValueError:
                # TODO: napari 0.4.16 has ntf.show_error, but napari workflows requires 0.4.15 that doesn't
                # ntf.show_error(f'Skipped {_x[1][i].stem}, wrong dimensions')
                ntf.show_info(f'Skipped {_x[1][i].stem}, wrong dimensions')
                skip.append(i)
            finally:
                if final_axes is None:
                    final_axes = axes

        # remove skipped files
        skip.reverse()
        for i in skip:
            _x[0].pop(i)
            _x[1].pop(i)

    else:
        if 'S' not in new_axes:
            new_axes = 'S' + new_axes

        _x, final_axes = reshape_data(_x, new_axes)

    return _x, final_axes


def load_from_disk(path: Union[str, Path], axes: str):
    """
    Load images from disk. If the dimensions don't agree, the method returns a tuple of list ([images], [files]). If
    the dimensions agree, the images are stacked along the `S` dimension of `axes` or along a new dimension if `S` is
    not in `axes`.

    :param axes:
    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    images = []
    dims_agree = True
    for f in image_files:
        images.append(imread(str(f)))

        # TODO here we take the first one as reference, is it the best way?
        dims_agree = dims_agree and (images[0].shape == images[-1].shape)

    # TODO we should check whether dimensions and axes agree
    if len(images) > 0 and dims_agree:
        if 'S' in axes:
            ind_S = axes.find('S')
            final_images = np.concatenate(images, axis=ind_S)
            new_axes = axes
        else:
            new_axes = 'S' + axes
            final_images = np.stack(images, axis=0)
        return final_images, new_axes

    return (images, image_files), axes


def lazy_load_generator(path: Union[str, Path]):
    """

    :param path:
    :return:
    """
    images_path = Path(path)
    image_files = [f for f in images_path.glob('*.tif*')]

    def generator(file_list):
        counter = 0
        for f in file_list:
            counter = counter + 1
            yield imread(str(f)), f, counter

    return generator(image_files), len(image_files)
