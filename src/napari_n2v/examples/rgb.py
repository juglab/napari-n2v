import os
from pathlib import Path
import urllib
import zipfile

import napari
from skimage import io

from napari_n2v import TrainingWidgetWrapper

if __name__ == "__main__":
    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(TrainingWidgetWrapper(viewer))

    # create a folder for our data
    data_dir = Path('./../data')
    if not data_dir.exists():
        os.mkdir(data_dir)

    # check if data has been downloaded already
    zipPath = Path(data_dir / 'RGB.zip')
    if not zipPath.exists():
        # download and unzip data
        data = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/RGB.zip', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    img_path = Path(data_dir / 'longBeach.png')
    img = io.imread(img_path)
    viewer.add_image(img, name='rgb data')

    napari.run()
