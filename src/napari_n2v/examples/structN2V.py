import os
import urllib

import numpy as np
import napari
from napari_n2v import TrainingWidgetWrapper

if __name__ == "__main__":
    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(TrainingWidgetWrapper(viewer))

    # create a folder for our data
    if not os.path.isdir('../data'):
        os.mkdir('../data')

    # check if data has been downloaded already
    dataPath = "data/gt.npy"
    if not os.path.exists(dataPath):
        _ = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/gt.npy', dataPath)
    X = np.load(dataPath).astype(np.float32)

    from scipy.ndimage import convolve

    purenoise = []
    noise_kernel = np.array([[1, 1, 1]]) / 3  # horizontal correlations
    a, b, c = X.shape
    for i in range(a):
        noise = np.random.rand(b, c) * 1.5
        noise = convolve(noise, noise_kernel)
        purenoise.append(noise)
    purenoise = np.array(purenoise)
    purenoise = purenoise - purenoise.mean()

    noisy_dataset = X + purenoise

    viewer.add_image(noisy_dataset, name='synthetic data')

    napari.run()
