from pathlib import Path
import titfffile
import napari

from napari_n2v import TrainingWidgetWrapper

# create a Viewer
viewer = napari.Viewer()

# add napari-n2v plugin
viewer.window.add_dock_widget(TrainingWidgetWrapper(viewer))

# load yout image
path = Path('path/to/your/image.tif')
data = titfffile.imread(path)

# add image to napari
viewer.add_image(data[0][0], name=data[0][1]['name'])

# start UI
napari.run()
