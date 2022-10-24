from pathlib import Path

from magicgui.widgets import Container

from qtpy.QtGui import QCursor, QIcon, QPixmap
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QWidget,
    QHBoxLayout
)
import pyqtgraph as pg
import webbrowser

from napari_n2v.resources import ICON_TF
from napari_n2v.utils import get_default_path


class TBPlotWidget(Container):
    """
    Widget used to display training graph including training and validation losses. The widget also includes a button
    to open TensorBoard in the browser.
    """
    def __setitem__(self, key, value):
        pass

    def __init__(self, min_width=None, min_height=None, max_width=None, max_height=None):
        super().__init__()

        if max_width:
            self.native.setMaximumWidth(max_width)
        if max_height:
            self.native.setMaximumHeight(max_height)
        if min_width:
            self.native.setMinimumWidth(min_width)
        if min_height:
            self.native.setMinimumHeight(min_height)

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(None)
        self.native.layout().addWidget(self.graphics_widget)

        # plot widget
        self.plot = self.graphics_widget.addPlot()
        self.plot.setLabel("bottom", "epoch")
        self.plot.setLabel("left", "loss")
        self.plot.addLegend(offset=(125, -50))

        # tensorboard button
        tb_button = QPushButton("Open in TensorBoard")
        tb_button.setToolTip('Open TensorBoard in your browser')
        tb_button.setIcon(QIcon(QPixmap(ICON_TF)))
        tb_button.setLayoutDirection(Qt.LeftToRight)
        tb_button.setIconSize(QSize(32, 29))
        tb_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        tb_button.clicked.connect(self.open_tb)

        # add to layout on the bottom left
        button_widget = QWidget()
        button_widget.setLayout(QHBoxLayout())
        button_widget.layout().addWidget(tb_button)
        button_widget.layout().addWidget(QLabel(''))
        self.native.layout().addWidget(button_widget)

        # set empty references
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.url = None
        self.tb = None

    def stop_tb(self):
        # haven't found any good way to stop the tb process, there's currently no API for it
        pass

    def open_tb(self):
        """
        Open TensorBoard in the browser.
        :return:
        """
        if not self.tb:
            from tensorboard import program

            self.tb = program.TensorBoard()

            path = str(Path(get_default_path(), 'models').absolute())
            self.tb.configure(argv=[None, '--logdir', path])
            self.url = self.tb.launch()

            webbrowser.open(self.url)
        else:
            webbrowser.open(self.url)

    def update_plot(self, epoch, train_loss, val_loss):
        """
        Add a new point to the graph.

        :param epoch: Epoch number, x-axis
        :param train_loss: Training loss at the end of `epoch`
        :param val_loss: Validation loss at the end of `epoch`
        :return:
        """
        # clear the plot
        self.plot.clear()

        # add the new points
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        # replot
        self.plot.plot(self.epochs,
                       self.train_loss,
                       pen=pg.mkPen(color=(204, 221, 255)),
                       symbol='o',
                       symbolSize=2,
                       name='Train')
        self.plot.plot(self.epochs,
                       self.val_loss,
                       pen=pg.mkPen(color=(244, 173, 173)),
                       symbol='o',
                       symbolSize=2,
                       name='Val')

    def clear_plot(self):
        """
        Clear the plot.
        :return:
        """
        self.plot.clear()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
