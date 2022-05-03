from magicgui.widgets import Container
from qtpy.QtWidgets import QPushButton
import pyqtgraph as pg
import webbrowser


class TBPlotWidget(Container):

    def __setitem__(self, key, value):
        pass

    def __init__(self, max_width=None, max_height=None):
        super().__init__()

        if max_width:
            self.native.setMaximumWidth(max_width)
        if max_height:
            self.native.setMaximumHeight(max_height)

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(None)
        self.native.layout().addWidget(self.graphics_widget)

        # plot widget
        self.plot = self.graphics_widget.addPlot()
        self.plot.setLabel("bottom", "epoch")
        self.plot.setLabel("left", "loss")

        # tensorboard button
        tb_button = QPushButton("Open in tensorboard")
        tb_button.clicked.connect(self.open_tb)
        self.native.layout().addWidget(tb_button)

        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.url = None
        self.tb = None

    def stop_tb(self):
        # haven't found any good way to stop the tb process, there's currently no API for it
        pass

    def open_tb(self):
        if not self.tb:
            from tensorboard import program

            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', 'models'])
            self.url = self.tb.launch()

            webbrowser.open(self.url)
        else:
            webbrowser.open(self.url)

    def update_plot(self, epoch, train_loss, val_loss):
        self.plot.clear()

        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self.plot.plot(self.epochs, self.train_loss, pen=pg.mkPen(color=(204, 221, 255)), symbol='o', symbolSize=2)
        self.plot.plot(self.epochs, self.val_loss, pen=pg.mkPen(color=(244, 173, 173)), symbol='o', symbolSize=2)

    def clear_plot(self):
        self.plot.clear()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
