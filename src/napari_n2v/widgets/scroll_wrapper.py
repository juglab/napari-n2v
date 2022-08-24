from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea


class ScrollWidgetWrapper(QScrollArea):
    def __init__(self, widget):
        super().__init__()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # ScrollBarAsNeeded
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(widget)

