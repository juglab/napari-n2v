import webbrowser

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit
)
from qtpy.QtGui import QPixmap, QCursor, QFont

from napari_n2v.resources import ICON_GITHUB


def _create_link(link: str, text: str) -> QLabel:
    """

    :param link: the string this label should link to
    :return: returns a QLabel object with a hyperlink
    :rtype: object
    """
    label = QLabel()
    label.setContentsMargins(0, 5, 0, 5)
    # TODO: is there a non-dark mode in napari?
    label.setText("<a href=\'{}\' style=\'color:white\'>{}</a>".format(link, text))

    font = QFont()
    font.setPointSize(11)
    label.setFont(font)

    label.setOpenExternalLinks(True)

    return label


def _open_link(link: str):
    def link_opener(event):
        webbrowser.open(link)

    return link_opener


class BannerWidget(QWidget):

    def __init__(self,
                 title: str,
                 img_path: str,
                 short_desc: str,
                 wiki_link: str,
                 github_link: str):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # logo
        icon = QPixmap(img_path)
        img_widget = QLabel()
        img_widget.setPixmap(icon)
        img_widget.setFixedSize(128, 128)

        # right panel
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # title
        title = QLabel(title)
        title.setStyleSheet("font-weight: bold;")

        # description
        description_widget = QPlainTextEdit()
        description_widget.setReadOnly(True)
        description_widget.setPlainText(short_desc)
        description_widget.setFixedSize(256, 50)

        # bottom widget
        bottom_widget = QWidget()
        bottom_widget.setLayout(QHBoxLayout())

        # github logo
        gh_icon = QPixmap(ICON_GITHUB)
        gh_widget = QLabel()
        gh_widget.setPixmap(gh_icon)
        gh_widget.mousePressEvent = _open_link(github_link)
        gh_widget.setCursor(QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        gh_widget.setToolTip('Report issues')

        # add widgets
        bottom_widget.layout().addWidget(_create_link(wiki_link, "Documentation"))
        bottom_widget.layout().addWidget(gh_widget)

        right_widget.layout().addWidget(title)
        right_widget.layout().addWidget(description_widget)
        right_widget.layout().addWidget(bottom_widget)

        # add widgets
        layout.addWidget(img_widget)
        layout.addWidget(right_widget)
