from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QFileDialog
)


class FolderWidget(QWidget):
    """
    A widget used for selecting an existing folder.
    """
    def __init__(self, text):
        super().__init__()

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # text field
        self.text_field = QLineEdit('')
        self.layout().addWidget(self.text_field)

        # folder selection button
        self.button = QPushButton(text)
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._open_dialog)

    def _open_dialog(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        print(path)

        # set text in the text field
        self.text_field.setText(path)

    def get_folder(self):
        return self.text_field.text()

