from qtpy.QtWidgets import (
    QSpinBox,
    QProgressBar,
    QDoubleSpinBox,
)


class SpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class DoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


def create_double_spinbox(min_value: float = 0,
                          max_value: float = 1,
                          value: float = 0.5,
                          step: float = 0.1,
                          visible: bool = True,
                          tooltip: str = None,
                          n_decimal: int = 1) -> DoubleSpinBox:
    spin_box = DoubleSpinBox()
    spin_box.setDecimals(n_decimal)
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(step)
    spin_box.setValue(value)
    spin_box.setVisible(visible)
    spin_box.setToolTip(tooltip)
    spin_box.setMinimumHeight(50)
    spin_box.setContentsMargins(0, 3, 0, 3)
    return spin_box


def create_int_spinbox(min_value: int = 1,
                       max_value: int = 1000,
                       value: int = 2,
                       step: int = 1,
                       visible: bool = True,
                       tooltip: str = None) -> SpinBox:
    spin_box = SpinBox()
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)
    spin_box.setSingleStep(step)
    spin_box.setValue(value)
    spin_box.setVisible(visible)
    spin_box.setToolTip(tooltip)
    spin_box.setMinimumHeight(50)
    spin_box.setContentsMargins(0, 3, 0, 3)

    return spin_box


def create_progressbar(min_value: int = 0,
                       max_value: int = 100,
                       value: int = 0,
                       text_visible: bool = True,
                       visible: bool = True,
                       text_format: str = f'Epoch ?/{100}',
                       tooltip: str = None) -> QProgressBar:
    progress_bar = QProgressBar()
    progress_bar.setMinimum(min_value)
    progress_bar.setMaximum(max_value)
    progress_bar.setValue(value)
    progress_bar.setVisible(visible)
    progress_bar.setTextVisible(text_visible)
    progress_bar.setFormat(text_format)
    progress_bar.setToolTip(tooltip)
    progress_bar.setMinimumHeight(30)

    return progress_bar
