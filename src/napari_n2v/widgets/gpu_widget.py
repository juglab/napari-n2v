from qtpy.QtWidgets import QLabel


def create_gpu_label():
    from tensorflow import config
    n_gpu = len(config.list_physical_devices('GPU'))

    if n_gpu > 0:
        text = 'GPU'
        color = 'ADC2A9'
    else:
        text = 'CPU'
        color = 'FFDBA4'

    gpu_label = QLabel(text)
    gpu_label.setStyleSheet('font-weight: bold; color: #{};'.format(color))
    gpu_label.setToolTip('Indicates whether DenoiSeg will run on GPU or CPU')

    return gpu_label
