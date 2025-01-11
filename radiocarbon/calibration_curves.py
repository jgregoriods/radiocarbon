import numpy as np
import importlib.resources


def load_calibration_curve(file_name):
    with importlib.resources.path('radiocarbon.calcurves', file_name) as file_path:
        return np.loadtxt(file_path, delimiter=',')


CALIBRATION_CURVES = {
    'shcal13': load_calibration_curve('shcal13.14c'),
    'marine13': load_calibration_curve('marine13.14c'),
    'intcal13': load_calibration_curve('intcal13.14c'),
    'shcal20': load_calibration_curve('shcal20.14c'),
    'intcal20': load_calibration_curve('intcal20.14c'),
    'marine20': load_calibration_curve('marine20.14c')
}
