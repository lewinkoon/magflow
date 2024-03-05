import numpy as np
import os
import pyvista as pv

from core.api import parse_dicom


def gui():
    # fill the dictionary with data
    data = {"FH": [], "RL": [], "AP": []}
    for key in data:
        folder = f"files/{key}"
        files = os.listdir(folder)
        series = [
            parse_dicom(file, folder)
            for file in files
            if parse_dicom(file, folder) is not None
        ]
        series = sorted(series, key=lambda x: x["num"])
        data[key] = np.reshape(np.array(series), (40, 24))
