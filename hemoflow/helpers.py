import csv
import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut


def parse(axis):
    res = []
    folder_path = f"files/{axis}"
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        slice = {}
        with open(file_path, "rb") as file:
            ds = dcmread(file)
            slice["axis"] = ds[0x0008, 0x103E].value
            slice["num"] = ds[0x0020, 0x0013].value  # get instance number
            slice["spacing"] = ds[0x0028, 0x0030].value
            slice["pxl"] = ds.pixel_array
            if ds[0x0008, 0x0008].value[2] == "PHASE CONTRAST M":
                slice["height"] = ds[0x0018, 0x0088].value
                slice["loc"] = ds[0x0020, 0x1041].value  # get slice location
                slice["time"] = ds[0x0018, 0x1060].value
                slice["val"] = apply_modality_lut(
                    ds.pixel_array, ds
                )  # convert pixel values to velocity values
            res.append(slice)
    return res


def filter(series, time):
    # filter images by axis and time
    res = [slice["val"] for slice in series if slice["time"] == time]
    return res


def tabulate(fh, rl, ap, voxel, time):
    # convert data into tabular dictionary
    dimensions = fh[0].shape
    res = []
    for z, (imgx, imgy, imgz) in enumerate(zip(ap, fh, rl)):
        for index, (pxlx, pxly, pxlz) in enumerate(
            zip(imgx[::-1].flatten(), imgy[::-1].flatten(), imgz[::-1].flatten())
        ):
            row = {}
            row["x"] = np.unravel_index(index, dimensions)[1] * voxel[0]
            row["y"] = np.unravel_index(index, dimensions)[0] * voxel[1]
            row["z"] = z * voxel[2]
            row["t"] = time
            row["vx"] = pxlx
            row["vy"] = pxly
            row["vz"] = pxlz
            res.append(row)
    return res


def mask(fh, rl, ap, mk):
    fh, rl, ap = [], [], []
    for imgx, imgy, imgz, imgm in zip(ap, fh, rl, mk):
        imgx[imgm == 0] = 0
        imgy[imgm == 0] = 0
        imgz[imgm == 0] = 0

        fh.append(imgy)
        rl.append(imgz)
        ap.append(imgx)

    return fh, rl, ap


def export(data, time):
    # export data as csv
    if not os.path.exists("output"):
        os.makedirs("output")
    fields = data[0].keys()
    path = f"output/data.csv.{time}"
    with open(path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
