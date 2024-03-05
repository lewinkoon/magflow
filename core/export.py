import csv
import numpy as np
import os
from core.api import parse_dicom


def export():
    # read all the files
    data = []
    for axis in ("FH", "RL", "AP"):
        folder = f"files/{axis}"
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if parse_dicom(path) is not None:
                data.append(parse_dicom(path))

    fh = [img["val"] for img in data if img["axis"] == "FH" and img["time"] == 0]
    rl = [img["val"] for img in data if img["axis"] == "RL" and img["time"] == 0]
    ap = [img["val"] for img in data if img["axis"] == "AP" and img["time"] == 0]

    table = []
    for z, (imgy, imgx, imgz) in enumerate(zip(fh, rl, ap)):
        for index, (pxlx, pxly, pxlz) in enumerate(
            zip(imgy.flatten(), imgx.flatten(), imgz.flatten())
        ):
            row = {}
            row["x"] = np.unravel_index(index, (128, 128))[0]
            row["y"] = np.unravel_index(index, (128, 128))[1]
            row["z"] = z
            row["vx"] = pxlx
            row["vy"] = pxly
            row["vz"] = pxlz
            table.append(row)

    print(table[0])

    fields = table[0].keys()
    path = "output.csv"

    with open(path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in table:
            writer.writerow(row)
