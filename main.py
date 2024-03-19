import os
from functools import partial
import multiprocessing
import numpy as np
import sys
from tqdm import tqdm
from core.api import parse_dicom, write_csv


def export():
    # create a list of dictionries with the read data
    fh = parse_dicom("FH")
    rl = parse_dicom("RL")
    ap = parse_dicom("AP")
    mk = parse_dicom("MK")

    # get pixel spacing
    timeframes = sorted(set(item["time"] for item in fh))
    spcx = fh[0]["spacing"][0]
    spcy = fh[0]["spacing"][1]
    spcz = fh[0]["height"]

    # # export csv files
    # for t in timeframes:
    #     write_csv(fh, rl, ap, mk, spcx, spcy, spcz, t)

    # export to csv with multiprocessing
    with tqdm(total=len(timeframes)) as pbar:
        worker = partial(write_csv, fh, rl, ap, mk, spcx, spcy, spcz)
        with multiprocessing.Pool() as pool:
            for _ in pool.imap_unordered(worker, timeframes):
                pbar.update()


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export()
    else:
        export()
