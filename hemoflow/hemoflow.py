import os
from functools import partial
import multiprocessing
from tqdm import tqdm
import hemoflow.helpers as hf


def main():
    # create a list of dictionries with the read data
    fh = hf.parse("FH")
    rl = hf.parse("RL")
    ap = hf.parse("AP")
    mk = hf.parse("MK")

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
        worker = partial(hf.export, fh, rl, ap, mk, spcx, spcy, spcz)
        with multiprocessing.Pool() as pool:
            for _ in pool.imap_unordered(worker, timeframes):
                pbar.update()


if __name__ == "__main__":
    main()
