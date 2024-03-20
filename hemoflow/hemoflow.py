from functools import partial
import multiprocessing
from tqdm import tqdm
import hemoflow.helpers as hf
from hemoflow.logger import logger as log


def main():
    # create a list of dictionaries with the read data
    fh = hf.parse("FH")
    rl = hf.parse("RL")
    ap = hf.parse("AP")
    mk = hf.parse("MK")
    log.info("Reading mri flow files ...")

    # list unique trigger times
    log.info("Getting trigger times ...")
    timeframes = sorted(set(item["time"] for item in fh))

    # get voxel spacing
    log.info("Getting voxel spacing values ...")
    spcx = fh[0]["spacing"][0]
    spcy = fh[0]["spacing"][1]
    spcz = fh[0]["height"]

    # # export csv files
    # for t in timeframes:
    #     write_csv(fh, rl, ap, mk, spcx, spcy, spcz, t)

    # export to csv with multiprocessing
    log.info("Exporting data ...")
    with tqdm(total=len(timeframes)) as pbar:
        worker = partial(hf.export, fh, rl, ap, mk, spcx, spcy, spcz)
        with multiprocessing.Pool() as pool:
            for _ in pool.imap_unordered(worker, timeframes):
                pbar.update()


if __name__ == "__main__":
    main()
