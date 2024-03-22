import argparse
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger


def wrapper(fh, rl, ap, mk, voxel, volume, time):
    data = hf.tabulate(fh, rl, ap, mk, voxel, volume, time)
    hf.export(data, time)
    logger.info(f"Trigger time {time} exported")


def main():
    # setup argument parsing
    parser = argparse.ArgumentParser(
        description="Export mri flow dicom files to velocity field in csv format."
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        help="Select a timeframe",
    )
    args = parser.parse_args()
    logger.info("Script started successfully.")

    # create a list of dictionaries with the read data
    fh = hf.parse("FH")
    logger.info(f"FH series: {len(fh)} images.")
    rl = hf.parse("RL")
    logger.info(f"RL series: {len(rl)} images.")
    ap = hf.parse("AP")
    logger.info(f"AP series: {len(ap)} images.")
    mk = hf.parse("MK")
    logger.info(f"MK series: {len(mk)} images.")

    # list unique trigger times
    timeframes = sorted(set(item["time"] for item in fh))
    logger.info(f"Timeframes: {len(timeframes)}")

    # get volume dimensions
    volume = (
        fh[0]["val"].shape[0],
        fh[0]["val"].shape[1],
        len(set(item["loc"] for item in fh)),
    )
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px, {volume[2]} px)")

    # get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    # # export csv files
    # hf.check_folder()
    # for t in timeframes:
    #     data = hf.tabulate(fh, rl, ap, mk, voxel, t)
    #     hf.export(data, t)
    #     logger.info(f"Trigger time {t} exported")
    # logger.info("Script finished successfully.")

    # export csv files with multiprocessing
    worker = partial(wrapper, fh, rl, ap, mk, voxel, volume)
    with multiprocessing.Pool() as pool:
        pool.map(worker, timeframes)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
