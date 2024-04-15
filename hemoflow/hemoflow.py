import argparse
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger


def wrapper(fh, rl, ap, mk, voxel, time):
    fh = hf.filter(fh, time)
    rl = hf.filter(rl, time)
    ap = hf.filter(ap, time)

    if mk is not None:
        fh, rl, ap = hf.mask(fh, rl, ap, mk)

    data = hf.tabulate(fh, rl, ap, voxel, time)
    hf.export(data, time)
    logger.info(f"Trigger time {time} exported")


def main():
    # setup argument parsing
    parser = argparse.ArgumentParser(
        description="Export mri flow dicom files to velocity field in csv format."
    )
    parser.add_argument(
        "-m",
        "--mask",
        action="store_true",
        help="Mask velocity field with segmentation.",
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

    # create m series list if required
    if args.mask:
        mk = hf.parse("M")
        logger.info(f"M series: {len(mk)} images.")
    else:
        mk = None

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

    # export csv files with multiprocessing
    worker = partial(wrapper, fh, rl, ap, mk, voxel)
    with multiprocessing.Pool() as pool:
        pool.map(worker, timeframes)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
