import click
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger
import os
import sys
from pydicom import dcmread
import numpy as np


@click.group(
    epilog="Check out readme at https://github.com/lewinkoon/hemoflow for more details."
)
def cli():
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
    pass


@cli.command(help="Create volumetric velocity field from dicom files.")
@click.argument("path", default="files", type=click.Path())
@click.option("--frames", type=int, help="Number of frames in the sequence.")
def build(frames):

    def wrapper(fh, rl, ap, voxel, time):
        fh = hf.filter(fh, time)
        rl = hf.filter(rl, time)
        ap = hf.filter(ap, time)

        data = hf.tabulate(fh, rl, ap, voxel, time)
        hf.export(data, time)
        logger.info(f"Trigger time {time} exported with {len(data)} rows.")

    # create a list of dictionaries with the read data
    fh = hf.parse("FH", frames)
    logger.info(f"FH series: {len(fh)} images.")
    rl = hf.parse("RL", frames)
    logger.info(f"RL series: {len(rl)} images.")
    ap = hf.parse("AP", frames)
    logger.info(f"AP series: {len(ap)} images.")

    # list unique trigger times
    timeframes = sorted(set(item["time"] for item in fh))
    logger.info(f"Timeframes: {len(timeframes)}")

    # get volume dimensions
    volume = (
        fh[0]["pxl"].shape[0],
        fh[0]["pxl"].shape[1],
        len(set(item["loc"] for item in fh)),
    )
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px, {volume[2]} px)")

    # get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    # export csv files with multiprocessing
    worker = partial(wrapper, fh, rl, ap, voxel)
    with multiprocessing.Pool() as pool:
        pool.map(worker, timeframes)
    logger.info("Script finished successfully.")


@cli.command(help="Check dicom file metadata.")
@click.argument("path", required=True, type=click.Path())
def check(path):
    with open(path, "rb") as file:
        ds = dcmread(file)
        logger.info(f"File: {path}")
        logger.info(f"Axis: {ds[0x0008, 0x103E].value}")
        logger.info(f"Instance number: {ds[0x0020, 0x0013].value}")
        logger.info(f"Pixel spacing: {ds[0x0028, 0x0030].value} mm")
        logger.info(f"Image shape: {ds.pixel_array.shape}")
        logger.info(f"Slice location: {ds[0x0020, 0x1041].value}")
        logger.info(f"Spacing between slices: {ds[0x0018, 0x0088].value} mm")
        logger.info(f"Trigger time: {ds[0x0018, 0x1060].value} ms")


@cli.command(help="Remove exported data.")
def clean():
    path = "output"

    # check first if path exists
    if not os.path.exists(path):
        logger.error("Output files not exported yet.")
    else:
        try:
            os.rmdir(path)
            logger.info(f"Removed {path}")
        except OSError:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed {file_path}")
                except Exception as e:
                    logger.error(f"Error while deleting {file_path}: {e}")
            os.rmdir(path)
            logger.info(f"Removed {path}")
        except:
            logger.error("Output directory cannot be removed.")


@cli.command(help="Fix dicom series metadata.")
@click.argument("path", required=True, type=click.Path())
@click.option(
    "--instance",
    is_flag=True,
    help="Fix instance number of each frame.",
)
@click.option(
    "--channels",
    is_flag=True,
    help="Fix image channels.",
)
def fix(path, instance, channels):
    for idx, file in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as f:
            ds = dcmread(f)

            # fix instance number
            if instance:
                pre = ds[0x0020, 0x0013].value
                post = idx
                ds[0x0020, 0x0013].value = post
                logger.info(f"{file}: Changed instance number from {pre} to {post}.")

            if channels:
                ds.pixel_array = np.mean(ds.pixel_array, axis=2)
                logger.info(f"{file}: Fixed image channels.")

            # check output directory
            if not os.path.exists("output"):
                os.makedirs("output")

            ds.save_as(f"output/{idx:02d}.dcm")


@cli.command(help="Initialize input directory structure.")
@click.argument("path", default="files", type=click.Path())
def init(path):
    # check first if path exists
    if os.path.exists(path):
        logger.error(f"Directory already exists in {os.path.abspath(path)}")
    else:
        os.mkdir(path)
        # create fh directory
        fh_path = os.path.join(path, "FH")
        os.mkdir(fh_path)
        logger.info(f"Created FH directory in {fh_path}")

        # create ap directory
        ap_path = os.path.join(path, "AP")
        os.mkdir(ap_path)
        logger.info(f"Created AP directory in {ap_path}")

        # create rl directory
        rl_path = os.path.join(path, "RL")
        os.mkdir(rl_path)
        logger.info(f"Created Rl directory in {rl_path}")


if __name__ == "__main__":
    cli()
