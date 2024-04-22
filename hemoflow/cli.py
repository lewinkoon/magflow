import click
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger
import os


@click.group(
    epilog="Check out readme at https://github.com/lewinkoon/hemoflow for more details."
)
def cli():
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
    pass


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


@cli.command(help="Convert a dicom sequence to multiple dicoms.")
@click.argument("path", required=True, type=click.Path())
def fix(path):
    logger.info(f"Fixed dicom file from {path}")


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


if __name__ == "__main__":
    cli()
