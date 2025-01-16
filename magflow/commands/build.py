import multiprocessing
from functools import partial

import typer
from typing_extensions import Annotated

import magflow.utils as hf
from magflow.logger import logger

app = typer.Typer()


@app.command("build")
def build(
    raw: Annotated[bool, typer.Option(help="Export comma delimited values.")] = False,
    parallel: Annotated[
        bool, typer.Option(help="Activate multiprocessing mode.")
    ] = False,
):
    """
    Create volumetric velocity field from dicom files.
    """
    # create a list of dictionaries with the read data
    fh = hf.parse("FH")
    logger.info(f"FH series: {len(fh)} images.")
    rl = hf.parse("RL")
    logger.info(f"RL series: {len(rl)} images.")
    ap = hf.parse("AP")
    logger.info(f"AP series: {len(ap)} images.")

    # list unique trigger times
    timeframes = sorted(set(item["time"] for item in rl))
    logger.info(f"Timeframes: {timeframes}")

    # get volume dimensions
    volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

    # get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    if parallel:
        # map each timeframe to a different process
        worker = partial(hf.wrapper, raw, fh, rl, ap, voxel)
        with multiprocessing.Pool() as pool:
            pool.map(worker, timeframes)
        logger.info("Script finished successfully.")
    else:
        for time in timeframes:
            # filter data for a single timeframe
            fh_filtered = hf.filter(fh, time)
            rl_filtered = hf.filter(rl, time)
            ap_filtered = hf.filter(ap, time)

            data = hf.tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
            if raw:
                hf.tocsv(data, time)
            else:
                hf.tovtk(data, time)
            logger.info(f"Trigger time {time} exported with {len(data)} rows.")
