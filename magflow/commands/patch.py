import os
from pathlib import Path

import numpy as np
import pydicom as pd
import typer
from typing_extensions import Annotated

from magflow.logger import logger

app = typer.Typer()


@app.command(help="Patch dicom series metadata.")
def patch(
    path: Annotated[Path, typer.Argument(help="Path to DICOM file.")],
    instance: Annotated[
        bool, typer.Option(help="Patch instance number of each frame.")
    ],
    channels: Annotated[bool, typer.Option(help="Patch image channels.")],
):
    """
    Patch dicom series metadata.
    """
    for idx, file in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as f:
            ds = pd.dcmread(f)

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

            ds.save_as(f"output/{idx:04}.dcm")
