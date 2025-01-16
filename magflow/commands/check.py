import os
import random
from pathlib import Path

import pydicom as pd
import typer
from typing_extensions import Annotated

import magflow.utils as hf
from magflow.logger import logger

app = typer.Typer()


@app.command("check")
def check(input_path: Annotated[Path, typer.Argument(help="Path to DICOM file.")]):
    """
    Check dicom file metadata.
    """
    if os.path.isfile(input_path):
        with open(input_path, "rb") as f:
            ds = pd.dcmread(f)

            logger.info(f"File: {input_path}")
            logger.info(f"Image shape: {ds.pixel_array.shape}")

            hf.showtag(ds, 0x0008, 0x103E)  # axis name
            hf.showtag(ds, 0x0020, 0x0013)  # instance number
            hf.showtag(ds, 0x0028, 0x0030)  # pixel spacing
            hf.showtag(ds, 0x0018, 0x0088)  # spacing between slices
            hf.showtag(ds, 0x0020, 0x1041)  # slice location
            hf.showtag(ds, 0x0018, 0x1060)  # trigger time
    elif os.path.isdir(input_path):
        files = [
            f
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
        ]
        if not files:
            logger.info("The directory contains no files.")
        else:
            random_file = os.path.join(input_path, random.choice(files))
            with open(random_file, "rb") as f:
                ds = pd.dcmread(f)

                logger.info(f"File: {random_file}")
                logger.info(f"Image shape: {ds.pixel_array.shape}")

                hf.showtag(ds, 0x0008, 0x103E)  # axis name
                hf.showtag(ds, 0x0020, 0x0013)  # instance number
                hf.showtag(ds, 0x0028, 0x0030)  # pixel spacing
                hf.showtag(ds, 0x0018, 0x0088)  # spacing between slices
                hf.showtag(ds, 0x0020, 0x1041)  # slice location
                hf.showtag(ds, 0x0018, 0x1060)  # trigger time
    else:
        logger.info("The provided path is neither a file nor a directory.")
