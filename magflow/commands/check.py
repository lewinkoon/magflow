import random
from pathlib import Path

import pydicom as pd
import typer
from typing_extensions import Annotated

import magflow.utils.utils as hf
from magflow.utils.logger import logger

app = typer.Typer()


@app.command("check")
def check(input_path: Annotated[Path, typer.Argument(help="Path to DICOM file.")]):
    """
    Check dicom file metadata.
    """
    if input_path.is_file():
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
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.is_file()]
        if not files:
            logger.info("The directory contains no files.")
        else:
            random_file = random.choice(files)
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
