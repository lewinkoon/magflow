import os
import shutil
from enum import Enum
from pathlib import Path

import pydicom as pd
import typer
from typing_extensions import Annotated

from magflow.logger import logger

app = typer.Typer()


class Axis(str, Enum):
    fh = "fh"
    rl = "rl"
    ap = "ap"


@app.command("load")
def load(
    pathname: Annotated[Path, typer.Argument()],
    axis: Annotated[Axis, typer.Option(case_sensitive=False, help="Flow axis.")],
    multiframe: Annotated[bool, typer.Option(help="Fix multiframe dicom files.")],
):
    """
    Load dicom image series.
    """
    dst_dir = f"files/{axis}"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if multiframe:
        with open(pathname, "rb") as f:
            ds = pd.dcmread(f)
            n = int(ds.NumberOfFrames)
            logger.info(f"Detected {n} frames.")

            img = ds.pixel_array.squeeze()

            essentials = [0x0008, 0x0010, 0x0018, 0x0020, 0x0028]

            for idx in range(n):
                target = os.path.join(f"files/{axis}", f"img{idx:04}.dcm")

                file_meta = pd.dataset.FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pd.uid.MRImageStorage
                file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
                file_meta.TransferSyntaxUID = pd.uid.ImplicitVRLittleEndian

                tmp_ds = pd.dataset.Dataset()
                tmp_ds.file_meta = file_meta
                tmp_ds.is_little_endian = (
                    tmp_ds.file_meta.TransferSyntaxUID.is_little_endian
                )
                tmp_ds.is_implicit_VR = (
                    tmp_ds.file_meta.TransferSyntaxUID.is_implicit_VR
                )

                for group in essentials:
                    for key, value in ds.group_dataset(group).items():
                        tmp_ds[key] = value

                # sfgs = ds.SharedFunctionalGroupsSequence[0]
                pffgs = ds.PerFrameFunctionalGroupsSequence[idx]

                # copy velocity tags
                for tag_name in ["RescaleIntercept", "RescaleSlope", "RescaleType"]:
                    if tag_name in pffgs[(0x0028, 0x9145)][0]:
                        value = pffgs[(0x0028, 0x9145)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                # copy velocity tags
                for tag_name in [
                    "SpacingBetweenSlices",
                    "PixelSpacing",
                    "SliceThickness",
                ]:
                    if tag_name in pffgs[(0x0028, 0x9110)][0]:
                        value = pffgs[(0x0028, 0x9110)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                # copy trigger time
                for tag_name in ["NominalCardiacTriggerDelayTime"]:
                    if tag_name in pffgs[(0x0018, 0x9118)][0]:
                        value = pffgs[(0x0018, 0x9118)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                del tmp_ds.NumberOfFrames
                tmp_ds.InstanceNumber = idx + 1
                tmp_ds.PixelData = img[idx, :].squeeze().tobytes()
                tmp_ds.save_as(target, write_like_original=False)

                logger.info(f"Image exported as {target}")
    else:
        for idx, filename in enumerate(os.listdir(pathname)):
            src_file = os.path.join(pathname, filename)
            dst_file = os.path.join(dst_dir, f"img{idx:04}.dcm")

            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)
                logger.info(f"Copied {src_file} to {dst_file}.")
