import shutil
from pathlib import Path

import pydicom as pd
import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn

from magflow.logger import logger

app = typer.Typer()


@app.command("load")
def load(
    pathname: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path(".tmp"),
):
    """
    Load dicom image series from a directory.
    """
    files = list(pathname.iterdir())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        for filename in files:  # replaced track loop
            if not filename.is_file() or filename.suffix.lower() != ".dcm":
                continue

            # Guess the axis based on filename.
            axis_value = None
            for candidate in ["fh", "ap", "rl"]:
                if candidate in filename.name.lower():
                    axis_value = candidate
                    break
            if axis_value is None:
                logger.warning(f"Axis not determined for file {filename}. Skipping.")
                continue

            task = progress.add_task(f"Loading {axis_value} files...", total=None)
            dst_dir = output_dir / axis_value
            dst_dir.mkdir(parents=True, exist_ok=True)

            try:
                ds = pd.dcmread(filename)  # use dcmread with Path directly
            except pd.errors.InvalidDicomError:
                logger.warning(f"{filename} is not a valid DICOM file.")
                continue

            if not hasattr(ds, "NumberOfFrames") or not hasattr(
                ds, "PerFrameFunctionalGroupsSequence"
            ):
                logger.warning(f"{filename} missing necessary DICOM attributes.")
                continue

            n = int(ds.NumberOfFrames)
            # logger.info(f"Detected {n} frames in file {filename}.")

            try:
                img = ds.pixel_array.squeeze()
            except Exception as e:
                logger.warning(f"Error accessing pixel data in {filename}: {e}")
                continue

            essentials = [0x0008, 0x0010, 0x0018, 0x0020, 0x0028]

            for idx in range(n):
                # Ensure PerFrameFunctionalGroupsSequence has the required index.
                try:
                    pffgs = ds.PerFrameFunctionalGroupsSequence[idx]
                except IndexError:
                    logger.warning(f"Frame {idx} missing in {filename}.")
                    continue

                target = dst_dir / f"img{idx:04}.dcm"

                file_meta = pd.dataset.FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pd.uid.MRImageStorage
                file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
                file_meta.TransferSyntaxUID = pd.uid.ImplicitVRLittleEndian

                tmp_ds = pd.dataset.Dataset()
                tmp_ds.file_meta = file_meta
                tmp_ds.is_little_endian = True
                tmp_ds.is_implicit_VR = True

                for group in essentials:
                    for elem in ds:
                        if elem.tag.group == group:
                            tmp_ds.add(elem)

                for tag_name in ["RescaleIntercept", "RescaleSlope", "RescaleType"]:
                    if tag_name in pffgs.get((0x0028, 0x9145), [{}])[0]:
                        value = pffgs[(0x0028, 0x9145)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                for tag_name in [
                    "SpacingBetweenSlices",
                    "PixelSpacing",
                    "SliceThickness",
                ]:
                    if tag_name in pffgs.get((0x0028, 0x9110), [{}])[0]:
                        value = pffgs[(0x0028, 0x9110)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                for tag_name in ["NominalCardiacTriggerDelayTime"]:
                    if tag_name in pffgs.get((0x0018, 0x9118), [{}])[0]:
                        value = pffgs[(0x0018, 0x9118)][0][tag_name].value
                        setattr(tmp_ds, tag_name, value)

                if hasattr(tmp_ds, "NumberOfFrames"):
                    del tmp_ds.NumberOfFrames
                tmp_ds.InstanceNumber = idx + 1
                tmp_ds.PixelData = img[idx, :].squeeze().tobytes()
                tmp_ds.save_as(target, write_like_original=False)
                # Update progress after exporting each frame
                progress.advance(task, advance=1)
