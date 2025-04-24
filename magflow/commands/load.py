from pathlib import Path

import pydicom
from pydicom.dataset import FileMetaDataset
import typer
from typing_extensions import Annotated

from magflow.logger import logger

app = typer.Typer()


@app.command("load")
def load(
    pathname: Annotated[Path, typer.Argument()],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = Path(".tmp"),
):
    """
    Load dicom image series from a directory and organize by timesteps and axes.

    This function processes DICOM files, extracting individual frames from multi-frame files,
    and organizes them in a directory structure based on cardiac timestep and anatomical axis.
    """
    files = list(pathname.iterdir())

    # Process each file in the directory
    for filename in files:
        # Skip non-file items or files without .dcm extension
        if not filename.is_file() or filename.suffix.lower() != ".dcm":
            continue

        # Guess the axis based on filename (fh: foot-head, ap: anterior-posterior, rl: right-left)
        axis_value = None
        for candidate in ["fh", "ap", "rl"]:
            if candidate in filename.name.lower():
                axis_value = candidate
                break
        if axis_value is None:
            logger.warning(f"Axis not determined for file {filename.name}. Skipping.")
            continue

        # Read the DICOM file
        try:
            dcm = pydicom.dcmread(filename)
        except pydicom.errors.InvalidDicomError:
            logger.warning(f"{filename.name} is not a valid DICOM file.")
            continue

        # Validate multi-frame DICOM
        if not hasattr(dcm, "NumberOfFrames"):
            logger.warning(f"{filename.name} missing necessary DICOM attributes.")
            continue

        # Load and extract pixel data from the DICOM file
        try:
            img = dcm.pixel_array
            logger.info(f"DICOM pixel array dimensions: {img.shape}")
        except Exception as e:
            logger.warning(f"Error accessing pixel data in {filename.name}: {e}")
            continue

        # Process each frame in the multi-frame DICOM
        for idx in range(dcm.NumberOfFrames):
            # Create a new dataset for the single frame
            frame = pydicom.dataset.Dataset()

            # Get per-frame metadata
            try:
                pffgs = dcm.PerFrameFunctionalGroupsSequence[idx]
            except IndexError:
                logger.warning(f"Frame {idx} missing in {filename.name}.")
                continue

            # Extract cardiac phase information (timing)
            timestep = 0
            if (0x0018, 0x9118) in pffgs:  # Cardiac Triggering Sequence
                if "NominalCardiacTriggerDelayTime" in pffgs[(0x0018, 0x9118)][0]:
                    delay_time = pffgs[(0x0018, 0x9118)][0][
                        "NominalCardiacTriggerDelayTime"
                    ].value
                    timestep = int(delay_time)

            # Create directory structure: timestep/axis/
            timestep_dir = output_dir / f"{timestep:03d}"
            dst_dir = timestep_dir / axis_value
            dst_dir.mkdir(parents=True, exist_ok=True)
            target = dst_dir / f"img{idx:04}.dcm"

            # Copy important metadata from original DICOM
            for group in [
                0x0008,
                0x0010,
                0x0018,
                0x0020,
                0x0028,
            ]:
                for elem in dcm:
                    if elem.tag.group == group:
                        frame.add(elem)

            # Copy pixel value transformation metadata
            if (0x0028, 0x9145) in pffgs:  # Pixel Value Transformation Sequence
                for tag_name in ["RescaleIntercept", "RescaleSlope", "RescaleType"]:
                    if tag_name in pffgs[(0x0028, 0x9145)][0]:
                        value = pffgs[(0x0028, 0x9145)][0][tag_name].value
                        setattr(frame, tag_name, value)

            # Copy geometric metadata
            if (0x0028, 0x9110) in pffgs:  # Plane Position Sequence
                for tag_name in [
                    "SpacingBetweenSlices",
                    "PixelSpacing",
                    "SliceThickness",
                ]:
                    if tag_name in pffgs[(0x0028, 0x9110)][0]:
                        value = pffgs[(0x0028, 0x9110)][0][tag_name].value
                        setattr(frame, tag_name, value)

            # Copy cardiac timing metadata
            if (0x0018, 0x9118) in pffgs:  # Cardiac Triggering Sequence
                for tag_name in ["NominalCardiacTriggerDelayTime"]:
                    if tag_name in pffgs[(0x0018, 0x9118)][0]:
                        value = pffgs[(0x0018, 0x9118)][0][tag_name].value
                        setattr(frame, tag_name, value)

            # Remove multiframe-specific attributes
            if hasattr(frame, "NumberOfFrames"):
                del frame.NumberOfFrames

            # Update attributes specific to single-frame
            frame.InstanceNumber = idx + 1

            # Set the pixel data for this frame
            frame.PixelData = img[idx].tobytes()

            # Update image type to indicate single frame
            if "ImageType" in frame:
                image_type_list = list(frame.ImageType)
                if "MULTIFRAME" in image_type_list:
                    image_type_list.remove("MULTIFRAME")
                frame.ImageType = image_type_list

            # Copy the file meta information
            file_meta = pydicom.dataset.FileMetaDataset()
            for elem in dcm.file_meta:
                file_meta.add(elem)

            # Create a new SOP Instance UID for each frame
            frame.SOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.MediaStorageSOPInstanceUID = frame.SOPInstanceUID

            # Create the final single-frame DICOM
            output = pydicom.dataset.FileDataset(
                filename_or_obj="",
                dataset=frame,
                file_meta=file_meta,
                preamble=dcm.preamble,
            )
            output.save_as(target, write_like_original=False)

    logger.info(f"DICOM files organized by timestep and axis in {output_dir}")
