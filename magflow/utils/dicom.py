import pydicom
from magflow.utils.logger import logger
import os
import numpy as np


def determine_axis(filename):
    """
    Determine the axis from the filename.
    """
    for candidate in ["fh", "ap", "rl", "m"]:
        if candidate in filename.name.lower():
            return candidate
    return None


def extract_timestep(pffgs):
    """
    Extract the cardiac phase information (timing) from the DICOM file.
    """
    timestep = 0
    if (0x0018, 0x9118) in pffgs:  # Cardiac Triggering Sequence
        if "NominalCardiacTriggerDelayTime" in pffgs[(0x0018, 0x9118)][0]:
            delay_time = pffgs[(0x0018, 0x9118)][0][
                "NominalCardiacTriggerDelayTime"
            ].value
            timestep = int(delay_time)
    return timestep


def copy_metadata(frame, dcm, pffgs):
    """
    Copy important metadata from the original DICOM to the new frame.
    """
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

    # Set image orientation to sagittal (right-left view)
    frame.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    frame.ImagePositionPatient = [0, 0, 0]

    # Update image type to indicate single frame
    if "ImageType" in frame:
        image_type_list = list(frame.ImageType)
        if "MULTIFRAME" in image_type_list:
            image_type_list.remove("MULTIFRAME")
        frame.ImageType = image_type_list

    return frame


def process_frame(dcm, idx, img, axis_value, output_dir):
    """Process a single frame from a multi-frame DICOM file."""
    # Create a new dataset for the single frame
    frame = pydicom.dataset.Dataset()

    # Get per-frame metadata
    try:
        pffgs = dcm.PerFrameFunctionalGroupsSequence[idx]
    except IndexError:
        logger.warning(f"Frame {idx} missing in DICOM.")
        return False

    # Extract cardiac phase information and determine output location
    timestep = extract_timestep(pffgs)
    timestep_dir = output_dir / f"{timestep:03d}"
    dst_dir = timestep_dir / axis_value
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / f"img{idx:04}.dcm"

    # Copy and set metadata
    frame = copy_metadata(frame, dcm, pffgs)

    # Remove multiframe-specific attributes
    if hasattr(frame, "NumberOfFrames"):
        del frame.NumberOfFrames

    # Update attributes specific to single-frame
    frame.InstanceNumber = idx + 1
    frame.PixelData = img[idx].tobytes()

    # Create new SOP Instance UID and file metadata
    frame.SOPInstanceUID = pydicom.uid.generate_uid()
    file_meta = pydicom.dataset.FileMetaDataset()
    for elem in dcm.file_meta:
        file_meta.add(elem)
    file_meta.MediaStorageSOPInstanceUID = frame.SOPInstanceUID

    # Create the final single-frame DICOM
    output = pydicom.dataset.FileDataset(
        filename_or_obj="",
        dataset=frame,
        file_meta=file_meta,
        preamble=dcm.preamble,
    )
    output.save_as(target, write_like_original=False)
    return True


def process_file(filename, output_dir, progress=None, task_id=None):
    """Process a single DICOM file, extracting all frames."""
    # Determine axis from filename
    axis_value = determine_axis(filename)
    if axis_value is None:
        logger.warning(f"Axis not determined for file {filename.name}. Skipping.")
        return 0

    # Read the DICOM file
    try:
        dcm = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        logger.warning(f"{filename.name} is not a valid DICOM file.")
        return 0

    # Validate multi-frame DICOM
    if not hasattr(dcm, "NumberOfFrames"):
        logger.warning(f"{filename.name} missing necessary DICOM attributes.")
        return 0

    # Load and extract pixel data from the DICOM file
    try:
        img = dcm.pixel_array
        logger.info(f"Pixel array dimensions in {filename.name}: {img.shape}")
    except Exception as e:
        logger.warning(f"Error accessing pixel data in {filename.name}: {e}")
        return 0

    # Process each frame
    frames_processed = 0

    # Create nested progress bar for frame extraction if a progress context is provided
    if progress:
        frames_task_id = progress.add_task(
            f"[cyan]Extracting frames from {filename.name}...", total=dcm.NumberOfFrames
        )

    for idx in range(dcm.NumberOfFrames):
        if process_frame(dcm, idx, img, axis_value, output_dir):
            frames_processed += 1
        if progress:
            progress.update(frames_task_id, advance=1)

    if progress and task_id is not None:
        progress.update(task_id, advance=1)

    return frames_processed


def parse_dicom_files(input_dir=".tmp"):
    """
    Parse DICOM files from the nested directory structure.
    """
    result = {"fh": [], "rl": [], "ap": []}

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        raise FileNotFoundError(f"Input directory {input_dir} not found.")

    # Get all timestep directories
    timestep_dirs = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )

    if not timestep_dirs:
        logger.error(f"No timestep directories found in {input_dir}")
        raise FileNotFoundError(f"No timestep directories found in {input_dir}")

    logger.info(f"Found {len(timestep_dirs)} timestep directories")

    # Process each timestep directory
    for timestep_dir in timestep_dirs:
        timestep_path = os.path.join(input_dir, timestep_dir)

        # Find axis directories in this timestep
        for axis in ["fh", "rl", "ap"]:
            axis_path = os.path.join(timestep_path, axis)

            if not os.path.exists(axis_path):
                logger.warning(
                    f"Axis directory {axis_path} not found for timestep {timestep_dir}"
                )
                continue

            # Process all DICOM files in this axis directory
            dicom_files = [
                f
                for f in os.listdir(axis_path)
                if f.lower().endswith(".dcm")
                or os.path.isfile(os.path.join(axis_path, f))
            ]

            if not dicom_files:
                logger.warning(f"No DICOM files found in {axis_path}")
                continue

            for filename in dicom_files:
                file_path = os.path.join(axis_path, filename)
                try:
                    slice_data = {}
                    with open(file_path, "rb") as binary_file:
                        ds = pydicom.dcmread(binary_file)

                        # assign image array
                        if ds[0x0028, 0x0004].value == "MONOCHROME2":
                            img = ds.pixel_array
                        else:
                            img = np.mean(ds.pixel_array, axis=2)
                        slice_data["pxl"] = img
                        slice_data["val"] = pydicom.pixels.apply_modality_lut(img, ds)

                        # assign tags
                        slice_data["axis"] = axis  # use directory name as axis
                        slice_data["num"] = ds[0x0020, 0x0013].value  # instance number
                        slice_data["spacing"] = ds[
                            0x0028, 0x0030
                        ].value  # pixel spacing
                        slice_data["height"] = ds[
                            0x0018, 0x0088
                        ].value  # spacing between slices
                        slice_data["time"] = int(
                            ds[0x0020, 0x9153].value
                        )  # trigger time
                        slice_data["timestep_dir"] = (
                            timestep_dir  # store the timestep directory name
                        )

                        result[axis].append(slice_data)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

    # Log the results
    for axis, data in result.items():
        logger.info(f"{axis.upper()} series: {len(data)} images found")

    return result


def showtag(dataset, group, element):
    try:
        tag_name = dataset[group, element].name
        tag_value = dataset[group, element].value
        logger.info(f"{tag_name}: {tag_value}")
    except KeyError:
        logger.error(f"[{group:04x},{element:04x}]: Not found.")
