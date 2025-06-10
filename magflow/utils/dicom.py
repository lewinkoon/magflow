from pathlib import Path

import numpy as np
import pydicom
import typer
from rich.progress import Progress

import magflow.utils.data as dat
from magflow.utils.logger import logger


def get_axis_from_path(path):
    """
    Determine the axis from the filename or directory path.
    """
    # Convert Path object to string for checking
    path_str = str(path).lower()

    for candidate in ["fh", "ap", "rl", "m"]:
        if candidate in path_str:
            return candidate
    return None


def get_cardiac_timestep(pffgs):
    """
    Extract the cardiac phase information (timing) from the DICOM file.
    """
    timestep = 0
    if (0x0018, 0x9118) in pffgs and "NominalCardiacTriggerDelayTime" in pffgs[
        (0x0018, 0x9118)
    ][0]:  # Cardiac Triggering Sequence
        delay_time = pffgs[(0x0018, 0x9118)][0]["NominalCardiacTriggerDelayTime"].value
        timestep = int(delay_time)
    return timestep


def copy_dicom_metadata(frame, dcm, pffgs):
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


def extract_single_frame(dcm, idx, img, axis_value, output_dir):
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
    timestep = get_cardiac_timestep(pffgs)
    timestep_dir = output_dir / f"{timestep:03d}"
    dst_dir = timestep_dir / axis_value
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / f"img{idx:04}.dcm"

    # Copy and set metadata
    frame = copy_dicom_metadata(frame, dcm, pffgs)

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


def organize_single_dicom(filename, output_dir, progress=None, task_id=None):
    """Process a single-frame DICOM file, organizing by series and timestep."""
    # Read the DICOM file
    try:
        dcm = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        logger.warning(f"{filename.name} is not a valid DICOM file.")
        return 0

    # Extract series information from DICOM tags or parent directory
    try:
        # Try to get series description or use parent directory name as axis
        if hasattr(dcm, "SeriesDescription"):
            series_desc = dcm.SeriesDescription.lower()
            axis_value = None
            for candidate in ["fh", "ap", "rl", "m"]:
                if candidate == series_desc:
                    axis_value = candidate
                    break

        if axis_value is None:
            logger.warning(f"Axis not determined for file {filename.name}. Skipping.")
            return 0

        # Extract timestep information
        timestep = 0
        if hasattr(dcm, "TriggerTime"):
            timestep = int(dcm.TriggerTime)
        elif hasattr(dcm, "ContentTime"):
            # Use content time as fallback
            time_str = str(dcm.ContentTime).split(".")[0]
            if len(time_str) >= 6:
                timestep = int(time_str[-2:])  # Use last 2 digits as timestep
        elif hasattr(dcm, "InstanceNumber"):
            timestep = dcm.InstanceNumber

        # Create output directory structure
        timestep_dir = output_dir / f"{timestep:03d}"
        dst_dir = timestep_dir / axis_value
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        instance_num = getattr(dcm, "InstanceNumber", 1)
        target = dst_dir / f"img{instance_num:04}.dcm"

        # Copy the file to the organized structure
        # You can either copy as-is or modify metadata
        dcm.save_as(target, write_like_original=False)

        logger.info(f"Processed single-frame DICOM: {filename.name} -> {target}")
        return 1

    except Exception as e:
        logger.error(f"Error processing single-frame DICOM {filename.name}: {e}")
        return 0


def process_dicom(filename, output_dir, progress=None, task_id=None):
    """Process a DICOM file, handling both multi-frame and single-frame files."""
    # Read the DICOM file first to determine type
    try:
        dcm = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        logger.warning(f"{filename.name} is not a valid DICOM file.")
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        return 0

    # Check if it's a multi-frame DICOM
    if hasattr(dcm, "NumberOfFrames") and dcm.NumberOfFrames > 1:
        # Process as multi-frame DICOM (existing logic)
        return extract_multiframe_dicom(filename, output_dir, progress, task_id)
    else:
        # Process as single-frame DICOM
        frames_processed = organize_single_dicom(
            filename, output_dir, progress, task_id
        )
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        return frames_processed


def extract_multiframe_dicom(filename, output_dir, progress=None, task_id=None):
    """Process a multi-frame DICOM file, extracting all frames."""
    # Read the DICOM file
    try:
        dcm = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        logger.warning(f"{filename.name} is not a valid DICOM file.")
        return 0

    # Initialize axis_value
    axis_value = None

    # Determine axis from DICOM SeriesDescription first
    if dcm and hasattr(dcm, "SeriesDescription"):
        series_desc = dcm.SeriesDescription.upper().strip()
        if series_desc in ["FH", "AP", "RL", "M"]:
            axis_value = series_desc.lower()

    # If still not found, skip this file
    if axis_value is None:
        logger.warning(f"Axis not determined for file {filename.name}. Skipping.")
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
        if extract_single_frame(dcm, idx, img, axis_value, output_dir):
            frames_processed += 1
        if progress:
            progress.update(frames_task_id, advance=1)

    if progress and task_id is not None:
        progress.update(task_id, advance=1)

    return frames_processed


def parse_dicom_directory(input_dir=".tmp"):
    """
    Parse DICOM files from the nested directory structure.
    """
    result = {"fh": [], "rl": [], "ap": []}

    # Check if input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        raise FileNotFoundError(f"Input directory {input_dir} not found.")

    # Get all timestep directories
    timestep_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if not timestep_dirs:
        logger.error(f"No timestep directories found in {input_dir}")
        raise FileNotFoundError(f"No timestep directories found in {input_dir}")

    logger.info(f"Found {len(timestep_dirs)} timestep directories")

    # Process each timestep directory
    for timestep_dir in timestep_dirs:
        # Find axis directories in this timestep
        for axis in ["fh", "rl", "ap"]:
            axis_path = timestep_dir / axis

            if not axis_path.exists():
                logger.warning(
                    f"Axis directory {axis_path} not found for timestep {timestep_dir.name}"
                )
                continue

            # Process all DICOM files in this axis directory
            dicom_files = [
                f
                for f in axis_path.iterdir()
                if f.name.lower().endswith(".dcm") or f.is_file()
            ]

            if not dicom_files:
                logger.warning(f"No DICOM files found in {axis_path}")
                continue

            for file_path in dicom_files:
                try:
                    slice_data = {}
                    with file_path.open("rb") as binary_file:
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

                        # Extract time
                        time_value = 0
                        try:
                            # First try: Nominal Cardiac Trigger Delay Time (0020,9153)
                            if (0x0020, 0x9153) in ds:
                                time_value = int(ds[0x0020, 0x9153].value)
                            # Second try: Trigger Time (0018,1060)
                            elif (0x0018, 0x1060) in ds:
                                time_value = int(float(ds[0x0018, 0x1060].value))

                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Error extracting time from {file_path}: {e}, using 0"
                            )
                            time_value = 0
                        slice_data["time"] = time_value

                        slice_data["timestep_dir"] = (
                            timestep_dir.name  # store the timestep directory name
                        )

                        result[axis].append(slice_data)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

    # Log the results
    for axis, data in result.items():
        logger.info(f"{axis.upper()} series: {len(data)} images found")

    return result


def find_dicom_files(pathname: Path, recursive: bool) -> list[Path]:
    """Discover DICOM files in the given path."""
    if recursive:
        files = [f for f in pathname.rglob("*") if f.is_file()]
    else:
        files = [
            f for f in pathname.iterdir() if f.is_file() and f.suffix.lower() == ".dcm"
        ]

    if not files:
        logger.error(f"No DICOM files found in {pathname}")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(files)} DICOM files")
    return files


def load_and_validate_data(temp_dir: Path) -> dict:
    """Extract DICOM images and validate data consistency."""
    output_path = Path("data")  # This should be passed as parameter
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Parsing DICOM files from {temp_dir}...")
    try:
        axes_data = parse_dicom_directory(temp_dir)
    except FileNotFoundError as e:
        logger.error(f"Error loading DICOM files: {e}")
        raise typer.Exit(code=1) from e

    fh, rl, ap = axes_data["fh"], axes_data["rl"], axes_data["ap"]

    if not (fh and rl and ap):
        logger.error("One or more series are empty. Cannot proceed.")
        raise typer.Exit(code=1)

    # Log metadata
    timeframes = sorted({item["time"] for item in rl})
    logger.info(f"Timeframes: {timeframes}")

    volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    return axes_data


def show_dicom_tag(dataset, group, element):
    try:
        tag_name = dataset[group, element].name
        tag_value = dataset[group, element].value
        logger.info(f"{tag_name}: {tag_value}")
    except KeyError:
        logger.error(f"[{group:04x},{element:04x}]: Not found.")


def validate_axis_data(axes_data: dict) -> tuple:
    """Validate and extract axes data."""
    fh, rl, ap = axes_data["fh"], axes_data["rl"], axes_data["ap"]

    if not (fh and rl and ap):
        logger.error("One or more series are empty. Cannot proceed.")
        raise typer.Exit(code=1)

    return fh, rl, ap


def get_volume_metadata(fh: list) -> tuple:
    """Extract volume dimensions and voxel spacing from DICOM data."""
    # Get volume dimensions
    volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

    # Get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    return volume, voxel


def _export_timeframe_data(
    time: int,
    fh: list,
    rl: list,
    ap: list,
    voxel: tuple,
    output_dir: str,
    raw: bool,
    progress: Progress,
    subtask_id: int,
) -> bool:
    """Process a single timeframe and export data."""
    try:
        # Filter data for this timeframe
        progress.update(
            subtask_id, description=f"[cyan]Timeframe {time:03d} (Filtering)"
        )
        fh_filtered = [item["val"] for item in fh if item["time"] == time]
        rl_filtered = [item["val"] for item in rl if item["time"] == time]
        ap_filtered = [item["val"] for item in ap if item["time"] == time]

        # Check that filtering produced valid data
        if not (fh_filtered and rl_filtered and ap_filtered):
            logger.warning(f"No data found for timeframe {time}. Skipping.")
            return False

        progress.update(subtask_id, advance=1)

        # Convert to tabular format
        progress.update(
            subtask_id, description=f"[cyan]Timeframe {time:03d} (Tabulating)"
        )
        data = dat.tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
        progress.update(subtask_id, advance=1)

        # Export to requested format
        progress.update(
            subtask_id, description=f"[cyan]Timeframe {time:03d} (Exporting)"
        )
        if raw:
            dat.tocsv(data, time, output_dir)
        else:
            dat.tovtk(data, time, output_dir)

        logger.info(f"Trigger time {time} exported with {len(data)} rows.")
        return True

    except Exception as e:
        logger.error(f"Failed processing timeframe {time}: {e}")
        return False


def batch_process_dicoms(files: list[Path], temp_dir: Path) -> int:
    """Process all DICOM files and return total frames processed."""
    total_frames = 0

    with Progress() as progress:
        file_task = progress.add_task(
            "[bold green]Processing DICOM files...", total=len(files)
        )

        for filename in files:
            frames_processed = process_dicom(filename, temp_dir, progress, file_task)
            total_frames += frames_processed

    logger.info(f"Total frames processed: {total_frames}")
    return total_frames


def export_all_timeframes(
    timeframes: list,
    fh: list,
    rl: list,
    ap: list,
    voxel: tuple,
    output_dir: str,
    raw: bool,
) -> None:
    """Process all timeframes with progress tracking."""
    with Progress() as progress:
        overall_task = progress.add_task(
            "[bold green]Overall Progress", total=len(timeframes)
        )

        for time in timeframes:
            subtask = progress.add_task(f"[cyan]Timeframe {time:03d}", total=3)

            success = _export_timeframe_data(
                time, fh, rl, ap, voxel, output_dir, raw, progress, subtask
            )

            # Complete the subtask and advance overall progress
            progress.update(subtask, completed=3)
            progress.update(overall_task, advance=1)

            if not success:
                logger.warning(f"Failed processing timeframe {time:03d}")

    logger.info("Sequential processing complete.")
