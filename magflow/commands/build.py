import multiprocessing
from functools import partial
import numpy as np
import pydicom
import vtk
import csv

import typer
from typing_extensions import Annotated
import os

from magflow.utils.logger import logger
from rich.progress import Progress

app = typer.Typer()


def parse_dicom_files(input_dir=".tmp"):
    """
    Parse DICOM files from the nested directory structure.
    Directory structure:
    input_dir/
        timestep_dir/  (e.g., "000", "028", "057")
            axis_dir/  (e.g., "fh", "rl", "ap")
                dicom_files
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


def filter_by_time(series, frame):
    """Filter series by a specific time frame."""
    return [item["val"] for item in series if item["time"] == frame]


def tabulate(fh, rl, ap, voxel, time):
    """Convert 3D velocity data into tabulated format."""
    if not fh:
        logger.error("No image data available for tabulation.")
        return []

    dimensions = fh[0].shape
    res = []
    for z, (imgx, imgy, imgz) in enumerate(zip(ap, fh, rl)):
        for index, (pxlx, pxly, pxlz) in enumerate(
            zip(imgx[::-1].flatten(), imgy[::-1].flatten(), imgz[::-1].flatten())
        ):
            row = {}
            row["x"] = np.unravel_index(index, dimensions)[1] * voxel[0]
            row["y"] = np.unravel_index(index, dimensions)[0] * voxel[1]
            row["z"] = z * voxel[2]
            row["t"] = time
            row["vx"] = pxlx
            row["vy"] = pxly
            row["vz"] = pxlz
            res.append(row)
    return res


def mask(fh, rl, ap, mk):
    """Apply mask to velocity data."""
    masked_fh, masked_rl, masked_ap = [], [], []
    for imgx, imgy, imgz, imgm in zip(fh, rl, ap, mk):
        # apply mask in-place on copies if needed
        imgx_masked = imgx.copy()
        imgy_masked = imgy.copy()
        imgz_masked = imgz.copy()
        imgx_masked[imgm == 0] = 0
        imgy_masked[imgm == 0] = 0
        imgz_masked[imgm == 0] = 0

        masked_fh.append(imgy_masked)
        masked_rl.append(imgz_masked)
        masked_ap.append(imgx_masked)
    return masked_fh, masked_rl, masked_ap


def tocsv(data, time, output_dir="output"):
    """Export velocity data to CSV format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fields = data[0].keys()
    path = f"{output_dir}/data.csv.{time}"
    with open(path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return True


def tovtk(data, time, output_dir="output"):
    """Export velocity data to VTK format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points = vtk.vtkPoints()

    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Velocity")

    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName("Time")

    for point in data:
        points.InsertNextPoint(point["x"], point["y"], point["z"])

        vector = (point["vx"], point["vy"], point["vz"])
        vectors.InsertNextTuple(vector)

        scalar = point["t"]
        scalars.InsertNextTuple([scalar])

    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(128, 128, 40)
    sgrid.SetPoints(points)
    sgrid.GetPointData().SetVectors(vectors)
    sgrid.GetPointData().SetScalars(scalars)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(f"{output_dir}/data.vts.{time}")
    writer.SetInputData(sgrid)
    writer.Write()
    return True


def process_timeframe(raw, fh, rl, ap, voxel, time, output_dir="output"):
    """Process a single timeframe of velocity data."""
    try:
        # Filter data for this timeframe
        fh_filtered = filter_by_time(fh, time)
        rl_filtered = filter_by_time(rl, time)
        ap_filtered = filter_by_time(ap, time)

        # Convert to tabular format
        data = tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)

        # Export to requested format
        if raw:
            tocsv(data, time, output_dir)
        else:
            tovtk(data, time, output_dir)

        logger.info(f"Trigger time {time} exported with {len(data)} rows.")
        return True
    except Exception as e:
        logger.error(f"Failed processing timeframe {time}: {e}")
        return False


@app.command("build")
def build(
    raw: Annotated[bool, typer.Option(help="Export comma delimited values.")] = False,
    parallel: Annotated[
        bool, typer.Option(help="Activate multiprocessing mode.")
    ] = False,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for generated files.")
    ] = "output",
    workers: Annotated[
        int, typer.Option(help="Number of worker processes for parallel mode.")
    ] = None,
    input_dir: Annotated[
        str, typer.Option(help="Input directory containing DICOM files.")
    ] = ".tmp",
):
    """Create volumetric velocity field from dicom files."""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Load DICOM series using the new nested directory structure
        logger.info(f"Parsing DICOM files from {input_dir}...")
        try:
            axes_data = parse_dicom_files(input_dir)
        except FileNotFoundError as e:
            logger.error(f"Error loading DICOM files: {e}")
            raise typer.Exit(code=1)

        # Extract data for easier access
        fh, rl, ap = axes_data["fh"], axes_data["rl"], axes_data["ap"]

        # Validate data consistency
        if not (fh and rl and ap):
            logger.error("One or more series are empty. Cannot proceed.")
            raise typer.Exit(code=1)

        # Ensure all series have compatible dimensions
        try:
            dimensions = {
                "FH": (len(fh), fh[0]["pxl"].shape),
                "RL": (len(rl), rl[0]["pxl"].shape),
                "AP": (len(ap), ap[0]["pxl"].shape),
            }
            logger.info(f"Series dimensions: {dimensions}")
        except (IndexError, KeyError) as e:
            logger.error(f"Series data is malformed: {e}")
            raise typer.Exit(code=1)

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
            # Use process count optimized for CPU
            if workers is None:
                workers = max(1, multiprocessing.cpu_count() - 1)

            logger.info(f"Starting parallel processing with {workers} workers")

            # Create a partial function with all parameters except timeframe
            process_func = partial(
                process_timeframe, raw, fh, rl, ap, voxel, output_dir=output_dir
            )

            with multiprocessing.Pool(processes=workers) as pool:
                with Progress() as progress:
                    task = progress.add_task("Processing", total=len(timeframes))
                    for result in pool.imap_unordered(process_func, timeframes):
                        # Update progress based on success/failure
                        progress.update(task, advance=1)
                        if not result:
                            logger.warning("Failed processing a timeframe")

            logger.info("Parallel processing complete.")
        else:
            # Sequential processing with progress bar
            with Progress() as progress:
                task = progress.add_task("Processing", total=len(timeframes))
                for time in timeframes:
                    success = process_timeframe(
                        raw, fh, rl, ap, voxel, time, output_dir
                    )
                    progress.update(task, advance=1)
                    if not success:
                        logger.warning(f"Failed processing timeframe {time}")

            logger.info("Sequential processing complete.")

    except Exception as e:
        logger.error(f"Build process failed: {e}")
        raise typer.Exit(code=1)

    logger.info("Build process completed successfully.")
