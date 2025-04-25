import csv
import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
import vtk
from magflow.utils.logger import logger
import zipfile
import io
import pydicom as pd


def tabulate(fh, rl, ap, voxel, time):
    """Convert 3D velocity data into tabulated format."""
    if not fh:
        logger.error("No image data available for tabulation.")
        return []

    if not len(fh) == len(rl) == len(ap):
        logger.error(f"Dimension mismatch: FH={len(fh)}, RL={len(rl)}, AP={len(ap)}")
        return []

    dimensions = fh[0].shape
    res = []
    for z, (imgx, imgy, imgz) in enumerate(zip(ap, fh, rl)):
        # Verify dimensions match
        if not imgx.shape == imgy.shape == imgz.shape:
            logger.warning(f"Slice {z} has mismatched dimensions. Skipping.")
            continue

        x_flat = imgx[::-1].flatten()
        y_flat = imgy[::-1].flatten()
        z_flat = imgz[::-1].flatten()

        for index in range(len(x_flat)):
            row = {}
            row["x"] = np.unravel_index(index, dimensions)[1] * voxel[0]
            row["y"] = np.unravel_index(index, dimensions)[0] * voxel[1]
            row["z"] = z * voxel[2]
            row["t"] = time
            row["vx"] = x_flat[index]
            row["vy"] = y_flat[index]
            row["vz"] = z_flat[index]
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


def create_zip_buffer(directory_path):
    """Create a zip file in memory from the directory contents."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add all DICOM files to the zip, maintaining directory structure
        for file_path in directory_path.glob("**/*.dcm"):
            relative_path = file_path.relative_to(directory_path)
            zip_file.write(file_path, arcname=relative_path)

    # Reset buffer position to start
    zip_buffer.seek(0)
    return zip_buffer


def determine_axis(filename):
    """Determine the directional axis based on filename."""
    for axis in ["fh", "rl", "ap"]:
        if axis in filename.lower():
            return axis
    return None


def load_dicom_file(file):
    """Load a DICOM file and validate it has required attributes."""
    try:
        ds = pd.dcmread(file)

        # Check for required multi-frame DICOM attributes
        if not hasattr(ds, "NumberOfFrames") or not hasattr(
            ds, "PerFrameFunctionalGroupsSequence"
        ):
            return None, f"{file.name} missing necessary DICOM attributes."

        return ds, None
    except pd.errors.InvalidDicomError:
        return None, f"{file.name} is not a valid DICOM file."
    except Exception as e:
        return None, f"Error reading {file.name}: {str(e)}"


def create_frame_dataset(original_ds, pffgs, frame_idx, pixel_data, essential_groups):
    """Create a new DICOM dataset for a single frame."""
    # Create DICOM file metadata
    file_meta = pd.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pd.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
    file_meta.TransferSyntaxUID = pd.uid.ImplicitVRLittleEndian

    # Create a new DICOM dataset for this frame
    frame_ds = pd.dataset.Dataset()
    frame_ds.file_meta = file_meta
    frame_ds.is_little_endian = True
    frame_ds.is_implicit_VR = True

    # Copy essential DICOM elements from the original dataset
    for group in essential_groups:
        for elem in original_ds:
            if elem.tag.group == group:
                frame_ds.add(elem)

    # Copy rescale parameters
    if (0x0028, 0x9145) in pffgs:
        for param_name in ["RescaleIntercept", "RescaleSlope", "RescaleType"]:
            if param_name in pffgs[(0x0028, 0x9145)][0]:
                value = pffgs[(0x0028, 0x9145)][0][param_name].value
                setattr(frame_ds, param_name, value)

    # Copy spatial parameters
    if (0x0028, 0x9110) in pffgs:
        for param_name in ["SpacingBetweenSlices", "PixelSpacing", "SliceThickness"]:
            if param_name in pffgs[(0x0028, 0x9110)][0]:
                value = pffgs[(0x0028, 0x9110)][0][param_name].value
                setattr(frame_ds, param_name, value)

    # Copy cardiac timing information
    if (0x0018, 0x9118) in pffgs:
        for param_name in ["NominalCardiacTriggerDelayTime"]:
            if param_name in pffgs[(0x0018, 0x9118)][0]:
                value = pffgs[(0x0018, 0x9118)][0][param_name].value
                setattr(frame_ds, param_name, value)

    # Remove NumberOfFrames attribute as we're creating single-frame images
    if hasattr(frame_ds, "NumberOfFrames"):
        del frame_ds.NumberOfFrames

    # Set unique instance number
    frame_ds.InstanceNumber = frame_idx + 1

    # Set the pixel data from the current frame
    frame_ds.PixelData = pixel_data.tobytes()

    return frame_ds
