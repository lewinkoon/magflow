import csv
import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
import vtk
from magflow.logger import logger
import zipfile
import io
import pydicom as pd


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


def parse(axis):
    res = []
    folder_path = f".tmp/{axis}"
    # changed: ensure folder exists
    if not os.path.exists(folder_path):
        logger.error(f"Folder {folder_path} does not exist.")
        raise FileNotFoundError(f"Folder {folder_path} not found.")
    for idx, filename in enumerate(os.listdir(folder_path)):  # changed variable name
        file_path = os.path.join(folder_path, filename)
        # changed: use 'slice_data' to avoid shadowing built-in 'slice'
        slice_data = {}
        with open(file_path, "rb") as binary_file:  # changed inner variable name
            ds = dcmread(binary_file)

            # assign image array
            if ds[0x0028, 0x0004].value == "MONOCHROME2":
                img = ds.pixel_array
            else:
                img = np.mean(ds.pixel_array, axis=2)
            slice_data["pxl"] = img
            slice_data["val"] = apply_modality_lut(img, ds)

            # assign tags
            slice_data["axis"] = ds[0x0008, 0x103E].value  # axis name
            slice_data["num"] = ds[0x0020, 0x0013].value  # instance number
            slice_data["spacing"] = ds[0x0028, 0x0030].value  # pixel spacing
            slice_data["height"] = ds[0x0018, 0x0088].value  # spacing between slices
            slice_data["time"] = int(ds[0x0020, 0x9153].value)  # trigger time

            res.append(slice_data)
    return res


# Renamed to avoid shadowing the built-in "filter"
def filter_by_time(series, frame):
    res = [
        item["val"] for item in series if item["time"] == frame
    ]  # changed variable name
    return res


def tabulate(fh, rl, ap, voxel, time):
    # convert data into tabular dictionary
    if not fh:  # added guard to prevent IndexError
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


# Revised mask: do not reset input lists, instead create new masked lists.
def mask(fh, rl, ap, mk):
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


def tocsv(data, time):
    if not os.path.exists("output"):
        os.makedirs("output")

    fields = data[0].keys()
    path = f"output/data.csv.{time}"
    with open(path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def tovtk(data, time):
    if not os.path.exists("output"):
        os.makedirs("output")

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
    writer.SetFileName(f"output/data.vts.{time}")
    writer.SetInputData(sgrid)
    writer.Write()


def showtag(dataset, group, element):
    try:
        tag_name = dataset[group, element].name
        tag_value = dataset[group, element].value
        logger.info(f"{tag_name}: {tag_value}")
    except KeyError:
        logger.error(f"[{group:04x},{element:04x}]: Not found.")


def wrapper(raw, fh, rl, ap, voxel, time):
    fh_filtered = filter_by_time(fh, time)
    rl_filtered = filter_by_time(rl, time)
    ap_filtered = filter_by_time(ap, time)

    data = tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
    if raw:
        tocsv(data, time)
    else:
        tovtk(data, time)
    logger.info(f"Trigger time {time} exported with {len(data)} rows.")
