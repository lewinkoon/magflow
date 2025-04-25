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
