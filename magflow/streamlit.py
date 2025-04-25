# Standard library imports
import tempfile
from pathlib import Path
import zipfile
import io
import pydicom as pd


import pydicom as pd
import streamlit as st

# Local imports
from magflow.utils.data import (
    create_zip_buffer,
    determine_axis,
    load_dicom_file,
    create_frame_dataset,
)

# Constants
ESSENTIAL_TAGS = [0x0008, 0x0010, 0x0018, 0x0020, 0x0028]


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


# Streamlit app setup
st.set_page_config(page_title="Magflow", page_icon="🩸")
st.title("Magflow 🩸")
st.markdown(
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
)

# File uploader widget
uploaded_files = st.file_uploader(
    "Upload DICOM files", type=["dcm"], accept_multiple_files=True
)

if st.button("Process DICOM files", use_container_width=True) and uploaded_files:
    # Display a status indicator for file processing
    status_area = st.status("Processing DICOM files...", expanded=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files_processed = 0

        # Process each file
        for file in uploaded_files:
            status_area.write(f"Loading {file.name}...")

            # Determine the directional axis
            axis_value = determine_axis(file.name)
            if axis_value is None:
                st.warning(f"Axis not determined for file {file.name}. Skipping.")
                continue

            # Create subdirectory for this axis type
            dst_dir = output_dir / axis_value
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Load and validate DICOM file
            ds, error_msg = load_dicom_file(file)
            if ds is None:
                st.warning(error_msg)
                continue

            try:
                # Extract the pixel data array
                img = ds.pixel_array.squeeze()
            except Exception as e:
                st.warning(f"Error accessing pixel data in {file.name}: {e}")
                continue

            # Process each frame
            num_frames = int(ds.NumberOfFrames)
            for idx in range(num_frames):
                # Access the per-frame metadata
                try:
                    pffgs = ds.PerFrameFunctionalGroupsSequence[idx]
                except IndexError:
                    st.warning(f"Frame {idx} missing in {file.name}.")
                    continue

                # Define output file path for this frame
                target = dst_dir / f"img{idx:04}.dcm"

                # Create and save the frame dataset
                frame_dataset = create_frame_dataset(
                    ds, pffgs, idx, img[idx, :].squeeze(), ESSENTIAL_TAGS
                )
                frame_dataset.save_as(target, write_like_original=False)

            files_processed += 1

        # Create a downloadable zip if files were processed
        if files_processed > 0:
            status_area.write("Building zip file...")
            zip_buffer = create_zip_buffer(output_dir)

    if zip_buffer:
        # Update status to indicate completion
        status_area.update(label="Ready to download!", state="complete")

        # Add download button for the processed files
        st.download_button(
            label="Download processed DICOM files",
            data=zip_buffer,
            file_name="processed_dicom_files.zip",
            mime="application/zip",
            use_container_width=True,
        )
