# Standard library imports
import tempfile
from pathlib import Path


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
