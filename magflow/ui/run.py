import streamlit as st
import io
import pydicom as dcm
import pandas as pd
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut
import random
import zipfile


@st.cache_data(show_spinner=False)
def load(f):
    axis = f.name.split(".")[0]
    ds = dcm.dcmread(f)
    n = int(ds.NumberOfFrames)
    img = ds.pixel_array.squeeze()
    essentials = [0x0008, 0x0010, 0x0018, 0x0020, 0x0028]

    frames = []
    bar = st.progress(0, text=f"Frame 0/{n}")
    for idx in range(n):
        # target = os.path.join(f"files/{axis}", f"img{idx:04}.dcm")
        file_meta = dcm.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dcm.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        tmp_ds = dcm.dataset.Dataset()
        tmp_ds.file_meta = file_meta
        tmp_ds.is_little_endian = tmp_ds.file_meta.TransferSyntaxUID.is_little_endian
        tmp_ds.is_implicit_VR = tmp_ds.file_meta.TransferSyntaxUID.is_implicit_VR

        for group in essentials:
            for key, value in ds.group_dataset(group).items():
                tmp_ds[key] = value

        # sfgs = ds.SharedFunctionalGroupsSequence[0]
        pffgs = ds.PerFrameFunctionalGroupsSequence[idx]

        # copy velocity tags
        vtags = ["RescaleIntercept", "RescaleSlope", "RescaleType"]
        for tag_name in vtags:
            if tag_name in pffgs[(0x0028, 0x9145)][0]:
                value = pffgs[(0x0028, 0x9145)][0][tag_name].value
                setattr(tmp_ds, tag_name, value)

        # copy spatial tags
        stags = ["SpacingBetweenSlices", "PixelSpacing", "SliceThickness"]
        for tag_name in stags:
            if tag_name in pffgs[(0x0028, 0x9110)][0]:
                value = pffgs[(0x0028, 0x9110)][0][tag_name].value
                setattr(tmp_ds, tag_name, value)

        # copy trigger time
        ttag = "NominalCardiacTriggerDelayTime"
        if ttag in pffgs[(0x0018, 0x9118)][0]:
            value = pffgs[(0x0018, 0x9118)][0][ttag].value
            setattr(tmp_ds, ttag, value)

        del tmp_ds.NumberOfFrames
        tmp_ds.InstanceNumber = idx + 1
        tmp_ds.PixelData = img[idx, :].squeeze().tobytes()
        # tmp_ds.save_as(target, write_like_original=False)

        slice = {}
        # assign image array
        if tmp_ds[0x0028, 0x0004].value == "MONOCHROME2":
            pixels = tmp_ds.pixel_array
        else:
            pixels = np.mean(tmp_ds.pixel_array, axis=2)
        slice["pxl"] = pixels
        slice["val"] = apply_modality_lut(pixels, tmp_ds)

        # assign tags
        slice["axis"] = tmp_ds[0x0008, 0x103E].value  # axis name
        slice["num"] = tmp_ds[0x0020, 0x0013].value  # instance number
        slice["spacing"] = tmp_ds[0x0028, 0x0030].value  # pixel spacing
        slice["height"] = tmp_ds[0x0018, 0x0088].value  # spacing between slices
        slice["time"] = int(tmp_ds[0x0020, 0x9153].value)  # trigger time

        frames.append(slice)
        bar.progress((idx + 1) / n, text=f"{axis} frame {idx + 1}/{n}")

    return frames


@st.cache_data(show_spinner=False)
def build(_f1, _f2, _f3, timeframes):
    zbuffer = io.BytesIO()
    dataframes = []
    bar = st.progress(0, text=f"Timestep 0/{len(timeframes)}")
    with zipfile.ZipFile(zbuffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, time in enumerate(timeframes):
            # filter data for a single timeframe
            fh = [item["val"] for item in _f1 if item["time"] == time]
            rl = [item["val"] for item in _f2 if item["time"] == time]
            ap = [item["val"] for item in _f3 if item["time"] == time]

            # get the dimensions of the images
            dimensions = fh[0].shape
            num_pixels = dimensions[0] * dimensions[1]
            num_slices = len(fh)

            # prepare grid indices for x and y positions for a single 2D slice
            x_vals = np.arange(dimensions[1]) * voxel[0]
            y_vals = np.arange(dimensions[0]) * voxel[1]
            xv, yv = np.meshgrid(x_vals, y_vals)

            # create lists to store data for each column
            x_data = np.tile(xv.ravel(), num_slices)
            y_data = np.tile(yv.ravel(), num_slices)
            z_data = np.repeat(np.arange(num_slices) * voxel[2], num_pixels)
            t_data = np.full(num_pixels * num_slices, time)

            # flatten all the images in ap, fh, and rl
            vx_data = np.concatenate([img[::-1].ravel() for img in ap])
            vy_data = np.concatenate([img[::-1].ravel() for img in fh])
            vz_data = np.concatenate([img[::-1].ravel() for img in rl])

            # create a DataFrame from the data
            df = pd.DataFrame(
                {
                    "x": x_data,
                    "y": y_data,
                    "z": z_data,
                    "t": t_data,
                    "vx": vx_data,
                    "vy": vy_data,
                    "vz": vz_data,
                }
            )

            # fields = table[0].keys()
            # buffer = io.StringIO()
            # writer = csv.DictWriter(buffer, fieldnames=fields)
            # writer.writeheader()
            # for row in table:
            #     writer.writerow(row)
            zf.writestr(f"data.csv.{time}", df.to_csv().encode("utf-8"))
            dataframes.append(df)
            bar.progress((idx + 1) / 24, text=f"Timestep {idx + 1}/{len(timeframes)}")
    data = pd.concat(dataframes, ignore_index=True)
    bar.empty()

    return data, zbuffer


st.title("Magflow 🩸")
st.markdown(
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
)

with st.sidebar:
    st.subheader("FH")
    file1 = st.file_uploader("Upload feet-head axis dicom files")
    st.subheader("RL")
    file2 = st.file_uploader("Upload right-left axis dicom files")
    st.subheader("AP")
    file3 = st.file_uploader("Upload anterior-posterior axis dicom files")


if file1 and file2 and file3:
    with st.status("Loading data...", expanded=True) as status:
        fh = load(file1)
        rl = load(file2)
        ap = load(file3)

        data = fh + rl + ap
        sample = random.choice(data)

        # list unique trigger times
        timeframes = sorted(set(item["time"] for item in data))
        st.write(f"Timesteps: {timeframes}")

        # get volume dimensions
        volume = (sample["pxl"].shape[0], sample["pxl"].shape[1])
        st.write(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

        # get voxel spacing
        voxel = (
            sample["spacing"][0],
            sample["spacing"][1],
            sample["height"],
        )
        st.write(
            f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
        )
        status.update(label="Loading complete!", state="complete", expanded=True)

if st.button("Build", type="primary", use_container_width=True):
    df, zbuffer = build(fh, rl, ap, timeframes)
    st.dataframe(df[df["t"] == 0], use_container_width=True)
    st.download_button(
        label="Download ZIP",
        data=zbuffer,
        file_name="data.zip",
        mime="application/zip",
        use_container_width=True,
    )
