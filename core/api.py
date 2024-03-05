import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut


def parse_dicom(file):
    # extract image and metadata from a dicom file as a dictionary
    with open(file, "rb") as file:
        ds = dcmread(file)
        if ds[0x0008, 0x0008].value[2] == "PHASE CONTRAST M":
            tags = {}
            tags["axis"] = ds[0x0008, 0x103E].value
            tags["num"] = ds[0x0020, 0x0013].value  # get instance number
            tags["loc"] = ds[0x0020, 0x1041].value  # get slice location
            tags["time"] = ds[0x0018, 0x1060].value
            # tags["img"] = ds.pixel_array
            tags["val"] = apply_modality_lut(
                ds.pixel_array, ds
            )  # convert pixel values to velocity values
            return tags


def get_velocities(data, time):
    # format the velocity field as an array for a given timestep
    fh = np.array([img["val"] for img in data["FH"][:, time]])
    rl = np.array([img["val"] for img in data["RL"][:, time]])
    ap = np.array([img["val"] for img in data["AP"][:, time]])

    return np.stack((fh, rl, ap), axis=-1)

    # # create a three-dimensional grid filled with cells values from images
    # arr = get_velocities(data, 15)
    # grid = pv.ImageData()
    # grid.dimensions = (144, 144, 80)
    # grid.origin = (0, 0, 0)
    # grid.spacing = (1.875, 1.875, 2.5)
    # grid["velocity"] = arr.reshape((-1, 3))
    # print(grid)
    # orto = grid.slice_orthogonal(135, 135, 52.5)

    # clip the grid with the segmented ct
    # aorta = pv.read("files/main.stl")
    # clipped = grid.clip_surface(aorta, invert=True, crinkle=False)

    # # preview the problem
    # p = pv.Plotter()
    # p.add_mesh(orto, show_edges=True, component=1, cmap="jet")
    # # p.add_mesh(streamlines.tube(radius=0.1))
    # # p.add_mesh(aorta, show_edges=True)
    # p.show_axes()
    # p.show()
