# import the required libraries
import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
import pyvista as pv


def parse_dicom(file, folder):
    # extract image and metadata from a dicom file as a dictionary
    file_path = os.path.join(folder, file)
    with open(file_path, "rb") as file:
        ds = dcmread(file)
        if ds[0x0008, 0x0008].value[2] == "PHASE CONTRAST M":
            tags = dict()
            tags["num"] = ds[0x0020, 0x0013].value  # get instance number
            tags["loc"] = ds[0x0020, 0x1041].value  # get slice location
            tags["img"] = ds.pixel_array
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


def main():
    # fill the dictionary with data
    data = {"FH": [], "RL": [], "AP": []}
    for key in data:
        folder = f"files/{key}"
        files = os.listdir(folder)
        series = [
            parse_dicom(file, folder)
            for file in files
            if parse_dicom(file, folder) is not None
        ]
        series = sorted(series, key=lambda x: x["num"])
        data[key] = np.reshape(np.array(series), (80, 20))

    # create a three-dimensional grid filled with cells values from images
    arr = get_velocities(data, 15)
    grid = pv.ImageData()
    grid.dimensions = (144, 144, 80)
    grid.origin = (0, 0, 0)
    grid.spacing = (1.875, 1.875, 2.5)
    grid["velocity"] = arr.reshape((-1, 3))
    print(grid)
    orto = grid.slice_orthogonal(135, 135, 52.5)

    # build a set of streamlines
    # streamlines, src = grid.streamlines(
    #     source_center=(72, 72, 40),
    #     source_radius=2.0,
    #     n_points=20,
    #     return_source=True,
    #     max_time=100.0,
    # )

    # clip the grid with the segmented ct
    # aorta = pv.read("files/main.stl")
    # clipped = grid.clip_surface(aorta, invert=True, crinkle=False)

    # preview the problem
    p = pv.Plotter()
    p.add_mesh(orto, show_edges=True, component=1, cmap="jet")
    # p.add_mesh(streamlines.tube(radius=0.1))
    # p.add_mesh(aorta, show_edges=True)
    p.show_axes()
    p.show()


if __name__ == "__main__":
    main()
