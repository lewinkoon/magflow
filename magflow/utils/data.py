import csv
import numpy as np
from pathlib import Path
import vtk
from magflow.utils.logger import logger


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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fields = data[0].keys() if data else []
    path = output_path / f"data.csv.{time}"

    try:
        with open(path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        return True
    except Exception as e:
        logger.error(f"Error writing CSV file: {e}")
        return False


def tovtk(data, time, output_dir="output"):
    """Export velocity data to VTK format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
    writer.SetFileName(f"{output_path}/data.vts.{time}")
    writer.SetInputData(sgrid)
    writer.Write()
    return True
