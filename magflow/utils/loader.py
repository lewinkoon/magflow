import json
import pickle
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d, splev, splprep


def resample(points, num_points=20):
    """Resample a set of 3D points to create a uniform distribution."""
    # Calculate cumulative distance along the line
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])

    # Total length of the centerline
    total_length = distances[-1]

    if len(points) < 4:
        # For very few points, use linear interpolation
        # Normalize distances to [0, 1]
        t = distances / total_length

        # Create interpolation functions for x, y, z
        fx = interp1d(t, points[:, 0])
        fy = interp1d(t, points[:, 1])
        fz = interp1d(t, points[:, 2])

        # Generate equidistant parameter values
        uniform_t = np.linspace(0, 1, num_points)

        # Interpolate new points
        new_points = np.vstack((fx(uniform_t), fy(uniform_t), fz(uniform_t))).T
    else:
        # For more points, use spline interpolation for smoother results
        # Fit a spline to the points
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)

        # Generate equidistant parameter values
        uniform_u = np.linspace(0, 1, num_points)

        # Evaluate the spline at these parameter values
        x_new, y_new, z_new = splev(uniform_u, tck)
        new_points = np.column_stack((x_new, y_new, z_new))

    return new_points


def load_timesteps(patient_data_dir: Path) -> list[int]:
    """
    Load available timesteps for a patient.

    Args:
        patient_data_dir: Path to patient's data directory

    Returns:
        Sorted list of available timesteps
    """
    if not patient_data_dir.exists():
        return []

    available_files = [
        f.name for f in patient_data_dir.iterdir() if f.name.startswith("data.vts.")
    ]

    timesteps = [int(f.split(".")[-1]) for f in available_files]
    return sorted(timesteps)


def load_flow(filepath: Path) -> pv.UnstructuredGrid | None:
    """
    Load velocity field data from VTS file.

    Args:
        filepath: Path to VTS file

    Returns:
        PyVista dataset or None if loading fails
    """
    try:
        return pv.read(str(filepath), force_ext=".vts")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_biomodel(biomodel_path: Path) -> pv.PolyData | None:
    """
    Load and transform biomodel data.

    Args:
        biomodel_path: Path to biomodel VTK file

    Returns:
        Transformed PyVista PolyData or None if loading fails
    """
    if not biomodel_path.exists():
        return None

    try:
        biomodel = pv.read(str(biomodel_path))

        # Apply transformations
        biomodel.rotate_y(-90, inplace=True)
        biomodel.rotate_z(-90, inplace=True)
        biomodel.translate([0, 300, 0], inplace=True)

        # Flip z-coordinates
        points = np.array(biomodel.points)
        points[:, 2] = -points[:, 2]
        biomodel.points = points

        return biomodel

    except Exception as e:
        print(f"Error loading biomodel {biomodel_path}: {e}")
        return None


def load_centerline(centerline_path: Path, num_points: int = 24) -> pv.PolyData | None:
    """
    Load and process centerline data.

    Args:
        centerline_path: Path to centerline JSON file
        num_points: Number of points for resampling

    Returns:
        Processed centerline as PyVista PolyData or None if loading fails
    """
    if not centerline_path.exists():
        return None

    try:
        # Load centerline data
        with centerline_path.open() as f:
            data = json.load(f)

        # Validate JSON structure
        if "markups" not in data or len(data["markups"]) == 0:
            raise ValueError("Invalid centreline data: missing 'markups' array")

        if "controlPoints" not in data["markups"][0]:
            raise ValueError(
                "Invalid centreline data: missing 'controlPoints' in first markup"
            )

        control_points = data["markups"][0]["controlPoints"]
        positions = []

        for i, point in enumerate(control_points):
            if "position" not in point:
                print(f"Warning: Skipping control point {i} - missing 'position' field")
                continue

            if len(point["position"]) != 3:
                print(f"Warning: Skipping control point {i} - invalid position format")
                continue

            positions.append(point["position"])

        if len(positions) < 2:
            raise ValueError(
                f"Insufficient valid control points: {len(positions)} (minimum 2 required)"
            )

        # Convert and transform
        centerline_points = np.array(positions, dtype=np.float64)
        rotation_matrix = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
        centerline_points[:, 2] += 300
        centerline_points = centerline_points @ rotation_matrix.T

        # Resample
        centerline = resample(centerline_points, num_points=num_points)

        # Create PyVista object
        centerline_data = pv.PolyData(centerline)
        lines_array = np.hstack([len(centerline), np.arange(len(centerline))])
        centerline_data.lines = lines_array

        return centerline_data

    except Exception as e:
        print(f"Error processing centerline {centerline_path}: {e}")
        return None


def load_patient(
    patient_id: str, assets_dir: Path, num_centerline_points: int = 24
) -> dict:
    """
    Load all data for a single patient.

    Args:
        patient_id: Patient identifier
        assets_dir: Root directory containing patient data
        num_centerline_points: Number of centerline points

    Returns:
        Dictionary containing patient data
    """
    patient_data = {"timesteps": {}}
    patient_dir = assets_dir / patient_id

    # Load timestep data
    data_dir = patient_dir / "Data"
    timesteps = load_timesteps(data_dir)

    if not timesteps:
        print(f"Warning: No timestep files found for patient {patient_id}")
        return patient_data

    # Load velocity data for each timestep
    for ts in timesteps:
        filepath = data_dir / f"data.vts.{ts}"
        velocity_data = load_flow(filepath)
        if velocity_data is not None:
            patient_data["timesteps"][ts] = velocity_data

    # Load biomodel
    biomodel_path = patient_dir / "Biomodel" / "Biomodel.vtk"
    biomodel = load_biomodel(biomodel_path)
    if biomodel is not None:
        patient_data["biomodel"] = biomodel
    else:
        print(f"Warning: Biomodel not found for patient {patient_id}")

    # Load centerline
    centerline_path = patient_dir / "Biomodel" / "Centerline.mrk.json"
    centerline = load_centerline(centerline_path, num_centerline_points)
    if centerline is not None:
        patient_data["centerline"] = centerline
    else:
        print(f"Warning: Centerline file not found for patient {patient_id}")

    return patient_data


def validate_patient(patient_data: dict, patient_id: str) -> dict[str, bool]:
    """
    Validate completeness of patient data.

    Args:
        patient_data: Patient data dictionary
        patient_id: Patient identifier

    Returns:
        Dictionary with validation results
    """
    validation = {
        "has_timesteps": bool(patient_data.get("timesteps")),
        "has_biomodel": "biomodel" in patient_data,
        "has_centerline": "centerline" in patient_data,
        "timestep_count": len(patient_data.get("timesteps", {})),
    }

    return validation


def load_metric_cache(
    patient_id: str, metric_name: str, cache_dir: Path = Path("cache")
) -> dict | None:
    """Load cached metric for a patient."""
    cache_file = cache_dir / f"{patient_id}_{metric_name}.pkl"
    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {metric_name} cache for {patient_id}: {e}")
    return None


def save_metric_cache(
    patient_id: str, metric_name: str, data, cache_dir: Path = Path("cache")
):
    """Save computed metric to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{patient_id}_{metric_name}.pkl"

    try:
        with cache_file.open("wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to cache {metric_name} for {patient_id}: {e}")
