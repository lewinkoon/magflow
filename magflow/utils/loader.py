import json
import pickle
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d, splev, splprep


def resample(points, num_points=20):
    """
    Resample 3D points to create uniform spacing along a curve.

    Uses linear interpolation for <4 points, spline interpolation otherwise.

    Args:
        points: Array of 3D points with shape (n, 3)
        num_points: Target number of resampled points

    Returns:
        Array of resampled points with shape (num_points, 3)
    """
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
    Extract available timestep numbers from VTS files in patient directory.

    Args:
        patient_data_dir: Directory containing data.vts.{timestep} files

    Returns:
        Sorted list of timestep numbers, empty if directory doesn't exist
    """
    if not patient_data_dir.exists():
        return []

    timesteps = []
    for f in patient_data_dir.iterdir():
        if f.name.startswith("data.vts.") and f.is_file():
            try:
                timestep = int(f.name.split(".")[-1])
                timesteps.append(timestep)
            except ValueError:
                print(f"Warning: Invalid timestep format in file {f.name}")
                continue

    return sorted(timesteps)


def load_flow(filepath: Path) -> pv.UnstructuredGrid | None:
    """
    Load velocity field data from VTS file.

    Args:
        filepath: Path to VTS file containing flow data

    Returns:
        PyVista UnstructuredGrid with velocity data, or None if loading fails
    """
    try:
        return pv.read(str(filepath), force_ext=".vts")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_biomodel(biomodel_path: Path) -> pv.PolyData | None:
    """
    Load biomodel geometry and apply coordinate transformations.

    Applies rotations, translation, and z-coordinate flipping to align
    with the flow simulation coordinate system.

    Args:
        biomodel_path: Path to biomodel VTK file

    Returns:
        Transformed PyVista PolyData, or None if loading fails
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
    Load centerline from 3D Slicer markup JSON and process into uniform spacing.

    Extracts control points, applies coordinate transformations, and resamples
    to create a smooth centerline with uniform point distribution.

    Args:
        centerline_path: Path to 3D Slicer centerline JSON file
        num_points: Number of points for resampled centerline

    Returns:
        PyVista PolyData with processed centerline, or None if loading fails
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
    Load complete patient dataset including flow data, biomodel, centerline, and metadata.

    Args:
        patient_id: Unique patient identifier
        assets_dir: Root directory containing patient subdirectories
        num_centerline_points: Number of points for centerline resampling

    Returns:
        Dictionary with keys: 'timesteps' (flow data), 'biomodel', 'centerline', 'metadata'
    """
    patient_data = {"timesteps": {}}
    patient_dir = assets_dir / patient_id

    # Load metadata
    metadata_path = patient_dir / "meta.json"
    metadata = load_metadata(metadata_path)
    if metadata is not None:
        patient_data["metadata"] = metadata
    else:
        print(f"Warning: Metadata file not found for patient {patient_id}")

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


def load_metadata(metadata_path: Path) -> dict | None:
    """
    Load patient metadata from JSON file.

    Args:
        metadata_path: Path to meta.json file

    Returns:
        Dictionary containing patient metadata, or None if loading fails
    """
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata {metadata_path}: {e}")
        return None


def validate_patient(patient_data: dict, patient_id: str) -> dict[str, bool]:
    """
    Check completeness and validity of loaded patient data.

    Args:
        patient_data: Patient data dictionary from load_patient()
        patient_id: Patient identifier for error reporting

    Returns:
        Dictionary with validation flags and counts
    """
    validation = {
        "has_timesteps": bool(patient_data.get("timesteps")),
        "has_biomodel": "biomodel" in patient_data,
        "has_centerline": "centerline" in patient_data,
        "has_metadata": "metadata" in patient_data,
        "timestep_count": len(patient_data.get("timesteps", {})),
    }

    return validation


def load_metric_cache(
    patient_id: str, metric_name: str, cache_dir: Path = Path("cache")
) -> dict | None:
    """
    Load previously computed metric from pickle cache file.

    Args:
        patient_id: Patient identifier
        metric_name: Name of the cached metric
        cache_dir: Directory containing cache files

    Returns:
        Cached metric data, or None if not found or loading fails
    """
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
    """
    Save computed metric data to pickle cache file for future use.

    Args:
        patient_id: Patient identifier
        metric_name: Name of the metric to cache
        data: Metric data to save
        cache_dir: Directory to store cache files
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{patient_id}_{metric_name}.pkl"

    try:
        with cache_file.open("wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to cache {metric_name} for {patient_id}: {e}")
