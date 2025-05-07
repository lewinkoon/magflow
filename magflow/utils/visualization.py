import numpy as np
import pyvista as pv
from rich import print


def load_biomodel(biomodel_path):
    """Load and transform the biomodel for visualization."""
    biomodel_data = pv.read(str(biomodel_path))

    # Apply transformations
    biomodel_data.rotate_y(-90, inplace=True)
    biomodel_data.rotate_z(-90, inplace=True)
    print("Applied rotations: Y-axis: -90°, Z-axis: -90°")

    biomodel_data.translate([0, 300, 0], inplace=True)
    print("Translated model: Y-axis: +300 units")

    # Flip Z coordinates
    points = np.array(biomodel_data.points)
    points[:, 2] = -points[:, 2]
    biomodel_data.points = points
    print("Mirrored biomodel in XY plane")

    return biomodel_data


def extract_aorta(dataset, biomodel_data):
    """Extract the aorta region using boolean intersection."""
    intersection = dataset.select_enclosed_points(biomodel_data)
    aorta = dataset.extract_points(
        intersection.point_data["SelectedPoints"].astype(bool)
    )
    velocity_vectors = aorta.point_data["Velocity"]
    velocity_magnitude = np.linalg.norm(velocity_vectors, axis=1)
    aorta.point_data["VelocityMagnitude"] = velocity_magnitude
    return aorta


def render_volume(aorta):
    """Create a uniform grid for volume rendering."""
    bounds = aorta.bounds
    print(f"Bounds: {bounds}")

    # Create a uniform grid with appropriate resolution
    uniform = pv.ImageData(
        dimensions=(128, 128, 128),
        spacing=(
            (bounds[1] - bounds[0]) / 127,
            (bounds[3] - bounds[2]) / 127,
            (bounds[5] - bounds[4]) / 127,
        ),
        origin=(bounds[0], bounds[2], bounds[4]),
    )

    # Sample data onto the uniform grid
    return uniform.sample(aorta)


def generate_streamlines(aorta, center):
    """Generate streamlines visualization from velocity vectors in the aorta."""
    print("Generating streamlines from velocity field")
    streamlines, source = aorta.streamlines(
        return_source=True,
        source_center=center,
        source_radius=10,
    )
    return streamlines, source


def calculate_velocity_statistics(aorta):
    """Calculate and return statistics about the velocity magnitude field.

    Returns:
        dict: Dictionary containing mean and peak velocity values.
    """
    velocity_magnitude = aorta.point_data["VelocityMagnitude"]

    # Calculate statistics
    mean_velocity = np.mean(velocity_magnitude)
    peak_velocity = np.max(velocity_magnitude)

    # Return as dictionary for easy access
    return {"mean": mean_velocity, "peak": peak_velocity}
