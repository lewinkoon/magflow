import numpy as np
import pyvista as pv
from rich import print
from scipy.interpolate import interp1d, splev, splprep


def load_biomodel(biomodel_path):
    """Load and transform the biomodel for visualization."""
    biomodel_data = pv.read(str(biomodel_path))

    # Apply transformations
    biomodel_data.rotate_y(-90, inplace=True)
    biomodel_data.rotate_z(-90, inplace=True)

    biomodel_data.translate([0, 300, 0], inplace=True)

    # Flip Z coordinates
    points = np.array(biomodel_data.points)
    points[:, 2] = -points[:, 2]
    biomodel_data.points = points

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


def ortoplanes(points, dataset, radius=30):
    # Initialize storage for cross-sections
    cross_sections = []

    # Calculate tangent vectors at each point
    tangents = np.zeros_like(points)

    # For the first point
    tangents[0] = points[1] - points[0]
    tangents[0] = tangents[0] / np.linalg.norm(tangents[0])

    # For the last point
    tangents[-1] = points[-1] - points[-2]
    tangents[-1] = tangents[-1] / np.linalg.norm(tangents[-1])

    # For interior points, use central difference
    for i in range(1, len(points) - 1):
        tangents[i] = points[i + 1] - points[i - 1]
        tangents[i] = tangents[i] / np.linalg.norm(tangents[i])

    # Create orthogonal planes and slice dataset
    for i, point in enumerate(points):
        # The normal to the plane is the tangent at this point
        normal = tangents[i]

        # Slice the dataset with a plane at this point and normal
        full_slice = dataset.slice(normal=normal, origin=point)

        # Create a sphere to limit the extent of the slice
        sphere = pv.Sphere(radius=radius, center=point)

        # Use boolean intersection to limit the slice to the sphere's radius
        cross_section = full_slice.clip_surface(sphere)
        cross_sections.append(cross_section)

    return cross_sections
