import json

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from rich import print
from scipy.interpolate import interp1d, splev, splprep
from scipy.spatial import KDTree


def load_centreline(centreline_path):
    """Load and validate centreline data from JSON file."""
    try:
        with centreline_path.open() as f:
            data = json.load(f)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Centreline file not found at {centreline_path}"
        ) from err
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in centreline file: {e}") from e

    # Validate JSON structure
    if "markups" not in data or len(data["markups"]) == 0:
        raise ValueError("Invalid centreline data: missing 'markups' array")

    if "controlPoints" not in data["markups"][0]:
        raise ValueError(
            "Invalid centreline data: missing 'controlPoints' in first markup"
        )

    return data


def extract_positions(control_points):
    """Extract valid 3D positions from control points."""
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

    return positions


def apply_transformations(points, z_translation=300):
    """Apply translation and rotation transformations to points."""
    # Define transformation parameters
    rotation_matrix = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)

    # Apply transformations
    points[:, 2] += z_translation
    return points @ rotation_matrix.T


def create_polydata(centreline_points):
    """Create PyVista PolyData object from centreline points."""
    centreline_data = pv.PolyData(centreline_points)
    lines_array = np.hstack([len(centreline_points), np.arange(len(centreline_points))])
    centreline_data.lines = lines_array
    return centreline_data


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


def calculate_wss_for_point(
    wall_point, normal, field_tree, field_points, velocity_field, mu=0.004
):
    """Calculate WSS for a single wall point.

    Args:
        wall_point: Wall point coordinates
        normal: Wall normal vector
        field_tree: KDTree for field points
        field_points: Array of field point coordinates
        velocity_field: Array of velocity vectors
        mu: Blood viscosity in Pa·s

    Returns:
        WSS value for the point
    """
    # Find points along normal direction (within 2mm)
    distances, indices = field_tree.query(wall_point, k=10)

    if len(indices) > 1:
        # Get velocities at these points
        velocities = velocity_field[indices]

        # Project points onto normal direction
        projected_distances = np.array(
            [np.dot(field_points[idx] - wall_point, normal) for idx in indices]
        )

        # Sort by distance along normal
        sorted_indices = np.argsort(projected_distances)
        sorted_distances = projected_distances[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        # Calculate velocity gradient at the wall (using points within 2mm)
        valid_indices = np.where(sorted_distances > 0)[0]

        if len(valid_indices) > 1:
            # Use first point for gradient calculation
            idx = valid_indices[0]
            dist = sorted_distances[idx]
            vel = sorted_velocities[idx]

            # Project velocity onto tangential plane
            vel_normal_component = np.dot(vel, normal) * normal
            vel_tangential = vel - vel_normal_component

            # Calculate WSS (Pa) = viscosity * velocity_gradient
            return mu * np.linalg.norm(vel_tangential) / dist

    return 0.0


def calculate_wss_timestep(wall_points, wall_normals, aorta, mu=0.004):
    """Calculate WSS for all wall points at a single timestep.

    Args:
        wall_points: Array of wall point coordinates
        wall_normals: Array of wall normal vectors
        aorta: Aorta dataset with velocity field
        mu: Blood viscosity in Pa·s

    Returns:
        Array of WSS values for all wall points
    """
    # Extract velocity field
    velocity_field = np.array(aorta["Velocity"]) * 10  # Convert to mm/s
    field_points = np.array(aorta.points)

    # Create KDTree for field points
    field_tree = KDTree(field_points)

    # Calculate WSS for wall points
    wss = np.zeros(len(wall_points))

    # For each wall point, find closest velocity points and calculate velocity gradient
    for i in range(len(wall_points)):
        wss[i] = calculate_wss_for_point(
            wall_points[i],
            wall_normals[i],
            field_tree,
            field_points,
            velocity_field,
            mu,
        )

    return wss


def calculate_osi(timesteps, timestep_data, biomodel_data, mu=0.004):
    """Calculate Oscillatory Shear Index (OSI) for all wall points.

    Args:
        timesteps: List of timestep values
        timestep_data: Dictionary mapping timesteps to datasets
        biomodel_data: Biomodel mesh data
        mu: Blood viscosity in Pa·s

    Returns:
        Array of OSI values for all wall points
    """
    # Extract the wall points from biomodel
    biomodel_data.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    wall_points = np.array(biomodel_data.points)
    wall_normals = np.array(biomodel_data["Normals"])

    # Calculate the time-averaged WSS vector and magnitude
    wss_vectors_sum = np.zeros((len(wall_points), 3))
    wss_magnitude_sum = np.zeros(len(wall_points))

    # For each timestep, add the WSS vector components and magnitudes
    for ts in timesteps:
        dataset = timestep_data[ts]
        aorta = extract_aorta(dataset, biomodel_data)

        velocity_field = np.array(aorta["Velocity"]) * 10  # Convert to mm/s
        field_points = np.array(aorta.points)

        field_tree = KDTree(field_points)

        for i in range(len(wall_points)):
            wall_point = wall_points[i]
            normal = wall_normals[i]

            distances, indices = field_tree.query(wall_point, k=10)

            if len(indices) > 1:
                projected_distances = np.array(
                    [np.dot(field_points[idx] - wall_point, normal) for idx in indices]
                )

                sorted_indices = np.argsort(projected_distances)
                sorted_distances = projected_distances[sorted_indices]
                sorted_velocities = velocity_field[indices][sorted_indices]

                valid_indices = np.where(sorted_distances > 0)[0]

                if len(valid_indices) > 1:
                    idx = valid_indices[0]
                    dist = sorted_distances[idx]
                    vel = sorted_velocities[idx]

                    # Project velocity onto tangential plane
                    vel_normal_component = np.dot(vel, normal) * normal
                    vel_tangential = vel - vel_normal_component

                    # Calculate WSS vector (direction is important for OSI)
                    wss_vector = mu * vel_tangential / dist

                    # Add to sums
                    wss_vectors_sum[i] += wss_vector
                    wss_magnitude_sum[i] += np.linalg.norm(wss_vector)

    # Calculate OSI: 0.5 * (1 - |∑wss_vector| / ∑|wss_vector|)
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle division by zero
        osi_values = 0.5 * (
            1 - np.linalg.norm(wss_vectors_sum, axis=1) / wss_magnitude_sum
        )

    # Replace NaN values with 0 (occurs when there's no WSS)
    osi_values = np.nan_to_num(osi_values)

    return osi_values


def calculate_velocity_gradients(dataset):
    """Calculate velocity gradients using finite differences."""
    # Get velocity vectors and convert to mm/s
    velocity = dataset["Velocity"] * 10  # Convert from cm/s to mm/s

    # Calculate gradients using PyVista's built-in gradient function
    dataset_copy = dataset.copy()
    dataset_copy["u"] = velocity[:, 0]  # x-component
    dataset_copy["v"] = velocity[:, 1]  # y-component
    dataset_copy["w"] = velocity[:, 2]  # z-component

    # Compute gradients for each velocity component
    grad_u = dataset_copy.gradient("u", gradient_name="grad_u")
    grad_v = dataset_copy.gradient("v", gradient_name="grad_v")
    grad_w = dataset_copy.gradient("w", gradient_name="grad_w")

    return grad_u["grad_u"], grad_v["grad_v"], grad_w["grad_w"]


def calculate_strain_rate_tensor(grad_u, grad_v, grad_w):
    """Calculate the strain rate tensor components."""
    # Strain rate tensor: Sij = 0.5 * (∂ui/∂xj + ∂uj/∂xi)
    # For 3D flow: 9 components but only 6 unique due to symmetry

    n_points = len(grad_u)
    strain_rate = np.zeros((n_points, 3, 3))

    # Fill strain rate tensor for each point
    for i in range(n_points):
        # Extract gradient components
        du_dx, du_dy, du_dz = grad_u[i]
        dv_dx, dv_dy, dv_dz = grad_v[i]
        dw_dx, dw_dy, dw_dz = grad_w[i]

        # Symmetric strain rate tensor
        strain_rate[i, 0, 0] = du_dx  # S11
        strain_rate[i, 1, 1] = dv_dy  # S22
        strain_rate[i, 2, 2] = dw_dz  # S33
        strain_rate[i, 0, 1] = strain_rate[i, 1, 0] = 0.5 * (du_dy + dv_dx)  # S12
        strain_rate[i, 0, 2] = strain_rate[i, 2, 0] = 0.5 * (du_dz + dw_dx)  # S13
        strain_rate[i, 1, 2] = strain_rate[i, 2, 1] = 0.5 * (dv_dz + dw_dy)  # S23

    return strain_rate


def calculate_viscous_dissipation(strain_rate, mu):
    """Calculate viscous dissipation function Φ = 2μ * Sij * Sij."""
    # Viscous dissipation: Φ = 2μ * Σ(Sij²)
    dissipation = np.zeros(len(strain_rate))

    for i in range(len(strain_rate)):
        # Calculate Frobenius norm squared of strain rate tensor
        sij_squared = np.sum(strain_rate[i] ** 2)
        dissipation[i] = 2 * mu * sij_squared

    return dissipation


def calculate_gradient_tensor(aorta):
    """Calculate the full velocity gradient tensor for all points in the aorta.

    Args:
        aorta: PyVista dataset containing velocity field

    Returns:
        np.ndarray: Gradient tensor of shape (n_points, 3, 3)
    """
    velocity = aorta["Velocity"]
    grad_tensor = np.zeros((aorta.n_points, 3, 3))

    for i in range(3):  # velocity components
        vel_component = velocity[:, i]
        aorta.point_data["temp_vel"] = vel_component
        grad_result = aorta.compute_derivative(scalars="temp_vel")
        grad_tensor[:, i, :] = grad_result["gradient"]

    # Clean up temporary data
    if "temp_vel" in aorta.point_data:
        del aorta.point_data["temp_vel"]

    return grad_tensor


def calculate_strain_rate_from_gradient(grad_tensor):
    """Calculate strain rate tensor from velocity gradient tensor.

    Args:
        grad_tensor: Velocity gradient tensor of shape (n_points, 3, 3)

    Returns:
        np.ndarray: Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    """
    # Calculate strain rate tensor: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    strain_rate = 0.5 * (grad_tensor + np.transpose(grad_tensor, (0, 2, 1)))
    return strain_rate


def calculate_viscous_dissipation_from_strain(strain_rate, mu=0.004):
    """Calculate viscous dissipation function from strain rate tensor.

    Args:
        strain_rate: Strain rate tensor of shape (n_points, 3, 3)
        mu: Dynamic viscosity of blood (Pa·s)

    Returns:
        np.ndarray: Viscous dissipation values for each point
    """
    # Calculate viscous dissipation function: Φ = μ * Σ(S_ij^2)
    dissipation = mu * np.sum(strain_rate**2, axis=(1, 2))
    return dissipation


def estimate_cell_volumes(aorta):
    """Estimate cell volumes for the aorta dataset.

    Args:
        aorta: PyVista dataset

    Returns:
        np.ndarray: Array of estimated cell volumes
    """
    # Try to compute cell sizes first
    aorta.compute_cell_sizes(length=False, area=False, volume=True)

    if "Volume" in aorta.cell_data:
        return aorta.cell_data["Volume"]
    else:
        # Fallback: use unit volumes
        return np.ones(aorta.n_cells)


def map_point_data_to_cells(aorta, point_data):
    """Map point data to cell data by averaging.

    Args:
        aorta: PyVista dataset
        point_data: Array of data values at points

    Returns:
        np.ndarray: Cell-averaged data values
    """
    cell_data = np.zeros(aorta.n_cells)

    for i in range(aorta.n_cells):
        cell = aorta.get_cell(i)
        point_ids = cell.point_ids
        cell_data[i] = np.mean(point_data[point_ids])

    return cell_data


def calculate_viscous_energy_loss_timestep(aorta, mu=0.004):
    """Calculate viscous energy loss for a single timestep.

    Args:
        aorta: PyVista dataset containing velocity field
        mu: Dynamic viscosity of blood (Pa·s)

    Returns:
        tuple: (total_energy_loss, total_dissipation, avg_dissipation_rate)
    """
    if aorta.n_points == 0:
        return 0, 0, 0

    # Check if velocity data is available
    if "Velocity" not in aorta.point_data:
        return 0, 0, 0

    # Calculate velocity gradients
    grad_tensor = calculate_gradient_tensor(aorta)

    # Calculate strain rate tensor
    strain_rate = calculate_strain_rate_from_gradient(grad_tensor)

    # Calculate viscous dissipation
    dissipation = calculate_viscous_dissipation_from_strain(strain_rate, mu)

    # Estimate cell volumes
    cell_volumes = estimate_cell_volumes(aorta)

    # Map point dissipation to cells
    cell_dissipation = map_point_data_to_cells(aorta, dissipation)

    # Calculate total viscous energy loss by integrating over volume
    total_energy_loss = np.sum(cell_dissipation * cell_volumes)  # W (J/s)
    total_dissipation = np.sum(cell_dissipation)
    avg_dissipation_rate = np.mean(dissipation)

    return total_energy_loss, total_dissipation, avg_dissipation_rate


def create_energy_loss_plots(
    timesteps, viscous_energy_loss, avg_dissipation_rate, mean_flow_rates
):
    """
    Create comprehensive visualization of viscous energy loss analysis.

    Parameters:
    -----------
    timesteps : list
        List of timestep values
    viscous_energy_loss : dict
        Energy loss values for each timestep (in W)
    avg_dissipation_rate : dict
        Average dissipation rates for each timestep (in W/m³)
    mean_flow_rates : dict
        Mean flow rates for each timestep (in ml/s)

    Returns:
    --------
    tuple: (fig, correlation_coefficient)
    """
    # Convert data to lists for plotting
    energy_loss_mw = [
        viscous_energy_loss[ts] * 1000 for ts in timesteps
    ]  # Convert to mW
    dissipation_rates = [avg_dissipation_rate[ts] for ts in timesteps]
    flow_rates = [mean_flow_rates[ts] for ts in timesteps]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Viscous Energy Loss Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Total viscous energy loss over time
    axes[0, 0].plot(
        timesteps, energy_loss_mw, "o-", linewidth=2, color="red", markersize=6
    )
    axes[0, 0].set_title("Total Viscous Energy Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Energy Loss (mW)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="both", which="major", labelsize=10)

    # Plot 2: Average dissipation rate
    axes[0, 1].plot(
        timesteps, dissipation_rates, "o-", linewidth=2, color="blue", markersize=6
    )
    axes[0, 1].set_title("Average Viscous Dissipation Rate", fontweight="bold")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Dissipation Rate (W/m³)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis="both", which="major", labelsize=10)

    # Plot 3: Distribution of energy loss values
    axes[1, 0].hist(
        energy_loss_mw, bins=15, color="orange", alpha=0.7, edgecolor="black"
    )
    axes[1, 0].set_title("Distribution of Energy Loss Values", fontweight="bold")
    axes[1, 0].set_xlabel("Energy Loss (mW)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=10)

    # Plot 4: Energy loss vs flow rate correlation
    axes[1, 1].scatter(flow_rates, energy_loss_mw, alpha=0.7, s=50, c="purple")
    axes[1, 1].set_title("Energy Loss vs Flow Rate", fontweight="bold")
    axes[1, 1].set_xlabel("Mean Flow Rate (ml/s)")
    axes[1, 1].set_ylabel("Energy Loss (mW)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=10)

    # Calculate and display correlation coefficient
    corr_coef = np.corrcoef(flow_rates, energy_loss_mw)[0, 1]
    axes[1, 1].text(
        0.05,
        0.95,
        f"r = {corr_coef:.3f}",
        transform=axes[1, 1].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, corr_coef
