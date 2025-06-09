import math

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from rich import print


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


def render_patient(
    plotter: pv.Plotter,
    patient_id: str,
    patient_data: dict,
    row: int,
    col: int,
    selected_timestep: int | None = None,
) -> int | None:
    """
    Add a single patient's visualization to a subplot.

    Args:
        plotter: PyVista plotter instance
        patient_id: Patient identifier
        patient_data: Patient's complete data dictionary
        row: Subplot row position
        col: Subplot column position
        selected_timestep: Specific timestep to use

    Returns:
        Timestep actually used for visualization
    """
    # Activate the appropriate subplot
    plotter.subplot(row, col)

    # Get patient timestep data
    timesteps = patient_data["timesteps"]

    # Select timestep (use specified or middle timestep)
    if selected_timestep and selected_timestep in timesteps:
        dataset = timesteps[selected_timestep]
        ts_used = selected_timestep
    else:
        sorted_ts = sorted(timesteps.keys())
        ts_used = sorted_ts[len(sorted_ts) // 2]
        dataset = timesteps[ts_used]

    # Get patient-specific biomodel
    biomodel = patient_data["biomodel"]

    # Extract aorta region
    aorta = extract_aorta(dataset, biomodel)

    # Add biomodel wireframe
    plotter.add_mesh(
        biomodel,
        color="white",
        opacity=0.1,
        show_edges=False,
    )

    # Add centerline if available
    if "centerline" in patient_data:
        plotter.add_mesh(
            patient_data["centerline"],
            color="blue",
            line_width=3,
            render_lines_as_tubes=True,
            opacity=0.8,
        )
    else:
        print(f"Patient {patient_id}: No centerline data available")

    # Add volume rendering
    plotter.add_volume(
        aorta,
        scalars="VelocityMagnitude",
        cmap="coolwarm",
        opacity="sigmoid_5",
        scalar_bar_args={
            "title": "Velocity (cm/s)",
            "position_x": 0.85,
            "position_y": 0.1,
            "width": 0.08,
            "height": 0.8,
            "n_labels": 5,
            "fmt": "%.0f",
            "vertical": True,
            "title_font_size": 12,
            "label_font_size": 12,
        },
    )

    # Configure viewing and add labels
    plotter.view_xy()
    plotter.add_text(
        f"Patient: {patient_id}\nTimestep: {ts_used}",
        position="upper_left",
        font_size=8,
        color="black",
    )


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


def cross_section(selected_point, points, dataset, radius=30, index=0):
    """Create a single orthogonal cross-section at the specified index along the centreline.

    Args:
        selected_point: The specific point along the centreline where to create the cross-section
        points: Array of all centreline points (needed for tangent calculation)
        dataset: The dataset to slice
        radius: Radius of the spherical clipping region
        index: Index of the selected point in the centreline array

    Returns:
        PyVista mesh representing the cross-section, or None if failed
    """
    try:
        # Calculate tangent vector at the selected point
        if index == 0:
            # For the first point, use forward difference
            tangent = points[1] - points[0]
        elif index == len(points) - 1:
            # For the last point, use backward difference
            tangent = points[-1] - points[-2]
        else:
            # For interior points, use central difference
            tangent = points[index + 1] - points[index - 1]

        # Normalize the tangent vector
        tangent = tangent / np.linalg.norm(tangent)

        # The normal to the plane is the tangent at this point
        normal = tangent

        # Slice the dataset with a plane at this point and normal
        full_slice = dataset.slice(normal=normal, origin=selected_point)

        # Create a sphere to limit the extent of the slice
        sphere = pv.Sphere(radius=radius, center=selected_point)

        # Use boolean intersection to limit the slice to the sphere's radius
        cross_section = full_slice.clip_surface(sphere)

        return cross_section

    except Exception as e:
        print(f"Error creating cross-section at index {index}: {e}")
        return None


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


def calculate_subplot_grid(n_items, max_cols=5):
    """
    Calculate optimal grid dimensions for subplot arrangement.

    Args:
        n_items (int): Number of items to display
        max_cols (int): Maximum number of columns (default: 5)

    Returns:
        tuple: (n_rows, n_cols) for subplot grid
    """
    if n_items == 0:
        return 0, 0
    elif n_items == 1:
        return 1, 1
    elif n_items <= 4:
        return 2, 2
    elif n_items <= 6:
        return 2, 3
    elif n_items <= 9:
        return 3, 3
    elif n_items <= 12:
        return 3, 4
    elif n_items <= 16:
        return 4, 4
    else:
        # For more than 16 items, use max_cols and calculate rows
        n_cols = min(max_cols, n_items)
        n_rows = math.ceil(n_items / n_cols)
        return n_rows, n_cols
