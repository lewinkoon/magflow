import numpy as np
from matplotlib import pyplot as plt

import magflow.utils.visualization as viz


def estimate_pressure_gradient_bernoulli(dataset, rho=1050):
    """
    Estimate pressure gradient using Bernoulli equation approach.
    For PyVista 0.45.2
    """
    # Get velocity data
    velocity = dataset.point_data.get("velocity", dataset.point_data.get("Velocity"))
    if velocity is None:
        raise ValueError("No velocity data found in dataset")

    # Calculate velocity magnitude
    vel_magnitude = np.linalg.norm(velocity, axis=1)

    # Add velocity magnitude as point data
    dataset.point_data["vel_magnitude"] = vel_magnitude

    # Compute gradient - correct syntax for PyVista 0.45.2
    dataset_with_grad = dataset.compute_derivative(
        scalars="vel_magnitude",
        gradient="vel_grad",  # Note: it's 'gradient', not 'gradient_name'
    )
    vel_grad = dataset_with_grad.point_data["vel_grad"]

    # Calculate pressure gradient using Bernoulli equation
    pressure_gradient = np.zeros_like(velocity)
    for i in range(len(velocity)):
        if vel_magnitude[i] > 1e-10:
            pressure_gradient[i] = -rho * velocity[i] * vel_grad[i] / vel_magnitude[i]

    return pressure_gradient


def estimate_pressure_gradient_navier_stokes(dataset, rho=1050, mu=0.004):
    """
    Estimate pressure gradient using Navier-Stokes equation approach.
    """
    velocity = dataset.point_data.get("velocity", dataset.point_data.get("Velocity"))
    if velocity is None:
        raise ValueError("No velocity data found in dataset")

    pressure_gradient = np.zeros_like(velocity)

    for i, component in enumerate(["x", "y", "z"]):
        # Add velocity component as scalar data
        dataset.point_data[f"vel_{component}"] = velocity[:, i]

        # Compute gradient - correct parameter name
        dataset_with_grad = dataset.compute_derivative(
            scalars=f"vel_{component}",
            gradient=f"vel_{component}_grad",  # Note: 'gradient', not 'gradient_name'
        )
        vel_grad = dataset_with_grad.point_data[f"vel_{component}_grad"]

        # Calculate convective term
        convective_term = np.sum(velocity * vel_grad, axis=1)
        pressure_gradient[:, i] = -rho * convective_term

    return pressure_gradient


def calculate_patient_pressure_gradients(
    patient_id, patient_timesteps, patient_biomodel, rho=1050, mu=0.004
):
    """
    Calculate pressure gradients for a single patient across all timesteps.

    Args:
        patient_id: Patient identifier for logging
        patient_timesteps: Dictionary of timestep data
        patient_biomodel: Patient biomodel for aorta extraction
        rho: Blood density in kg/m³
        mu: Blood viscosity in Pa·s

    Returns:
        Tuple of (pressure_gradients, pressure_drops) dictionaries
    """
    pressure_gradients = {}
    pressure_drops = {}

    print(f"Calculating pressure gradients for patient {patient_id}")

    for ts in patient_timesteps:
        dataset = patient_timesteps[ts]

        # Extract aorta region (you'll need to implement this or import from viz)
        aorta = viz.extract_aorta(dataset, patient_biomodel)

        # Method 1: Bernoulli approach
        pressure_grad_bernoulli = estimate_pressure_gradient_bernoulli(aorta, rho)

        # Method 2: Navier-Stokes approach
        pressure_grad_ns = estimate_pressure_gradient_navier_stokes(aorta, rho, mu)

        # Store results
        pressure_gradients[ts] = {
            "bernoulli": pressure_grad_bernoulli,
            "navier_stokes": pressure_grad_ns,
            "magnitude_bernoulli": np.linalg.norm(pressure_grad_bernoulli, axis=1),
            "magnitude_ns": np.linalg.norm(pressure_grad_ns, axis=1),
        }

        # Calculate maximum pressure gradient magnitudes
        pressure_drops[ts] = {
            "max_grad_bernoulli": np.max(pressure_gradients[ts]["magnitude_bernoulli"]),
            "max_grad_ns": np.max(pressure_gradients[ts]["magnitude_ns"]),
            "avg_grad_bernoulli": np.mean(
                pressure_gradients[ts]["magnitude_bernoulli"]
            ),
            "avg_grad_ns": np.mean(pressure_gradients[ts]["magnitude_ns"]),
        }

    return pressure_gradients, pressure_drops


def plot_pressure_gradient_time_series(
    ax, patient_ids, pressure_data, colors, title, ylabel, metric_key
):
    """
    Plot pressure gradient time series for multiple patients.

    Args:
        ax: Matplotlib axis object
        patient_ids: List of patient IDs
        pressure_data: Dictionary with pressure gradient data
        colors: Dictionary mapping patient IDs to colors
        title: Plot title
        ylabel: Y-axis label
        metric_key: Key to extract from pressure data (e.g., 'max_grad_bernoulli')
    """
    for patient_id in patient_ids:
        if pressure_data.get(patient_id):
            timesteps = sorted(pressure_data[patient_id].keys())
            values = [pressure_data[patient_id][ts][metric_key] for ts in timesteps]

            ax.plot(
                timesteps,
                values,
                marker="o",
                linewidth=2,
                markersize=4,
                label=patient_id,
                color=colors.get(patient_id, "black"),
                alpha=0.8,
            )

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_pressure_gradient_comparison_bars(
    ax, patient_ids, peak_data, colors, title, ylabel
):
    """
    Plot bar comparison of peak pressure gradients across patients.

    Args:
        ax: Matplotlib axis object
        patient_ids: List of patient IDs
        peak_data: Dictionary with peak pressure gradient values
        colors: Dictionary mapping patient IDs to colors
        title: Plot title
        ylabel: Y-axis label
    """
    patients = []
    values = []
    bar_colors = []

    for patient_id in patient_ids:
        if patient_id in peak_data:
            patients.append(patient_id)
            values.append(peak_data[patient_id])
            bar_colors.append(colors.get(patient_id, "gray"))

    bars = ax.bar(patients, values, color=bar_colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("Patient ID", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")


def calculate_peak_pressure_gradients(patient_ids, pressure_data, metric_key):
    """
    Calculate peak pressure gradients for each patient.

    Args:
        patient_ids: List of patient IDs
        pressure_data: Dictionary with pressure gradient data
        metric_key: Key to extract from pressure data

    Returns:
        Dictionary mapping patient IDs to peak values
    """
    peak_values = {}

    for patient_id in patient_ids:
        if pressure_data.get(patient_id):
            values = [
                pressure_data[patient_id][ts][metric_key]
                for ts in pressure_data[patient_id]
            ]
            peak_values[patient_id] = max(values) if values else 0

    return peak_values


def plot_pressure_gradient_distribution(patient_ids, pressure_gradients, colors):
    """
    Create distribution plots for pressure gradients.

    Args:
        patient_ids: List of patient IDs
        pressure_gradients: Dictionary with detailed pressure gradient data
        colors: Dictionary mapping patient IDs to colors
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Pressure Gradient Distribution Analysis", fontsize=16, fontweight="bold"
    )

    # Plot 1: Bernoulli magnitude distribution
    for patient_id in patient_ids:
        if patient_id in pressure_gradients:
            all_magnitudes = []
            for ts_data in pressure_gradients[patient_id].values():
                all_magnitudes.extend(ts_data["magnitude_bernoulli"])

            if all_magnitudes:
                axes[0, 0].hist(
                    all_magnitudes,
                    bins=50,
                    alpha=0.5,
                    label=patient_id,
                    color=colors.get(patient_id, "gray"),
                )

    axes[0, 0].set_title("Bernoulli Pressure Gradient Distribution")
    axes[0, 0].set_xlabel("Pressure Gradient Magnitude (Pa/m)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Navier-Stokes magnitude distribution
    for patient_id in patient_ids:
        if patient_id in pressure_gradients:
            all_magnitudes = []
            for ts_data in pressure_gradients[patient_id].values():
                all_magnitudes.extend(ts_data["magnitude_ns"])

            if all_magnitudes:
                axes[0, 1].hist(
                    all_magnitudes,
                    bins=50,
                    alpha=0.5,
                    label=patient_id,
                    color=colors.get(patient_id, "gray"),
                )

    axes[0, 1].set_title("Navier-Stokes Pressure Gradient Distribution")
    axes[0, 1].set_xlabel("Pressure Gradient Magnitude (Pa/m)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Method comparison scatter plot
    for patient_id in patient_ids:
        if patient_id in pressure_gradients:
            bernoulli_vals = []
            ns_vals = []

            for ts_data in pressure_gradients[patient_id].values():
                bernoulli_vals.extend(ts_data["magnitude_bernoulli"])
                ns_vals.extend(ts_data["magnitude_ns"])

            if bernoulli_vals and ns_vals:
                axes[1, 0].scatter(
                    bernoulli_vals,
                    ns_vals,
                    alpha=0.5,
                    label=patient_id,
                    color=colors.get(patient_id, "gray"),
                )

    axes[1, 0].set_title("Bernoulli vs Navier-Stokes Comparison")
    axes[1, 0].set_xlabel("Bernoulli Gradient (Pa/m)")
    axes[1, 0].set_ylabel("Navier-Stokes Gradient (Pa/m)")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.3, transform=axes[1, 0].transAxes)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Average gradients comparison
    avg_bernoulli = calculate_average_pressure_gradients(
        patient_ids, pressure_gradients, "avg_grad_bernoulli"
    )
    avg_ns = calculate_average_pressure_gradients(
        patient_ids, pressure_gradients, "avg_grad_ns"
    )

    x_pos = np.arange(len(patient_ids))
    width = 0.35

    axes[1, 1].bar(
        x_pos - width / 2,
        [avg_bernoulli.get(pid, 0) for pid in patient_ids],
        width,
        label="Bernoulli",
        alpha=0.7,
    )
    axes[1, 1].bar(
        x_pos + width / 2,
        [avg_ns.get(pid, 0) for pid in patient_ids],
        width,
        label="Navier-Stokes",
        alpha=0.7,
    )

    axes[1, 1].set_title("Average Pressure Gradients Comparison")
    axes[1, 1].set_xlabel("Patient ID")
    axes[1, 1].set_ylabel("Average Pressure Gradient (Pa/m)")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(patient_ids, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def calculate_average_pressure_gradients(patient_ids, pressure_data, metric_key):
    """
    Calculate average pressure gradients for each patient.

    Args:
        patient_ids: List of patient IDs
        pressure_data: Dictionary with pressure gradient data
        metric_key: Key to extract from pressure data

    Returns:
        Dictionary mapping patient IDs to average values
    """
    avg_values = {}

    for patient_id in patient_ids:
        if pressure_data.get(patient_id):
            values = [
                pressure_data[patient_id][ts][metric_key]
                for ts in pressure_data[patient_id]
            ]
            avg_values[patient_id] = np.mean(values) if values else 0

    return avg_values
