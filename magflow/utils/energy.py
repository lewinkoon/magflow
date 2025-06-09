from typing import Any

import numpy as np

from magflow.utils.visualization import extract_aorta


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


def calculate_patient_energy_loss(
    timesteps: list[int], timestep_data: dict[int, Any], biomodel: Any, mu: float
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Calculate energy loss for a single patient across all timesteps.

    Parameters:
    -----------
    timesteps : List[int]
        List of timesteps to process
    timestep_data : Dict[int, Any]
        Dictionary mapping timesteps to datasets
    biomodel : Any
        Patient biomodel data
    mu : float
        Blood viscosity

    Returns:
    --------
    Tuple of dictionaries containing energy loss, total dissipation, and avg dissipation rate
    """
    energy_loss = {}
    total_dissipation = {}
    avg_dissipation_rate = {}

    for ts in timesteps:
        dataset = timestep_data[ts]
        aorta = extract_aorta(dataset, biomodel)

        if aorta.n_points == 0 or "Velocity" not in aorta.point_data:
            # Skip this timestep if no valid data
            energy_loss[ts] = 0
            total_dissipation[ts] = 0
            avg_dissipation_rate[ts] = 0
            continue

        # Calculate energy loss components
        grad_tensor = calculate_gradient_tensor(aorta)
        strain_rate = calculate_strain_rate_from_gradient(grad_tensor)
        dissipation = calculate_viscous_dissipation_from_strain(strain_rate, mu)
        cell_volumes = estimate_cell_volumes(aorta)
        cell_dissipation = map_point_data_to_cells(aorta, dissipation)

        # Store results
        energy_loss[ts] = np.sum(cell_dissipation * cell_volumes)  # W (J/s)
        total_dissipation[ts] = np.sum(cell_dissipation)
        avg_dissipation_rate[ts] = np.mean(dissipation)

    return energy_loss, total_dissipation, avg_dissipation_rate


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


def plot_energy_loss_time_series(
    ax, patient_ids, all_viscous_energy_loss, patient_colors
):
    """Plot energy loss time series for all patients."""
    for patient_id in patient_ids:
        if patient_id in all_viscous_energy_loss:
            patient_timesteps = sorted(all_viscous_energy_loss[patient_id].keys())
            energy_loss_mw = [
                all_viscous_energy_loss[patient_id][ts] * 1000
                for ts in patient_timesteps
            ]

            ax.plot(
                patient_timesteps,
                energy_loss_mw,
                "o-",
                label=patient_id,
                color=patient_colors[patient_id],
                linewidth=2,
                markersize=4,
            )

    ax.set_title("Viscous Energy Loss by Patient", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy Loss (mW)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_dissipation_rate_time_series(
    ax, patient_ids, all_avg_dissipation_rate, patient_colors
):
    """Plot average dissipation rate time series for all patients."""
    for patient_id in patient_ids:
        if patient_id in all_avg_dissipation_rate:
            patient_timesteps = sorted(all_avg_dissipation_rate[patient_id].keys())
            avg_diss_values = [
                all_avg_dissipation_rate[patient_id][ts] for ts in patient_timesteps
            ]

            ax.plot(
                patient_timesteps,
                avg_diss_values,
                "o-",
                label=patient_id,
                color=patient_colors[patient_id],
                linewidth=2,
                markersize=4,
            )

    ax.set_title("Average Dissipation Rate by Patient", fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg Dissipation Rate (W/m³)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def calculate_peak_energies(patient_ids, all_viscous_energy_loss):
    """Calculate peak energy values for each patient."""
    peak_energies = []
    for patient_id in patient_ids:
        if patient_id in all_viscous_energy_loss:
            patient_energy_mw = [
                all_viscous_energy_loss[patient_id][ts] * 1000
                for ts in all_viscous_energy_loss[patient_id]
            ]
            peak_energies.append(max(patient_energy_mw) if patient_energy_mw else 0)
        else:
            peak_energies.append(0)
    return peak_energies


def calculate_mean_energies(patient_ids, all_viscous_energy_loss):
    """Calculate mean energy values for each patient."""
    mean_energies = []
    for patient_id in patient_ids:
        if patient_id in all_viscous_energy_loss:
            patient_energy_mw = [
                all_viscous_energy_loss[patient_id][ts] * 1000
                for ts in all_viscous_energy_loss[patient_id]
            ]
            mean_energies.append(np.mean(patient_energy_mw) if patient_energy_mw else 0)
        else:
            mean_energies.append(0)
    return mean_energies


def plot_energy_comparison_bars(ax, patient_ids, values, patient_colors, title, ylabel):
    """Plot bar comparison for energy values."""
    bars = ax.bar(
        patient_ids,
        values,
        color=[patient_colors[pid] for pid in patient_ids],
        alpha=0.7,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=False):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )


def calculate_energy_statistics(
    patient_ids: list[str],
    all_viscous_energy_loss: dict[str, dict],
    all_flow_rates: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Calculate comprehensive statistics for energy loss analysis.

    Parameters:
    -----------
    patient_ids : List[str]
        List of patient IDs
    all_viscous_energy_loss : Dict[str, Dict]
        Energy loss data for all patients
    all_flow_rates : Optional[Dict[str, Dict]]
        Flow rate data for correlation analysis

    Returns:
    --------
    Dictionary containing statistics for each patient
    """
    patient_statistics = {}

    for patient_id in patient_ids:
        if patient_id not in all_viscous_energy_loss:
            patient_statistics[patient_id] = {"error": "No data available"}
            continue

        # Convert to mW for analysis
        patient_timesteps = list(all_viscous_energy_loss[patient_id].keys())
        energy_loss_mw = [
            all_viscous_energy_loss[patient_id][ts] * 1000 for ts in patient_timesteps
        ]

        if not energy_loss_mw:
            patient_statistics[patient_id] = {"error": "Empty data"}
            continue

        # Basic statistics
        stats = {
            "peak_energy": max(energy_loss_mw),
            "mean_energy": np.mean(energy_loss_mw),
            "min_energy": min(energy_loss_mw),
            "peak_timestep": max(
                all_viscous_energy_loss[patient_id],
                key=all_viscous_energy_loss[patient_id].get,
            ),
            "energy_efficiency_ratio": min(energy_loss_mw) / max(energy_loss_mw),
        }

        # Cardiac cycle analysis
        cycle_duration = 1.0  # seconds
        stats["total_cycle_energy"] = np.trapezoid(
            list(all_viscous_energy_loss[patient_id].values()),
            dx=cycle_duration / len(patient_timesteps),
        )

        # Flow rate correlation if available
        if all_flow_rates and patient_id in all_flow_rates:
            correlation_data = calculate_flow_correlation(
                patient_id, all_viscous_energy_loss, all_flow_rates
            )
            stats.update(correlation_data)

        patient_statistics[patient_id] = stats

    return patient_statistics


def calculate_flow_correlation(patient_id, all_viscous_energy_loss, all_flow_rates):
    """Calculate correlation between energy loss and flow rates."""
    common_timesteps = sorted(
        set(all_viscous_energy_loss[patient_id].keys())
        & set(all_flow_rates[patient_id].keys())
    )

    if len(common_timesteps) <= 1:
        return {"flow_correlation": None, "correlation_strength": "insufficient_data"}

    patient_flow_rates = [all_flow_rates[patient_id][ts] for ts in common_timesteps]
    patient_energy_loss = [
        all_viscous_energy_loss[patient_id][ts] * 1000 for ts in common_timesteps
    ]

    correlation_coef = np.corrcoef(patient_flow_rates, patient_energy_loss)[0, 1]

    if abs(correlation_coef) > 0.7:
        strength = "strong"
    elif abs(correlation_coef) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    return {"flow_correlation": correlation_coef, "correlation_strength": strength}


def print_energy_analysis_summary(
    patient_ids: list[str],
    patient_statistics: dict[str, dict],
    all_viscous_energy_loss: dict[str, dict],
):
    """
    Print comprehensive summary of energy loss analysis.

    Parameters:
    -----------
    patient_ids : List[str]
        List of patient IDs
    patient_statistics : Dict[str, Dict]
        Statistics for each patient
    all_viscous_energy_loss : Dict[str, Dict]
        Energy loss data for cross-patient comparison
    """
    print("\nVISCOUS ENERGY LOSS SUMMARY - ALL PATIENTS")
    print("=" * 50)

    # Individual patient summaries
    for patient_id in patient_ids:
        if patient_id not in patient_statistics:
            print(f"\nPatient {patient_id}: No data available")
            continue

        stats = patient_statistics[patient_id]
        if "error" in stats:
            print(f"\nPatient {patient_id}: {stats['error']}")
            continue

        print(f"\nPatient {patient_id}:")
        print("-" * 20)
        print(f"Peak energy loss: {stats['peak_energy']:.2f} mW")
        print(f"Mean energy loss: {stats['mean_energy']:.2f} mW")
        print(f"Minimum energy loss: {stats['min_energy']:.2f} mW")
        print(f"Peak occurs at timestep: {stats['peak_timestep']}")
        print(f"Total energy per cycle: {stats['total_cycle_energy']:.4f} J")
        print(f"Energy efficiency ratio: {stats['energy_efficiency_ratio']:.3f}")

        if "flow_correlation" in stats and stats["flow_correlation"] is not None:
            print(f"Flow rate correlation: {stats['flow_correlation']:.3f}")
            print(f"Correlation strength: {stats['correlation_strength']}")

    # Cross-patient comparison
    print_cross_patient_comparison(patient_ids, patient_statistics)


def print_cross_patient_comparison(patient_ids, patient_statistics):
    """Print cross-patient comparison statistics."""
    valid_stats = {
        pid: stats for pid, stats in patient_statistics.items() if "error" not in stats
    }

    if not valid_stats:
        print("\nInsufficient data for cross-patient comparison")
        return

    peak_energies = [stats["peak_energy"] for stats in valid_stats.values()]
    mean_energies = [stats["mean_energy"] for stats in valid_stats.values()]

    print("\nCROSS-PATIENT COMPARISON")
    print("-" * 30)

    max_peak_patient = max(
        valid_stats.keys(), key=lambda p: valid_stats[p]["peak_energy"]
    )
    min_peak_patient = min(
        valid_stats.keys(), key=lambda p: valid_stats[p]["peak_energy"]
    )
    max_mean_patient = max(
        valid_stats.keys(), key=lambda p: valid_stats[p]["mean_energy"]
    )
    min_mean_patient = min(
        valid_stats.keys(), key=lambda p: valid_stats[p]["mean_energy"]
    )

    print(f"Highest peak energy loss: {max(peak_energies):.2f} mW ({max_peak_patient})")
    print(f"Lowest peak energy loss: {min(peak_energies):.2f} mW ({min_peak_patient})")
    print(f"Highest mean energy loss: {max(mean_energies):.2f} mW ({max_mean_patient})")
    print(f"Lowest mean energy loss: {min(mean_energies):.2f} mW ({min_mean_patient})")

    # Overall statistics
    print("\nOverall statistics:")
    print(f"Average peak energy loss across patients: {np.mean(peak_energies):.2f} mW")
    print(f"Average mean energy loss across patients: {np.mean(mean_energies):.2f} mW")
    print(f"Peak energy loss range: {max(peak_energies) - min(peak_energies):.2f} mW")
