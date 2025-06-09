from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from magflow.utils.visualization import extract_aorta


def calculate_vorticity(aorta, convert_velocity=True):
    """
    Calculate vorticity for a single timestep dataset.

    Parameters:
    -----------
    aorta : pv.UnstructuredGrid
        Aorta dataset with velocity field
    convert_velocity : bool
        Whether to convert velocity from cm/s to mm/s

    Returns:
    --------
    tuple: (aorta_with_vorticity, vorticity_magnitude, max_vorticity, avg_vorticity)
    """
    if aorta.n_points == 0:
        return None, 0, 0, 0

    try:
        # Calculate velocity gradients using PyVista's built-in functionality
        velocity = aorta["Velocity"]
        if convert_velocity:
            velocity = velocity * 10  # Convert from cm/s to mm/s

        # Add velocity components as separate scalar fields
        aorta["vx"] = velocity[:, 0]
        aorta["vy"] = velocity[:, 1]
        aorta["vz"] = velocity[:, 2]

        # Calculate gradients using PyVista's compute_derivative method
        grad_vx = aorta.compute_derivative(scalars="vx")
        grad_vy = aorta.compute_derivative(scalars="vy")
        grad_vz = aorta.compute_derivative(scalars="vz")

        # Extract gradient arrays (default gradient field name is "gradient")
        _dvx_dx, dvx_dy, dvx_dz = (
            grad_vx["gradient"][:, 0],
            grad_vx["gradient"][:, 1],
            grad_vx["gradient"][:, 2],
        )
        dvy_dx, _dvy_dy, dvy_dz = (
            grad_vy["gradient"][:, 0],
            grad_vy["gradient"][:, 1],
            grad_vy["gradient"][:, 2],
        )
        dvz_dx, dvz_dy, _dvz_dz = (
            grad_vz["gradient"][:, 0],
            grad_vz["gradient"][:, 1],
            grad_vz["gradient"][:, 2],
        )

        # Calculate vorticity components (curl of velocity field)
        # ω_x = ∂v_z/∂y - ∂v_y/∂z
        # ω_y = ∂v_x/∂z - ∂v_z/∂x
        # ω_z = ∂v_y/∂x - ∂v_x/∂y
        omega_x = dvz_dy - dvy_dz
        omega_y = dvx_dz - dvz_dx
        omega_z = dvy_dx - dvx_dy

        # Create vorticity vector field
        vorticity_vector = np.column_stack([omega_x, omega_y, omega_z])

        # Calculate vorticity magnitude
        vorticity_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

        # Store results
        aorta["Vorticity"] = vorticity_vector
        aorta["VorticityMagnitude"] = vorticity_mag

        return (
            aorta.copy(),
            vorticity_mag,
            np.max(vorticity_mag),
            np.mean(vorticity_mag),
        )

    except Exception as e:
        print(f"Error calculating vorticity: {e}")
        return None, 0, 0, 0


def calculate_patient_vorticity(
    patient_id: str, patient_data: dict[str, Any]
) -> tuple[dict, dict, dict, dict]:
    """
    Calculate vorticity for a single patient across all timesteps.

    Args:
        patient_id: Patient identifier
        patient_data: Patient-specific data containing timesteps and biomodel

    Returns:
        Tuple containing patient vorticity data, magnitude, max, and average values
    """
    patient_vorticity_data = {}
    patient_vorticity_magnitude = {}
    patient_max_vorticity = {}
    patient_avg_vorticity = {}

    patient_timesteps = list(patient_data["timesteps"].keys())
    patient_timestep_data = patient_data["timesteps"]
    patient_biomodel = patient_data["biomodel"]

    for ts in patient_timesteps:
        dataset = patient_timestep_data[ts]
        aorta = extract_aorta(dataset, patient_biomodel)

        aorta_result, vort_mag, max_vort, avg_vort = calculate_vorticity(aorta)

        patient_vorticity_data[ts] = aorta_result
        patient_vorticity_magnitude[ts] = vort_mag
        patient_max_vorticity[ts] = max_vort
        patient_avg_vorticity[ts] = avg_vort

    return (
        patient_vorticity_data,
        patient_vorticity_magnitude,
        patient_max_vorticity,
        patient_avg_vorticity,
    )


def create_vorticity_visualization(
    patient_ids: list[str],
    all_max_vorticity: dict[str, dict],
    all_avg_vorticity: dict[str, dict],
    patient_colors: dict[str, str],
) -> plt.Figure:
    """
    Create comprehensive multi-patient vorticity visualization.

    Args:
        patient_ids: List of patient identifiers
        all_max_vorticity: Maximum vorticity data for all patients
        all_avg_vorticity: Average vorticity data for all patients
        patient_colors: Color mapping for each patient

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-Patient Vorticity Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Maximum vorticity time series
    plot_vorticity_time_series(
        axes[0, 0],
        patient_ids,
        all_max_vorticity,
        patient_colors,
        "Maximum Vorticity by Timestep",
        "Max Vorticity (s⁻¹)",
    )

    # Plot 2: Average vorticity time series
    plot_vorticity_time_series(
        axes[0, 1],
        patient_ids,
        all_avg_vorticity,
        patient_colors,
        "Average Vorticity by Timestep",
        "Avg Vorticity (s⁻¹)",
    )

    # Plot 3: Peak maximum vorticity comparison
    peak_values = calculate_peak_vorticity_by_patient(patient_ids, all_max_vorticity)
    plot_vorticity_comparison_bars(
        axes[1, 0],
        peak_values,
        patient_colors,
        "Peak Maximum Vorticity Comparison",
        "Peak Max Vorticity (s⁻¹)",
    )

    # Plot 4: Average maximum vorticity comparison
    avg_values = calculate_average_vorticity_by_patient(patient_ids, all_max_vorticity)
    plot_vorticity_comparison_bars(
        axes[1, 1],
        avg_values,
        patient_colors,
        "Average Maximum Vorticity Comparison",
        "Avg Max Vorticity (s⁻¹)",
    )

    plt.tight_layout()
    return fig


def plot_vorticity_time_series(
    ax: plt.Axes,
    patient_ids: list[str],
    vorticity_data: dict[str, dict],
    patient_colors: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    """Plot vorticity time series for multiple patients."""
    for patient_id in patient_ids:
        if patient_id in vorticity_data:
            patient_timesteps = list(vorticity_data[patient_id].keys())
            vort_values = [vorticity_data[patient_id][ts] for ts in patient_timesteps]

            ax.plot(
                patient_timesteps,
                vort_values,
                "o-",
                linewidth=2,
                color=patient_colors[patient_id],
                markersize=4,
                label=patient_id,
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis="both", which="major", labelsize=10)


def calculate_peak_vorticity_by_patient(
    patient_ids: list[str], all_max_vorticity: dict[str, dict]
) -> dict[str, float]:
    """Calculate peak maximum vorticity for each patient."""
    peak_values = {}
    for patient_id in patient_ids:
        if all_max_vorticity.get(patient_id):
            peak_values[patient_id] = max(all_max_vorticity[patient_id].values())
    return peak_values


def calculate_average_vorticity_by_patient(
    patient_ids: list[str], all_max_vorticity: dict[str, dict]
) -> dict[str, float]:
    """Calculate average maximum vorticity for each patient."""
    avg_values = {}
    for patient_id in patient_ids:
        if all_max_vorticity.get(patient_id):
            avg_values[patient_id] = np.mean(
                list(all_max_vorticity[patient_id].values())
            )
    return avg_values


def plot_vorticity_comparison_bars(
    ax: plt.Axes,
    values_dict: dict[str, float],
    patient_colors: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    """Plot bar comparison of vorticity values between patients."""
    if not values_dict:
        return

    patients = list(values_dict.keys())
    values = list(values_dict.values())
    colors_list = [patient_colors[p] for p in patients]

    bars = ax.bar(patients, values, color=colors_list, alpha=0.7, edgecolor="black")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Patient")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def calculate_vorticity_statistics(
    patient_ids: list[str],
    all_max_vorticity: dict[str, dict],
    all_avg_vorticity: dict[str, dict],
    all_flow_rates: dict[str, dict] | None = None,
) -> tuple[dict[str, dict], dict[str, float]]:
    """
    Calculate comprehensive vorticity statistics for all patients.

    Args:
        patient_ids: List of patient identifiers
        all_max_vorticity: Maximum vorticity data
        all_avg_vorticity: Average vorticity data
        all_flow_rates: Optional flow rate data for correlation analysis

    Returns:
        Tuple containing patient-specific stats and cross-patient stats
    """
    patient_stats = {}

    for patient_id in patient_ids:
        if all_max_vorticity.get(patient_id):
            max_vort_values = list(all_max_vorticity[patient_id].values())
            avg_vort_values = list(all_avg_vorticity[patient_id].values())

            stats = {
                "peak_max_vorticity": max(max_vort_values),
                "mean_max_vorticity": np.mean(max_vort_values),
                "min_max_vorticity": min(max_vort_values),
                "peak_avg_vorticity": max(avg_vort_values),
                "mean_avg_vorticity": np.mean(avg_vort_values),
                "peak_timestep": max(
                    all_max_vorticity[patient_id], key=all_max_vorticity[patient_id].get
                ),
            }

            # Calculate correlation with flow rate if available
            if all_flow_rates and patient_id in all_flow_rates:
                patient_flow_rates = list(all_flow_rates[patient_id].values())
                if len(patient_flow_rates) == len(max_vort_values):
                    corr_coef = np.corrcoef(patient_flow_rates, max_vort_values)[0, 1]
                    stats["flow_correlation"] = corr_coef
                    stats["correlation_strength"] = get_correlation_strength(corr_coef)

            patient_stats[patient_id] = stats

    # Cross-patient comparison
    cross_patient_stats = calculate_cross_patient_statistics(
        patient_ids, all_max_vorticity
    )

    return patient_stats, cross_patient_stats


def calculate_cross_patient_statistics(
    patient_ids: list[str], all_max_vorticity: dict[str, dict]
) -> dict[str, float]:
    """Calculate statistics across all patients."""
    if len(all_max_vorticity) <= 1:
        return {}

    all_peak_values = [
        max(all_max_vorticity[pid].values())
        for pid in patient_ids
        if all_max_vorticity.get(pid)
    ]

    return {
        "highest_peak": max(all_peak_values),
        "lowest_peak": min(all_peak_values),
        "mean_peak": np.mean(all_peak_values),
        "std_peak": np.std(all_peak_values),
    }


def get_correlation_strength(corr_coef: float) -> str:
    """Determine correlation strength category."""
    abs_corr = abs(corr_coef)
    if abs_corr > 0.7:
        return "strong"
    elif abs_corr > 0.3:
        return "moderate"
    else:
        return "weak"


def print_vorticity_summary(
    patient_stats: dict[str, dict], cross_patient_stats: dict[str, float]
) -> None:
    """Print comprehensive vorticity analysis summary."""
    print("\nMULTI-PATIENT VORTICITY ANALYSIS SUMMARY")
    print("=" * 50)

    for patient_id, stats in patient_stats.items():
        print(f"\nPATIENT {patient_id}")
        print("-" * 20)
        print(f"Peak maximum vorticity: {stats['peak_max_vorticity']:.1f} s⁻¹")
        print(f"Mean maximum vorticity: {stats['mean_max_vorticity']:.1f} s⁻¹")
        print(f"Minimum maximum vorticity: {stats['min_max_vorticity']:.1f} s⁻¹")
        print(f"Peak occurs at timestep: {stats['peak_timestep']}")
        print(f"Peak average vorticity: {stats['peak_avg_vorticity']:.1f} s⁻¹")
        print(f"Mean average vorticity: {stats['mean_avg_vorticity']:.1f} s⁻¹")

        if "flow_correlation" in stats:
            print(f"Flow rate correlation: {stats['flow_correlation']:.3f}")
            print(f"Correlation strength: {stats['correlation_strength']}")

    if cross_patient_stats:
        print("\nCROSS-PATIENT COMPARISON")
        print("-" * 30)
        print(
            f"Highest peak vorticity across patients: {cross_patient_stats['highest_peak']:.1f} s⁻¹"
        )
        print(
            f"Lowest peak vorticity across patients: {cross_patient_stats['lowest_peak']:.1f} s⁻¹"
        )
        print(
            f"Mean peak vorticity across patients: {cross_patient_stats['mean_peak']:.1f} s⁻¹"
        )
        print(
            f"Standard deviation of peak vorticity: {cross_patient_stats['std_peak']:.1f} s⁻¹"
        )
