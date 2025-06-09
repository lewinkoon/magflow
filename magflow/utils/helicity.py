from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import magflow.utils.visualization as viz


def calculate_helicity_for_timestep(
    dataset: Any, biomodel: Any, vorticity_grid: Any | None
) -> tuple[np.ndarray | None, np.ndarray | None, float, float, float]:
    """
    Calculate helicity for a single timestep.

    Args:
        dataset: VTK dataset for the timestep
        biomodel: Biomodel data for aorta extraction
        vorticity_grid: Pre-calculated vorticity data

    Returns:
        Tuple of (helicity, helicity_magnitude, max_helicity, avg_helicity, abs_avg_helicity)
    """
    if vorticity_grid is None:
        return None, None, 0, 0, 0

    # Extract aorta and get velocity field
    aorta = viz.extract_aorta(dataset, biomodel)
    velocity = aorta["Velocity"] * 10  # Convert from cm/s to mm/s

    # Get vorticity data
    vorticity = vorticity_grid["Vorticity"]

    # Calculate helicity as dot product of velocity and vorticity
    helicity = np.sum(velocity * vorticity, axis=1)
    helicity_magnitude = np.abs(helicity)

    # Calculate statistics
    max_helicity = np.max(helicity_magnitude)
    avg_helicity = np.mean(helicity)
    abs_avg_helicity = np.mean(helicity_magnitude)

    return helicity, helicity_magnitude, max_helicity, avg_helicity, abs_avg_helicity


def calculate_patient_helicity(
    patient_id: str, patient_data: dict, vorticity_data: dict
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Calculate helicity for all timesteps of a single patient.

    Args:
        patient_id: Patient identifier
        patient_data: Patient data dictionary containing timesteps, biomodel
        vorticity_data: Pre-calculated vorticity data for the patient

    Returns:
        Tuple of dictionaries (helicity_data, helicity_magnitude, max_helicity,
                              avg_helicity, abs_avg_helicity)
    """
    timesteps = list(patient_data["timesteps"].keys())
    timestep_data = patient_data["timesteps"]
    biomodel = patient_data["biomodel"]

    # Initialize result dictionaries
    helicity_data = {}
    helicity_magnitude = {}
    max_helicity = {}
    avg_helicity = {}
    abs_avg_helicity = {}

    for ts in timesteps:
        dataset = timestep_data[ts]
        vorticity_grid = vorticity_data.get(ts)

        hel, hel_mag, max_hel, avg_hel, abs_avg_hel = calculate_helicity_for_timestep(
            dataset, biomodel, vorticity_grid
        )

        helicity_data[ts] = hel
        helicity_magnitude[ts] = hel_mag
        max_helicity[ts] = max_hel
        avg_helicity[ts] = avg_hel
        abs_avg_helicity[ts] = abs_avg_hel

    return (
        helicity_data,
        helicity_magnitude,
        max_helicity,
        avg_helicity,
        abs_avg_helicity,
    )


def plot_helicity_time_series(
    ax: plt.Axes,
    patient_ids: list[str],
    helicity_data: dict,
    patient_colors: dict,
    title: str,
    ylabel: str,
    add_zero_line: bool = False,
) -> None:
    """
    Plot helicity time series for multiple patients.

    Args:
        ax: Matplotlib axes object
        patient_ids: List of patient identifiers
        helicity_data: Dictionary containing helicity data for all patients
        patient_colors: Dictionary mapping patient IDs to colors
        title: Plot title
        ylabel: Y-axis label
        add_zero_line: Whether to add a horizontal line at y=0
    """
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if add_zero_line:
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    for patient_id in patient_ids:
        if helicity_data.get(patient_id):
            timesteps = list(helicity_data[patient_id].keys())
            values = list(helicity_data[patient_id].values())
            ax.plot(
                timesteps,
                values,
                "o-",
                linewidth=2,
                color=patient_colors[patient_id],
                markersize=6,
                label=patient_id,
            )

    ax.legend()
    ax.tick_params(axis="both", which="major", labelsize=10)


def plot_helicity_comparison_bars(
    ax: plt.Axes,
    patient_ids: list[str],
    helicity_data: dict,
    patient_colors: dict,
    title: str,
    ylabel: str,
) -> None:
    """
    Plot bar comparison of peak helicity values for all patients.

    Args:
        ax: Matplotlib axes object
        patient_ids: List of patient identifiers
        helicity_data: Dictionary containing helicity data for all patients
        patient_colors: Dictionary mapping patient IDs to colors
        title: Plot title
        ylabel: Y-axis label
    """
    peak_values = {}
    for patient_id in patient_ids:
        if helicity_data.get(patient_id):
            peak_values[patient_id] = max(helicity_data[patient_id].values())
        else:
            peak_values[patient_id] = 0

    patient_names = list(peak_values.keys())
    values = list(peak_values.values())
    colors = [patient_colors[pid] for pid in patient_names]

    ax.bar(patient_names, values, color=colors, alpha=0.7, edgecolor="black")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", labelsize=10)


def calculate_helicity_statistics(
    patient_ids: list[str],
    all_max_helicity: dict,
    all_avg_helicity: dict,
    all_abs_avg_helicity: dict,
) -> dict[str, dict]:
    """
    Calculate comprehensive statistics for each patient.

    Args:
        patient_ids: List of patient identifiers
        all_max_helicity: Dictionary containing max helicity data
        all_avg_helicity: Dictionary containing avg helicity data
        all_abs_avg_helicity: Dictionary containing abs avg helicity data

    Returns:
        Dictionary containing statistics for each patient
    """
    patient_stats = {}

    for patient_id in patient_ids:
        if patient_id not in all_max_helicity or not all_max_helicity[patient_id]:
            patient_stats[patient_id] = None
            continue

        max_hel_values = list(all_max_helicity[patient_id].values())
        avg_hel_values = list(all_avg_helicity[patient_id].values())
        abs_avg_hel_values = list(all_abs_avg_helicity[patient_id].values())

        # Find peak timestep
        peak_ts = max(
            all_max_helicity[patient_id], key=all_max_helicity[patient_id].get
        )

        patient_stats[patient_id] = {
            "peak_max_helicity": max(max_hel_values),
            "mean_max_helicity": np.mean(max_hel_values),
            "min_max_helicity": min(max_hel_values),
            "peak_avg_helicity": max(avg_hel_values),
            "min_avg_helicity": min(avg_hel_values),
            "mean_avg_helicity": np.mean(avg_hel_values),
            "peak_abs_avg_helicity": max(abs_avg_hel_values),
            "mean_abs_avg_helicity": np.mean(abs_avg_hel_values),
            "peak_timestep": peak_ts,
        }

    return patient_stats


def calculate_helicity_correlations(
    patient_id: str,
    all_max_helicity: dict,
    all_flow_rates: dict | None = None,
    all_max_vorticity: dict | None = None,
) -> dict[str, float]:
    """
    Calculate correlations between helicity and other parameters for a patient.

    Args:
        patient_id: Patient identifier
        all_max_helicity: Dictionary containing max helicity data
        all_flow_rates: Optional dictionary containing flow rate data
        all_max_vorticity: Optional dictionary containing max vorticity data

    Returns:
        Dictionary containing correlation coefficients
    """
    correlations = {}

    if patient_id not in all_max_helicity:
        return correlations

    patient_max_helicity = list(all_max_helicity[patient_id].values())

    # Flow rate correlation
    if all_flow_rates and patient_id in all_flow_rates:
        patient_flow_rates = list(all_flow_rates[patient_id].values())
        if len(patient_flow_rates) == len(patient_max_helicity):
            corr = np.corrcoef(patient_flow_rates, patient_max_helicity)[0, 1]
            correlations["flow_rate"] = corr

    # Vorticity correlation
    if all_max_vorticity and patient_id in all_max_vorticity:
        patient_max_vorticity = list(all_max_vorticity[patient_id].values())
        if len(patient_max_vorticity) == len(patient_max_helicity):
            corr = np.corrcoef(patient_max_vorticity, patient_max_helicity)[0, 1]
            correlations["vorticity"] = corr

    return correlations


def print_helicity_summary(
    patient_ids: list[str],
    patient_stats: dict,
    all_max_helicity: dict,
    all_flow_rates: dict | None = None,
    all_max_vorticity: dict | None = None,
) -> None:
    """
    Print comprehensive helicity analysis summary for all patients.

    Args:
        patient_ids: List of patient identifiers
        patient_stats: Dictionary containing patient statistics
        all_max_helicity: Dictionary containing max helicity data
        all_flow_rates: Optional dictionary containing flow rate data
        all_max_vorticity: Optional dictionary containing max vorticity data
    """
    print("\nMULTI-PATIENT HELICITY ANALYSIS SUMMARY")
    print("=" * 50)

    for patient_id in patient_ids:
        if patient_stats.get(patient_id) is None:
            print(f"\nPatient {patient_id}: No data available")
            continue

        stats = patient_stats[patient_id]
        print(f"\nPatient {patient_id}:")
        print("-" * 30)

        print(f"Peak maximum helicity: {stats['peak_max_helicity']:.1f} mm²/s²")
        print(f"Mean maximum helicity: {stats['mean_max_helicity']:.1f} mm²/s²")
        print(f"Minimum maximum helicity: {stats['min_max_helicity']:.1f} mm²/s²")
        print(f"Peak occurs at timestep: {stats['peak_timestep']}")
        print(f"Peak average helicity: {stats['peak_avg_helicity']:.1f} mm²/s²")
        print(f"Minimum average helicity: {stats['min_avg_helicity']:.1f} mm²/s²")
        print(f"Mean average helicity: {stats['mean_avg_helicity']:.1f} mm²/s²")
        print(
            f"Peak average absolute helicity: {stats['peak_abs_avg_helicity']:.1f} mm²/s²"
        )
        print(
            f"Mean average absolute helicity: {stats['mean_abs_avg_helicity']:.1f} mm²/s²"
        )

        # Print correlations
        correlations = calculate_helicity_correlations(
            patient_id, all_max_helicity, all_flow_rates, all_max_vorticity
        )

        if "flow_rate" in correlations:
            corr = correlations["flow_rate"]
            print(f"Flow rate correlation (max helicity): {corr:.3f}")

            if abs(corr) > 0.7:
                strength = "strong"
            elif abs(corr) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            print(f"Flow rate correlation strength: {strength}")

        if "vorticity" in correlations:
            corr = correlations["vorticity"]
            print(f"Vorticity correlation (max helicity): {corr:.3f}")
