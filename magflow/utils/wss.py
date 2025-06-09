from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

import magflow.utils.visualization as viz


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


def calculate_patient_wss(
    patient_id: str, patient_data: dict[str, Any], mu: float = 0.004
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
    """
    Calculate WSS for a single patient across all timesteps.

    Args:
        patient_id: Patient identifier
        patient_data: Dictionary containing patient timesteps and biomodel
        mu: Blood viscosity in Pa·s

    Returns:
        Tuple of (wss_values, max_wss_values, avg_wss_values) dictionaries
    """

    patient_timesteps = patient_data["timesteps"]
    patient_biomodel = patient_data["biomodel"]

    # Extract wall points and compute normals
    patient_biomodel.compute_normals(
        cell_normals=False, point_normals=True, inplace=True
    )
    wall_points = np.array(patient_biomodel.points)
    wall_normals = np.array(patient_biomodel["Normals"])

    # Initialize result dictionaries
    wss_values = {}
    max_wss_values = {}
    avg_wss_values = {}

    # Process each timestep
    for ts in patient_timesteps:
        dataset = patient_timesteps[ts]
        aorta = viz.extract_aorta(dataset, patient_biomodel)

        # Calculate WSS for this timestep
        wss = calculate_wss_timestep(wall_points, wall_normals, aorta, mu)

        # Store results
        wss_values[ts] = wss
        max_wss_values[ts] = np.max(wss)
        avg_wss_values[ts] = np.mean(wss)

    return wss_values, max_wss_values, avg_wss_values


def calculate_multi_patient_wss(
    patient_ids: list[str],
    all_patient_data: dict[str, dict[str, Any]],
    mu: float = 0.004,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """
    Calculate WSS for multiple patients.

    Args:
        patient_ids: List of patient identifiers
        all_patient_data: Dictionary containing all patient data
        mu: Blood viscosity in Pa·s

    Returns:
        Tuple of (all_wss_values, all_max_wss_values, all_avg_wss_values)
    """
    print("Calculating Wall Shear Stress for all patients...")

    all_wss_values = {}
    all_max_wss_values = {}
    all_avg_wss_values = {}

    for patient_id in patient_ids:
        if patient_id not in all_patient_data:
            print(f"Warning: No data found for patient {patient_id}")
            continue

        wss_vals, max_wss_vals, avg_wss_vals = calculate_patient_wss(
            patient_id, all_patient_data[patient_id], mu
        )

        all_wss_values[patient_id] = wss_vals
        all_max_wss_values[patient_id] = max_wss_vals
        all_avg_wss_values[patient_id] = avg_wss_vals

    print(f"WSS calculation completed for {len(all_wss_values)} patients")
    return all_wss_values, all_max_wss_values, all_avg_wss_values


def plot_wss_time_series(
    ax: plt.Axes,
    patient_ids: list[str],
    wss_data: dict[str, dict],
    patient_colors: dict[str, str],
    title: str,
    ylabel: str,
    wss_type: str = "max",
) -> None:
    """
    Plot WSS time series for multiple patients.

    Args:
        ax: Matplotlib axes object
        patient_ids: List of patient identifiers
        wss_data: WSS data dictionary
        patient_colors: Patient color mapping
        title: Plot title
        ylabel: Y-axis label
        wss_type: Type of WSS data ('max' or 'avg')
    """
    for patient_id in patient_ids:
        if patient_id in wss_data:
            patient_timesteps = sorted(wss_data[patient_id].keys())
            wss_values = [wss_data[patient_id][ts] for ts in patient_timesteps]

            ax.plot(
                patient_timesteps,
                wss_values,
                "o-",
                linewidth=2,
                markersize=4,
                color=patient_colors[patient_id],
                label=f"Patient {patient_id}",
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_wss_comparison_bars(
    ax: plt.Axes,
    patient_ids: list[str],
    wss_data: dict[str, dict],
    patient_colors: dict[str, str],
    title: str,
    ylabel: str,
    comparison_type: str = "peak",
) -> None:
    """
    Plot WSS comparison bar chart.

    Args:
        ax: Matplotlib axes object
        patient_ids: List of patient identifiers
        wss_data: WSS data dictionary
        patient_colors: Patient color mapping
        title: Plot title
        ylabel: Y-axis label
        comparison_type: Type of comparison ('peak' or 'average')
    """
    patient_values = []
    patient_labels = []

    for patient_id in patient_ids:
        if patient_id in wss_data:
            if comparison_type == "peak":
                value = max(wss_data[patient_id].values())
            else:  # average
                value = np.mean(list(wss_data[patient_id].values()))

            patient_values.append(value)
            patient_labels.append(patient_id)

    bars = ax.bar(
        patient_labels,
        patient_values,
        color=[patient_colors[p] for p in patient_labels],
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Patient")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars, patient_values, strict=False):
        offset = 0.05 if comparison_type == "peak" else 0.02
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def create_wss_visualization(
    patient_ids: list[str],
    all_max_wss_values: dict[str, dict],
    all_avg_wss_values: dict[str, dict],
    patient_colors: dict[str, str],
) -> None:
    """
    Create comprehensive multi-patient WSS visualization.

    Args:
        patient_ids: List of patient identifiers
        all_max_wss_values: Maximum WSS values for all patients
        all_avg_wss_values: Average WSS values for all patients
        patient_colors: Patient color mapping
    """
    # Main 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Multi-Patient Wall Shear Stress Analysis", fontsize=16, fontweight="bold"
    )

    # Plot 1: Maximum WSS time series
    plot_wss_time_series(
        axes[0, 0],
        patient_ids,
        all_max_wss_values,
        patient_colors,
        "Maximum Wall Shear Stress by Patient",
        "Maximum WSS (Pa)",
        "max",
    )

    # Plot 2: Average WSS time series
    plot_wss_time_series(
        axes[0, 1],
        patient_ids,
        all_avg_wss_values,
        patient_colors,
        "Average WSS by Patient",
        "Average WSS (Pa)",
        "avg",
    )

    # Plot 3: Peak maximum WSS comparison
    plot_wss_comparison_bars(
        axes[1, 0],
        patient_ids,
        all_max_wss_values,
        patient_colors,
        "Peak Maximum WSS Comparison",
        "Peak Maximum WSS (Pa)",
        "peak",
    )

    # Plot 4: Average maximum WSS comparison
    plot_wss_comparison_bars(
        axes[1, 1],
        patient_ids,
        all_max_wss_values,
        patient_colors,
        "Average Maximum WSS Comparison",
        "Average Maximum WSS (Pa)",
        "average",
    )

    plt.tight_layout()
    plt.show()

    # Create overlay comparison plot
    create_wss_overlay_plots(
        patient_ids, all_max_wss_values, all_avg_wss_values, patient_colors
    )


def create_wss_overlay_plots(
    patient_ids: list[str],
    all_max_wss_values: dict[str, dict],
    all_avg_wss_values: dict[str, dict],
    patient_colors: dict[str, str],
) -> None:
    """
    Create overlay comparison plots for better visualization.

    Args:
        patient_ids: List of patient identifiers
        all_max_wss_values: Maximum WSS values for all patients
        all_avg_wss_values: Average WSS values for all patients
        patient_colors: Patient color mapping
    """
    plt.figure(figsize=(14, 10))

    # Maximum WSS overlay
    plt.subplot(2, 1, 1)
    for patient_id in patient_ids:
        if patient_id in all_max_wss_values:
            patient_timesteps = sorted(all_max_wss_values[patient_id].keys())
            max_wss = [all_max_wss_values[patient_id][ts] for ts in patient_timesteps]

            plt.plot(
                patient_timesteps,
                max_wss,
                "o-",
                linewidth=2,
                markersize=4,
                color=patient_colors[patient_id],
                label=f"Patient {patient_id}",
            )

    plt.title(
        "Maximum Wall Shear Stress - All Patients", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Maximum WSS (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Average WSS overlay
    plt.subplot(2, 1, 2)
    for patient_id in patient_ids:
        if patient_id in all_avg_wss_values:
            patient_timesteps = sorted(all_avg_wss_values[patient_id].keys())
            avg_wss = [all_avg_wss_values[patient_id][ts] for ts in patient_timesteps]

            plt.plot(
                patient_timesteps,
                avg_wss,
                "o-",
                linewidth=2,
                markersize=4,
                color=patient_colors[patient_id],
                label=f"Patient {patient_id}",
            )

    plt.title("Average WSS - All Patients", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Average WSS (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_clinical_risk_assessment(peak_max_wss: float) -> str:
    """
    Get clinical risk assessment based on peak maximum WSS.

    Args:
        peak_max_wss: Peak maximum WSS value in Pa

    Returns:
        Clinical risk assessment string
    """
    if peak_max_wss < 0.4:
        return "LOW (potential for atherogenesis)"
    elif peak_max_wss <= 1.5:
        return "NORMAL (physiological range)"
    else:
        return "HIGH (potential for stenosis/turbulence)"


def calculate_wss_statistics(
    patient_ids: list[str],
    all_max_wss_values: dict[str, dict],
    all_avg_wss_values: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, float]]:
    """
    Calculate comprehensive WSS statistics.

    Args:
        patient_ids: List of patient identifiers
        all_max_wss_values: Maximum WSS values for all patients
        all_avg_wss_values: Average WSS values for all patients

    Returns:
        Tuple of (patient_statistics, cross_patient_statistics)
    """
    patient_stats = {}
    all_patient_peak_max = []
    all_patient_mean_max = []
    all_patient_peak_avg = []
    all_patient_mean_avg = []

    for patient_id in patient_ids:
        if patient_id not in all_max_wss_values:
            continue

        max_wss_vals = list(all_max_wss_values[patient_id].values())
        avg_wss_vals = list(all_avg_wss_values[patient_id].values())

        if max_wss_vals and avg_wss_vals:
            peak_max_wss = max(max_wss_vals)
            mean_max_wss = np.mean(max_wss_vals)
            peak_avg_wss = max(avg_wss_vals)
            mean_avg_wss = np.mean(avg_wss_vals)
            std_max_wss = np.std(max_wss_vals)
            std_avg_wss = np.std(avg_wss_vals)

            # Find timesteps with peak values
            peak_max_ts = max(
                all_max_wss_values[patient_id], key=all_max_wss_values[patient_id].get
            )
            peak_avg_ts = max(
                all_avg_wss_values[patient_id], key=all_avg_wss_values[patient_id].get
            )

            patient_stats[patient_id] = {
                "peak_max_wss": peak_max_wss,
                "mean_max_wss": mean_max_wss,
                "peak_avg_wss": peak_avg_wss,
                "mean_avg_wss": mean_avg_wss,
                "std_max_wss": std_max_wss,
                "std_avg_wss": std_avg_wss,
                "peak_max_ts": peak_max_ts,
                "peak_avg_ts": peak_avg_ts,
                "num_timesteps": len(max_wss_vals),
                "clinical_assessment": get_clinical_risk_assessment(peak_max_wss),
            }

            # Collect for cross-patient analysis
            all_patient_peak_max.append(peak_max_wss)
            all_patient_mean_max.append(mean_max_wss)
            all_patient_peak_avg.append(peak_avg_wss)
            all_patient_mean_avg.append(mean_avg_wss)

    # Cross-patient statistics
    cross_patient_stats = {}
    if all_patient_peak_max:
        cv_peak_max = np.std(all_patient_peak_max) / np.mean(all_patient_peak_max) * 100
        cv_mean_avg = np.std(all_patient_mean_avg) / np.mean(all_patient_mean_avg) * 100

        cross_patient_stats = {
            "highest_peak_max": max(all_patient_peak_max),
            "lowest_peak_max": min(all_patient_peak_max),
            "avg_peak_max": np.mean(all_patient_peak_max),
            "std_peak_max": np.std(all_patient_peak_max),
            "avg_mean_wss": np.mean(all_patient_mean_avg),
            "std_mean_avg": np.std(all_patient_mean_avg),
            "cv_peak_max": cv_peak_max,
            "cv_mean_avg": cv_mean_avg,
        }

    return patient_stats, cross_patient_stats


def print_wss_summary(
    patient_stats: dict[str, dict], cross_patient_stats: dict[str, float]
) -> None:
    """
    Print comprehensive WSS analysis summary.

    Args:
        patient_stats: Patient-specific statistics
        cross_patient_stats: Cross-patient statistics
    """
    print("\nMULTI-PATIENT WALL SHEAR STRESS ANALYSIS SUMMARY")
    print("=" * 60)

    for patient_id, stats in patient_stats.items():
        print(f"\nPatient {patient_id}:")
        print(
            f"  Peak maximum WSS: {stats['peak_max_wss']:.3f} Pa (at timestep {stats['peak_max_ts']})"
        )
        print(
            f"  Mean maximum WSS: {stats['mean_max_wss']:.3f} Pa (±{stats['std_max_wss']:.3f})"
        )
        print(
            f"  Peak average WSS: {stats['peak_avg_wss']:.3f} Pa (at timestep {stats['peak_avg_ts']})"
        )
        print(
            f"  Mean average WSS: {stats['mean_avg_wss']:.3f} Pa (±{stats['std_avg_wss']:.3f})"
        )
        print(f"  Number of timesteps: {stats['num_timesteps']}")
        print(f"  Clinical assessment: {stats['clinical_assessment']}")

    if cross_patient_stats:
        print("\nCROSS-PATIENT WSS COMPARISON")
        print("-" * 40)
        print(
            f"Highest peak maximum WSS: {cross_patient_stats['highest_peak_max']:.3f} Pa"
        )
        print(
            f"Lowest peak maximum WSS: {cross_patient_stats['lowest_peak_max']:.3f} Pa"
        )
        print(f"Average peak maximum WSS: {cross_patient_stats['avg_peak_max']:.3f} Pa")
        print(
            f"Standard deviation (peak max): {cross_patient_stats['std_peak_max']:.3f} Pa"
        )
        print(
            f"Average mean WSS across patients: {cross_patient_stats['avg_mean_wss']:.3f} Pa"
        )
        print(
            f"Standard deviation (mean avg): {cross_patient_stats['std_mean_avg']:.3f} Pa"
        )

        print("\nInter-patient variability:")
        print(f"  Peak maximum WSS CV: {cross_patient_stats['cv_peak_max']:.1f}%")
        print(f"  Mean average WSS CV: {cross_patient_stats['cv_mean_avg']:.1f}%")
