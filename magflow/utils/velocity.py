import numpy as np

import magflow.utils.visualization as viz


def calculate_velocities(patient_id, patient_data):
    """
    Calculate maximum and mean velocities for a single patient across all timesteps.

    Parameters:
    -----------
    patient_id : str
        Patient identifier
    patient_data : dict
        Dictionary containing patient timesteps and biomodel data

    Returns:
    --------
    tuple
        (max_velocities_dict, mean_velocities_dict) for the patient
    """
    patient_timesteps = patient_data["timesteps"]
    patient_biomodel = patient_data["biomodel"]

    max_velocities = {}
    mean_velocities = {}

    # Process each timestep for this patient
    for ts in patient_timesteps:
        dataset = patient_timesteps[ts]
        aorta = viz.extract_aorta(dataset, patient_biomodel)

        # Get velocity magnitude for the entire aorta volume
        velocity_magnitude = (
            aorta["VelocityMagnitude"] * 10
        )  # Convert from cm/s to mm/s

        # Store the maximum and mean velocity for this timestep
        if len(velocity_magnitude) > 0:
            max_velocities[ts] = np.max(velocity_magnitude)
            mean_velocities[ts] = np.mean(velocity_magnitude)
        else:
            max_velocities[ts] = 0
            mean_velocities[ts] = 0

    return max_velocities, mean_velocities


def plot_velocity_time_series(
    ax, patient_ids, velocity_data, patient_colors, title, ylabel, plot_type="max"
):
    """
    Plot velocity time series for all patients on a given axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    patient_ids : list
        List of patient identifiers
    velocity_data : dict
        Velocity data dictionary
    patient_colors : dict
        Color mapping for patients
    title : str
        Plot title
    ylabel : str
        Y-axis label
    plot_type : str
        Type of plot for legend labeling
    """
    for patient_id in patient_ids:
        if patient_id in velocity_data:
            patient_timesteps = sorted(velocity_data[patient_id].keys())
            velocities = [velocity_data[patient_id][ts] for ts in patient_timesteps]

            ax.plot(
                patient_timesteps,
                velocities,
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


def plot_velocity_comparison_bars(
    ax,
    patient_ids,
    velocity_data,
    patient_colors,
    title,
    ylabel,
    comparison_type="peak",
):
    """
    Plot bar comparison of velocities across patients.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    patient_ids : list
        List of patient identifiers
    velocity_data : dict
        Velocity data dictionary
    patient_colors : dict
        Color mapping for patients
    title : str
        Plot title
    ylabel : str
        Y-axis label
    comparison_type : str
        Type of comparison ("peak" or "average")
    """
    patient_values = []
    patient_labels = []

    for patient_id in patient_ids:
        if patient_id in velocity_data:
            if comparison_type == "peak":
                value = max(velocity_data[patient_id].values())
            else:  # average
                value = np.mean(list(velocity_data[patient_id].values()))

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
    label_offset = 5 if comparison_type == "peak" else 2
    for bar, value in zip(bars, patient_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def calculate_velocity_statistics(patient_ids, all_max_velocities, all_mean_velocities):
    """
    Calculate comprehensive velocity statistics for all patients.

    Parameters:
    -----------
    patient_ids : list
        List of patient identifiers
    all_max_velocities : dict
        Maximum velocities data
    all_mean_velocities : dict
        Mean velocities data

    Returns:
    --------
    dict
        Dictionary containing calculated statistics
    """
    patient_stats = {}
    cross_patient_stats = {
        "peak_max": [],
        "avg_max": [],
        "peak_mean": [],
        "avg_mean": [],
    }

    # Calculate per-patient statistics
    for patient_id in patient_ids:
        if patient_id not in all_max_velocities:
            continue

        max_vels = list(all_max_velocities[patient_id].values())
        mean_vels = list(all_mean_velocities[patient_id].values())

        if max_vels and mean_vels:
            peak_max_vel = max(max_vels)
            avg_max_vel = np.mean(max_vels)
            peak_mean_vel = max(mean_vels)
            avg_mean_vel = np.mean(mean_vels)

            # Find timesteps with peak values
            peak_max_ts = max(
                all_max_velocities[patient_id], key=all_max_velocities[patient_id].get
            )
            peak_mean_ts = max(
                all_mean_velocities[patient_id], key=all_mean_velocities[patient_id].get
            )

            patient_stats[patient_id] = {
                "peak_max_vel": peak_max_vel,
                "avg_max_vel": avg_max_vel,
                "peak_mean_vel": peak_mean_vel,
                "avg_mean_vel": avg_mean_vel,
                "peak_max_ts": peak_max_ts,
                "peak_mean_ts": peak_mean_ts,
                "num_timesteps": len(max_vels),
            }

            # Collect for cross-patient analysis
            cross_patient_stats["peak_max"].append(peak_max_vel)
            cross_patient_stats["avg_max"].append(avg_max_vel)
            cross_patient_stats["peak_mean"].append(peak_mean_vel)
            cross_patient_stats["avg_mean"].append(avg_mean_vel)

    # Calculate cross-patient statistics
    if cross_patient_stats["peak_max"]:
        cross_patient_stats.update(
            {
                "highest_peak_max": max(cross_patient_stats["peak_max"]),
                "lowest_peak_max": min(cross_patient_stats["peak_max"]),
                "mean_peak_max": np.mean(cross_patient_stats["peak_max"]),
                "std_peak_max": np.std(cross_patient_stats["peak_max"]),
                "mean_avg_mean": np.mean(cross_patient_stats["avg_mean"]),
                "std_avg_mean": np.std(cross_patient_stats["avg_mean"]),
            }
        )

    return patient_stats, cross_patient_stats


def print_velocity_summary(patient_stats, cross_patient_stats):
    """
    Print comprehensive velocity analysis summary.

    Parameters:
    -----------
    patient_stats : dict
        Per-patient statistics
    cross_patient_stats : dict
        Cross-patient statistics
    """
    print("\nMULTI-PATIENT VELOCITY ANALYSIS SUMMARY")
    print("=" * 50)

    for patient_id, stats in patient_stats.items():
        print(f"\nPatient {patient_id}:")
        print(
            f"  Peak maximum velocity: {stats['peak_max_vel']:.1f} mm/s "
            f"(at timestep {stats['peak_max_ts']})"
        )
        print(f"  Average maximum velocity: {stats['avg_max_vel']:.1f} mm/s")
        print(
            f"  Peak mean velocity: {stats['peak_mean_vel']:.1f} mm/s "
            f"(at timestep {stats['peak_mean_ts']})"
        )
        print(f"  Overall mean velocity: {stats['avg_mean_vel']:.1f} mm/s")
        print(f"  Number of timesteps: {stats['num_timesteps']}")

    if "highest_peak_max" in cross_patient_stats:
        print("\nCROSS-PATIENT COMPARISON")
        print("-" * 30)
        print(
            f"Highest peak maximum velocity: {cross_patient_stats['highest_peak_max']:.1f} mm/s"
        )
        print(
            f"Lowest peak maximum velocity: {cross_patient_stats['lowest_peak_max']:.1f} mm/s"
        )
        print(
            f"Average peak maximum velocity: {cross_patient_stats['mean_peak_max']:.1f} mm/s"
        )
        print(
            f"Standard deviation (peak max): {cross_patient_stats['std_peak_max']:.1f} mm/s"
        )
        print(
            f"Average mean velocity across patients: {cross_patient_stats['mean_avg_mean']:.1f} mm/s"
        )
        print(
            f"Standard deviation (avg mean): {cross_patient_stats['std_avg_mean']:.1f} mm/s"
        )
