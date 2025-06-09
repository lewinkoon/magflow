import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from magflow.utils.visualization import extract_aorta


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


def plot_osi_analysis(patient_ids, all_osi_values, patient_colors):
    """
    Create comprehensive OSI visualization for all patients.

    Parameters:
    -----------
    patient_ids : list
        List of patient identifiers
    all_osi_values : dict
        Dictionary containing OSI values for each patient
    patient_colors : dict
        Dictionary mapping patient IDs to colors

    Returns:
    --------
    tuple
        (fig, axes) matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(2, len(patient_ids), figsize=(6 * len(patient_ids), 10))
    if len(patient_ids) == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Multi-Patient OSI Analysis", fontsize=16, fontweight="bold")

    for idx, patient_id in enumerate(patient_ids):
        if patient_id in all_osi_values:
            osi_values = all_osi_values[patient_id]

            # Histogram of OSI values
            axes[0, idx].hist(
                osi_values,
                bins=50,
                color=patient_colors[patient_id],
                alpha=0.7,
                edgecolor="black",
            )
            axes[0, idx].set_title(f"{patient_id} - OSI Distribution")
            axes[0, idx].set_xlabel("OSI")
            axes[0, idx].set_ylabel("Frequency")
            axes[0, idx].grid(True, alpha=0.3)

            # Box plot of OSI values
            axes[1, idx].boxplot(
                osi_values,
                patch_artist=True,
                boxprops={"facecolor": patient_colors[patient_id], "alpha": 0.7},
            )
            axes[1, idx].set_title(f"{patient_id} - OSI Statistics")
            axes[1, idx].set_ylabel("OSI")
            axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def calculate_osi_statistics(patient_ids, all_osi_values):
    """
    Calculate comprehensive OSI statistics for all patients.

    Parameters:
    -----------
    patient_ids : list
        List of patient identifiers
    all_osi_values : dict
        Dictionary containing OSI values for each patient

    Returns:
    --------
    tuple
        (patient_stats, cross_patient_stats) dictionaries with statistics
    """
    patient_stats = {}

    for patient_id in patient_ids:
        if patient_id in all_osi_values:
            osi_values = all_osi_values[patient_id]
            patient_stats[patient_id] = {
                "mean": np.mean(osi_values),
                "median": np.median(osi_values),
                "max": np.max(osi_values),
                "min_nonzero": np.min(osi_values[osi_values > 0]),
                "high_osi_count": np.sum(osi_values > 0.2),
                "high_osi_percentage": np.sum(osi_values > 0.2) / len(osi_values) * 100,
                "total_points": len(osi_values),
            }

    # Cross-patient comparison
    all_means = [
        patient_stats[pid]["mean"] for pid in patient_ids if pid in patient_stats
    ]
    all_maxes = [
        patient_stats[pid]["max"] for pid in patient_ids if pid in patient_stats
    ]

    cross_patient_stats = {
        "mean_range": (min(all_means), max(all_means)) if all_means else (0, 0),
        "max_range": (min(all_maxes), max(all_maxes)) if all_maxes else (0, 0),
    }

    return patient_stats, cross_patient_stats


def print_osi_summary(patient_stats, cross_patient_stats):
    """
    Print comprehensive OSI analysis summary.

    Parameters:
    -----------
    patient_stats : dict
        Dictionary containing statistics for each patient
    cross_patient_stats : dict
        Dictionary containing cross-patient comparison statistics
    """
    print("\nOSI ANALYSIS SUMMARY")
    print("-" * 50)

    for patient_id, stats in patient_stats.items():
        print(f"\nPatient {patient_id}:")
        print(f"  Mean OSI: {stats['mean']:.4f}")
        print(f"  Median OSI: {stats['median']:.4f}")
        print(f"  Max OSI: {stats['max']:.4f}")
        print(f"  Min non-zero OSI: {stats['min_nonzero']:.4f}")
        print(
            f"  Points with OSI > 0.2: {stats['high_osi_count']} ({stats['high_osi_percentage']:.1f}%)"
        )

    print("\nCROSS-PATIENT COMPARISON")
    print("-" * 30)
    mean_min, mean_max = cross_patient_stats["mean_range"]
    max_min, max_max = cross_patient_stats["max_range"]
    print(f"Overall mean OSI range: {mean_min:.4f} - {mean_max:.4f}")
    print(f"Overall max OSI range: {max_min:.4f} - {max_max:.4f}")
