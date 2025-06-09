import numpy as np
from matplotlib import pyplot as plt

import magflow.utils.visualization as viz


def get_cross_section_index(percentage, num_points):
    """
    Calculate cross-section index from percentage along centerline.

    Parameters:
    -----------
    percentage : float
        Percentage along centerline (0-100%)
        0% = first section, 50% = middle section, 100% = last section
    num_points : int
        Total number of centerline points

    Returns:
    --------
    int
        Cross-section index (0 to num_points-1)
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    return int((percentage / 100.0) * (num_points - 1))


def section_flowrate(cross_section):
    """
    Calculate flow rate for a single cross-section at one timestep.

    Parameters:
    -----------
    cross_section : pyvista.PolyData
        Cross-section mesh with velocity data
    velocity_conversion_factor : float
        Factor to convert velocity units (default: 10 for cm/s to mm/s)

    Returns:
    --------
    float
        Flow rate in ml/s
    """
    if cross_section is None or cross_section.n_points == 0:
        return 0.0

    # Convert velocity
    velocity_mm = cross_section["Velocity"] * 10  # Convert to mm/s

    # Compute normals if not present
    if "Normals" not in cross_section.point_data:
        cross_section.compute_normals(point_normals=True, inplace=True)

    # Get normals and ensure dimension consistency
    normals = cross_section["Normals"]
    min_length = min(len(velocity_mm), len(normals))
    velocity_mm = velocity_mm[:min_length]
    normals = normals[:min_length]

    # Calculate normal component of velocity
    normal_vel = np.sum(velocity_mm * normals, axis=1)

    # Calculate flow rate using area-weighted integration
    total_area = cross_section.area
    flow_rate = np.abs(np.mean(normal_vel) * total_area) / 1000  # Convert to ml/s

    return flow_rate


def plot_flow_rates(ax, patient_id, flow_rates, color=None):
    """
    Plot flow rates for a single patient on a given axis.
    """
    # Get timesteps and flow rates for this patient
    patient_timesteps = sorted(flow_rates.keys())
    flow_rates_values = [flow_rates[ts] for ts in patient_timesteps]

    # Filter out zero flow rates for cleaner visualization
    valid_data = [
        (ts, fr)
        for ts, fr in zip(patient_timesteps, flow_rates_values, strict=False)
        if fr > 0
    ]

    if valid_data:
        valid_timesteps, valid_flow_rates = zip(*valid_data, strict=False)

        # Individual patient plot
        ax.plot(
            valid_timesteps,
            valid_flow_rates,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
        )
        ax.set_title(f"Patient {patient_id}", fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Flow Rate (ml/s)")
        ax.grid(True, alpha=0.3)

        # Add statistics
        max_flow = max(valid_flow_rates)
        mean_flow = np.mean(valid_flow_rates)
        ax.text(
            0.05,
            0.95,
            f"Max: {max_flow:.1f} ml/s\nMean: {mean_flow:.1f} ml/s",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    return ax


def calculate_flow_rates(patient_id, timesteps, biomodel, centerline, point):
    """
    Calculate flow rates for all timesteps of a patient.

    Returns:
    --------
    dict: Dictionary of {timestep: flow_rate}
    """
    flow_rates = {}

    # Process each timestep for this patient
    for ts in timesteps:
        dataset = timesteps[ts]
        aorta = viz.extract_aorta(dataset, biomodel)

        # Generate cross-section at the specified point
        cross_section = viz.cross_section(
            point,
            centerline,
            aorta,
            radius=20,
        )

        flow_rate = section_flowrate(cross_section)
        flow_rates[ts] = flow_rate

    return flow_rates


def plot_comparison(ids, flow_rates, colors=None):
    # Create overlay comparison plot
    plt.figure(figsize=(12, 8))

    for patient_id in ids:
        if patient_id not in flow_rates:
            continue

        # Get timesteps and flow rates for this patient
        patient_timesteps = sorted(flow_rates[patient_id].keys())
        patient_flow_rates = [flow_rates[patient_id][ts] for ts in patient_timesteps]

        # Filter out zero flow rates
        valid_data = [
            (ts, fr)
            for ts, fr in zip(patient_timesteps, patient_flow_rates, strict=False)
            if fr > 0
        ]
        if valid_data:
            valid_timesteps, valid_flow_rates = zip(*valid_data, strict=False)

            plt.plot(
                valid_timesteps,
                valid_flow_rates,
                "o-",
                linewidth=2,
                markersize=4,
                color=colors[patient_id],
                label=f"Patient {patient_id}",
            )

    plt.title("Flow Rate Comparison - All Patients", fontsize=14, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Flow Rate (ml/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_flow_rate_statistics(flow_rates_dict):
    """
    Calculate comprehensive statistics for flow rates.

    Parameters:
    -----------
    flow_rates_dict : dict
        Dictionary of flow rates {patient_id: {timestep: flow_rate}}

    Returns:
    --------
    dict
        Statistics for each patient and cross-patient comparisons
    """
    patient_stats = {}

    for patient_id, patient_flow_rates in flow_rates_dict.items():
        # Filter out zero flow rates
        valid_flow_rates = [fr for fr in patient_flow_rates.values() if fr > 0]

        if valid_flow_rates:
            stats = {
                "max_flow": max(valid_flow_rates),
                "mean_flow": np.mean(valid_flow_rates),
                "min_flow": min(valid_flow_rates),
                "std_flow": np.std(valid_flow_rates),
                "num_timesteps": len(valid_flow_rates),
            }
            patient_stats[patient_id] = stats

    # Cross-patient statistics
    all_peak_flows = [stats["max_flow"] for stats in patient_stats.values()]
    all_mean_flows = [stats["mean_flow"] for stats in patient_stats.values()]

    cross_patient_stats = {}
    if all_peak_flows:
        cross_patient_stats = {
            "highest_peak_flow": max(all_peak_flows),
            "lowest_peak_flow": min(all_peak_flows),
            "average_peak_flow": np.mean(all_peak_flows),
            "average_mean_flow": np.mean(all_mean_flows),
            "peak_flow_std": np.std(all_peak_flows),
            "mean_flow_std": np.std(all_mean_flows),
        }

    return {"patient_stats": patient_stats, "cross_patient_stats": cross_patient_stats}
