import os
from pathlib import Path

import pyvista as pv
import typer
from typing_extensions import Annotated

from magflow.utils.vtk import (
    calculate_velocity_statistics,
    extract_aorta,
    generate_streamlines,
    load_biomodel,
    render_volume,
)

app = typer.Typer()


class VisualizationCallbacks:
    """Container class for visualization callbacks to manage interactive UI elements."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.actors = {}  # Dictionary to store actor references for UI control

    def add_actor(self, name, actor):
        """Store an actor reference for later manipulation and visibility control."""
        self.actors[name] = actor

    def toggle_volume(self, flag):
        """Toggle visibility of the volume rendering based on checkbox state."""
        if "volume" in self.actors:
            self.actors["volume"].SetVisibility(flag)
            self.plotter.update()

    def toggle_streamlines(self, flag):
        """Toggle visibility of the streamlines based on checkbox state."""
        if "streamlines" in self.actors:
            self.actors["streamlines"].SetVisibility(flag)
            self.plotter.update()

    def sphere_callback(self, pos):
        """Print the current position of the sphere widget (used for debugging)."""
        print(f"Sphere position: {pos}")


@app.command("visualize")
def visualize(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing the VTS files to visualize",
        ),
    ] = Path("data"),
    timestep: Annotated[
        int,
        typer.Option("--timestep", "-t", help="Timestep to visualize (e.g. 0, 28, 57)"),
    ] = 85,
    biomodel_path: Annotated[
        Path,
        typer.Option(
            "--biomodel",
            "-b",
            help="Path to the biomodel VTK file to include in the visualization",
        ),
    ] = Path("assets/biomodel.vtk"),
):
    """Visualize VTK files showing velocity field over time with optional biomodel.

    This function creates an interactive 3D visualization with volume rendering
    and streamlines that can be toggled on/off by the user.
    """
    # Construct the full filepath for the selected timestep data
    filename = f"data.vts.{timestep}"
    filepath = data_dir / filename

    # Validate that the requested timestep file exists
    if not filepath.exists():
        typer.echo(f"Error: File {filepath} not found.")
        # List available timesteps to help the user
        available_files = [f for f in os.listdir(data_dir) if f.startswith("data.vts.")]
        typer.echo(
            f"Available timesteps: {[int(f.split('.')[-1]) for f in available_files]}"
        )
        return

    # Initialize the PyVista plotter for 3D visualization
    plotter = pv.Plotter()

    # Create callback manager for interactive UI elements
    callbacks = VisualizationCallbacks(plotter)

    # Define center point for streamline generation (coordinates in the dataset space)
    center = [130, 260, 40]

    # Load and process the VTS and biomodel data
    dataset = pv.read(
        str(filepath), force_ext=".vts"
    )  # Load the VTS dataset for the selected timestep
    biomodel_data = load_biomodel(biomodel_path)  # Load anatomical model
    aorta = extract_aorta(dataset, biomodel_data)  # Extract aorta region from dataset

    # Display available data arrays for debugging and information
    print("Data arrays in aorta dataset:")
    for name in aorta.array_names:
        print(f"  - {name}")

    # Calculate and display velocity statistics for the current timestep
    velocity_stats = calculate_velocity_statistics(aorta)
    typer.echo("Velocity Statistics:")
    typer.echo(f"  Mean Velocity: {velocity_stats['mean']:.2f} cm/s")
    typer.echo(f"  Peak Velocity: {velocity_stats['peak']:.2f} cm/s")

    # Generate volume and streamline representations
    volume = render_volume(aorta)  # Create volume representation of the aorta
    streamlines, source = generate_streamlines(
        aorta, center
    )  # Generate flow streamlines and seed points

    # Add velocity statistics text overlay to the visualization
    plotter.add_text(
        f"Mean Velocity: {velocity_stats['mean']:.2f} cm/s\nPeak Velocity: {velocity_stats['peak']:.2f} cm/s",
        position="upper_left",
        font_size=12,
        shadow=True,
    )

    # Add biomodel as a wireframe to provide anatomical context
    plotter.add_mesh(
        biomodel_data,
        color="white",
        opacity=0.2,
        label="Biomodel",
        show_edges=True,
        edge_color="black",
        style="wireframe",
    )

    # Add volume rendering of velocity magnitude
    volume_plot = plotter.add_volume(
        volume,
        scalars="VelocityMagnitude",  # Color by velocity magnitude
        clim=[0, 150],  # Color range in cm/s
        opacity="linear",  # Linear opacity mapping
        scalar_bar_args=dict(
            title="Velocity [cm/s]",
            n_labels=11,
            vertical=False,
        ),
        mapper="smart",  # Smart volume mapper for better rendering
        cmap="rainbow",  # Color map for velocity visualization
        blending="composite",  # Composite blending mode for volume rendering
    )
    typer.echo("Added volume rendering of velocity magnitude field")
    callbacks.add_actor("volume", volume_plot)

    # Add toggle checkbox for volume rendering (default: visible)
    plotter.add_checkbox_button_widget(
        callback=callbacks.toggle_volume,
        value=True,  # Initially visible
        position=(10, 10),  # Position in the window (pixels from bottom-left)
        size=30,
        color_on="green",
        color_off="red",
    )

    # Add streamlines visualization colored by velocity
    streamlines_actor = plotter.add_mesh(
        streamlines,
        line_width=1.0,
        label="Streamlines",
        cmap="rainbow",  # Color map matching the volume rendering
        show_scalar_bar=False,  # Don't show duplicate scalar bar
    )
    typer.echo("Added streamlines with unique coloring per line")
    callbacks.add_actor("streamlines", streamlines_actor)

    # Initially hide streamlines (will be toggled by checkbox)
    streamlines_actor.SetVisibility(False)

    # Add checkbox to toggle streamlines visibility
    plotter.add_checkbox_button_widget(
        callback=callbacks.toggle_streamlines,
        value=False,  # Initially hidden
        position=(10, 50),  # Position above volume checkbox
        size=30,
        color_on="green",
        color_off="red",
    )
    typer.echo("Added streamlines toggle checkbox (default: off)")

    # Add visualization of streamline seed points
    source_actor = plotter.add_mesh(
        source,
        color="yellow",
        point_size=8,
        render_points_as_spheres=True,
        label="Seed Points",
    )
    typer.echo("Added streamline seed points")
    callbacks.add_actor("seeds", source_actor)

    # Configure camera and grid for optimal viewing
    plotter.view_isometric()  # Set isometric view
    plotter.show_grid(font_size=12, n_xlabels=11, n_ylabels=11, n_zlabels=11)
    plotter.set_viewup([0, 1, 0])  # Set up direction
    plotter.add_axes()  # Add coordinate axes

    # Set window properties and display the visualization
    plotter.window_size = [1200, 900]  # Set window dimensions
    plotter.title = f"Flow Visualization - Timestep {timestep}"
    plotter.show()
