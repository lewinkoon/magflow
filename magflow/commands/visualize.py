import json
import os
from pathlib import Path

import pyvista as pv
import typer
from typing_extensions import Annotated

from magflow.utils.vtk import (
    load_biomodel,
    extract_aorta,
    render_volume,
    generate_streamlines,
)


app = typer.Typer()


class VisualizationCallbacks:
    """Container class for visualization callbacks."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.actors = {}

    def add_actor(self, name, actor):
        """Store an actor reference for later manipulation."""
        self.actors[name] = actor

    def toggle_volume(self, flag):
        """Toggle visibility of the volume rendering."""
        if "volume" in self.actors:
            self.actors["volume"].SetVisibility(flag)
            self.plotter.update()

    def toggle_streamlines(self, flag):
        """Toggle visibility of the streamlines."""
        if "streamlines" in self.actors:
            self.actors["streamlines"].SetVisibility(flag)
            self.plotter.update()

    def sphere_callback(self, pos):
        """Print the current position of the sphere widget."""
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
    """Visualize VTK files showing velocity field over time with optional biomodel."""
    # Format filename and check if file exists
    filename = f"data.vts.{timestep}"
    filepath = data_dir / filename

    if not filepath.exists():
        typer.echo(f"Error: File {filepath} not found.")
        available_files = [f for f in os.listdir(data_dir) if f.startswith("data.vts.")]
        typer.echo(
            f"Available timesteps: {[int(f.split('.')[-1]) for f in available_files]}"
        )
        return

    # Create plotter
    plotter = pv.Plotter()

    # Initialize callbacks manager
    callbacks = VisualizationCallbacks(plotter)

    center = [130, 260, 40]

    # Load and process data
    dataset = pv.read(str(filepath), force_ext=".vts")
    biomodel_data = load_biomodel(biomodel_path)
    aorta = extract_aorta(dataset, biomodel_data)
    volume = render_volume(aorta)
    streamlines, source = generate_streamlines(aorta, center)

    # Add biomodel as wireframe
    plotter.add_mesh(
        biomodel_data,
        color="white",
        opacity=0.2,
        label="Biomodel",
        show_edges=True,
        edge_color="black",
        style="wireframe",
    )

    # Add volume rendering
    volume_plot = plotter.add_volume(
        volume,
        scalars="VelocityMagnitude",
        clim=[0, 150],
        opacity="linear",
        scalar_bar_args=dict(
            title="Velocity [cm/s]",
            n_labels=11,
            vertical=False,
        ),
        mapper="smart",
        cmap="rainbow",
        blending="composite",
    )
    typer.echo("Added volume rendering of velocity magnitude field")
    callbacks.add_actor("volume", volume_plot)
    plotter.add_checkbox_button_widget(
        callback=callbacks.toggle_volume,
        value=True,
        position=(10, 10),
        size=30,
        color_on="green",
        color_off="red",
    )

    # Add streamlines with velocity color mapping
    streamlines_actor = plotter.add_mesh(
        streamlines,
        line_width=1.0,
        label="Streamlines",
        scalars="VelocityMagnitude",
        clim=[0, 150],
    )
    typer.echo("Added streamlines colored by velocity")
    callbacks.add_actor("streamlines", streamlines_actor)

    streamlines_actor.SetVisibility(False)  # Hide streamlines by default

    # Add checkbox to toggle streamlines visibility (default off)
    plotter.add_checkbox_button_widget(
        callback=callbacks.toggle_streamlines,
        value=False,
        position=(10, 50),
        size=30,
        color_on="green",
        color_off="red",
    )
    typer.echo("Added streamlines toggle checkbox (default: off)")

    # Add source points (streamline seeds)
    source_actor = plotter.add_mesh(
        source,
        color="yellow",
        point_size=8,
        render_points_as_spheres=True,
        label="Seed Points",
    )
    typer.echo("Added streamline seed points")
    callbacks.add_actor("seeds", source_actor)

    # Configure view settings
    plotter.view_isometric()
    plotter.show_grid(font_size=12, n_xlabels=11, n_ylabels=11, n_zlabels=11)
    plotter.set_viewup([0, 1, 0])
    plotter.add_axes()

    # Configure window
    plotter.window_size = [1200, 900]
    plotter.title = f"Flow Visualization - Timestep {timestep}"
    plotter.show()
