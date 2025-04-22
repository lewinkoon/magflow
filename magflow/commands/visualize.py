import pyvista as pv
import typer


app = typer.Typer()


@app.command("visualize")
def visualize():
    """Visualize VTK files showing velocity field over time."""

    # Create plotter
    plotter = pv.Plotter()

    # Load first dataset to setup the scene
    dataset = pv.read(
        r"C:\Users\Luis\Repositories\magflow\assets\data.vts.0", force_ext=".vts"
    )

    plotter.add_mesh(dataset, scalars="Velocity", cmap="plasma")

    # Setup view
    plotter.view_isometric()
    plotter.show_axes()
    plotter.set_viewup([0, 1, 0])

    # Start the visualization
    plotter.show()
