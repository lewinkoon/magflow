import typer

from .commands.build import app as build_app
from .commands.check import app as check_app
from .commands.clean import app as clean_app
from .commands.export import app as export_app
from .commands.serve import app as serve_app
from .commands.visualize import app as visualize_app

# Initialize the main app
app = typer.Typer(
    help="Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field."
)

# Add commands
app.add_typer(build_app)
app.add_typer(check_app)
app.add_typer(clean_app)
app.add_typer(export_app)
app.add_typer(serve_app)
app.add_typer(visualize_app)


if __name__ == "__main__":
    app()
