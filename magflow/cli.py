import typer

from .commands.build import app as build_app
from .commands.check import app as check_app
from .commands.clean import app as clean_app
from .commands.load import app as load_app
from .commands.patch import app as patch_app

# Initialize the main app
app = typer.Typer(
    help="Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field."
)

# Add commands
app.add_typer(build_app)
app.add_typer(check_app)
app.add_typer(clean_app)
app.add_typer(load_app)
app.add_typer(patch_app)


if __name__ == "__main__":
    app()
