from streamlit.web import cli
import os
import sys
import typer
from typing_extensions import Annotated

app = typer.Typer()

@app.command("serve")
def serve(port: Annotated[int, typer.Option(help="Port to run the server on.")] = 8501):
    """
    Start the web interface using streamlit.
    """
    # Get the commands directory
    commands_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up to get the package root directory
    package_root = os.path.dirname(commands_dir)
    # Path to ui.py in the package root
    ui_path = os.path.join(package_root, "ui.py")

    # Run streamlit with the path to ui.py
    cli.main_run(["C:/Users/Luis/Repositories/magflow/magflow/ui.py", "--server.port", str(port)])