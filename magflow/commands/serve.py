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

    # Run streamlit with the path to ui.py
    cli.main_run(
        [
            r"C:/Users/Luis/Repositories/magflow/magflow/ui.py",
            "--server.port",
            str(port),
        ]
    )
