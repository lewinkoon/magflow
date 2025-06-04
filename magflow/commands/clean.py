import shutil
from pathlib import Path

import typer

from magflow.utils.logger import logger

app = typer.Typer()


@app.command("clean")
def clean(
    clean_all: bool = typer.Option(
        False, "--all", help="Clean both files and output directories if provided"
    ),
):
    """
    Remove exported data.
    """
    paths = ["output", ".tmp"] if clean_all else ["output"]

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            logger.info(f"{path_str} files not exported yet.")
            continue  # Skipping nonexistent directories
        else:
            try:
                shutil.rmtree(path)  # Recursively delete directory and its contents
                logger.info(f"Removed directory {path_str} and all its contents")
            except Exception as e:
                logger.error(f"{path_str} could not be removed: {e}")
