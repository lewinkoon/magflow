import os
import shutil  # Added import for recursive deletion
import typer
from magflow.logger import logger

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
    paths = ["output", "files"] if clean_all else ["output"]

    for path in paths:
        if not os.path.exists(path):
            logger.info(f"{path} files not exported yet.")
            continue  # Skipping nonexistent directories
        else:
            try:
                shutil.rmtree(path)  # Recursively delete directory and its contents
                logger.info(f"Removed directory {path} and all its contents")
            except Exception as e:
                logger.error(f"{path} could not be removed: {e}")
