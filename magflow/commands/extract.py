from pathlib import Path

import typer
from typing_extensions import Annotated
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from magflow.utils.logger import logger
from magflow.utils.dicom import process_file

app = typer.Typer()


@app.command("extract")
def extract(
    pathname: Annotated[Path, typer.Argument()],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = Path(".tmp"),
):
    """
    Load dicom image series from a directory and organize by timesteps and axes.

    This function processes DICOM files, extracting individual frames from multi-frame files,
    and organizes them in a directory structure based on cardiac timestep and anatomical axis.
    """
    files = [
        f for f in pathname.iterdir() if f.is_file() and f.suffix.lower() == ".dcm"
    ]
    total_frames = 0

    # Use rich.progress to show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        # Create a task for file processing
        file_task = progress.add_task(
            "[green]Processing DICOM files...", total=len(files)
        )

        # Process each file in the directory
        for filename in files:
            frames_processed = process_file(filename, output_dir, progress, file_task)
            total_frames += frames_processed

    logger.info(
        f"DICOM processing complete: {total_frames} frames organized in {output_dir}"
    )
