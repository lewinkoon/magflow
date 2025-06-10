from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

import magflow.utils.dicom as dcm
from magflow.utils.logger import logger

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

    This function processes DICOM files recursively, extracting individual frames from multi-frame files,
    and organizes them in a directory structure based on cardiac timestep and anatomical axis.
    """
    # Collect DICOM files recursively, only .dcm extension
    files = [f for f in pathname.rglob("*") if f.is_file()]

    if not files:
        logger.warning(f"No files found in {pathname}")
        return

    total_frames = 0
    failed_files = 0

    # Use rich.progress to show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold green]{task.completed}/{task.total} files"),
    ) as progress:
        # Create a task for file processing
        file_task = progress.add_task(
            "[green]Processing DICOM files...", total=len(files)
        )

        # Process each file
        for filename in files:
            try:
                frames_processed = dcm.process_dicom(
                    filename, output_dir, progress, file_task
                )
                total_frames += frames_processed

                if frames_processed == 0:
                    failed_files += 1

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                failed_files += 1
                progress.update(file_task, advance=1)

    logger.info(
        f"DICOM processing complete: {total_frames} frames from {len(files) - failed_files} files organized in {output_dir}"
    )

    if total_frames == 0:
        logger.error("No frames were successfully processed")
        raise typer.Exit(code=1)
